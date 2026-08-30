from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from yenibot.config import load_config
from yenibot.data.validation import validate_full_kline_frame
from yenibot.labeling.triple_barrier import add_long_only_labels
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT, CostScenario
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.adapter import phase2_inputs_from_predictions
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.prediction_ledger import (
    append_forward_predictions,
    read_forward_ledger,
)
from yenibot.phase2.costs import net_long_return
from yenibot.experiment.future_oos import _recorded_evaluation_matches
from yenibot.experiment.configuration import validate_training_research_contract
from yenibot.training.dataset import SequenceDataset
from yenibot.automation.refresh_data import closed_boundary


def inputs():
    times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {
            "bar_open_time": times,
            "bar_close_time": times + pd.Timedelta(hours=1),
            "open": [100.0, 100.0, 80.0],
            "high": [100.0, 101.0, 101.0],
            "low": [100.0, 80.0, 80.0],
            "close": [100.0, 80.0, 101.0],
            "atr_14": 1.0,
        }
    )
    signals = pd.DataFrame({"decision_time": [times[1]], "prob_long": [0.8]})
    return bars, signals


def run(bars, signals, **kwargs):
    return run_long_only_backtest(
        bars,
        signals,
        gate=Phase2Gate(Path("."), False, True, False, False, False, ("audit",)),
        contract=kwargs.pop(
            "contract",
            replace(
                DEFAULT_PHASE2_CONTRACT,
                take_profit_atr=50,
                stop_loss_atr=30,
                max_holding_bars=2,
            ),
        ),
        cost_scenario=CostScenario("zero", 0, 0, 0, 0, 0),
        **kwargs,
    )


def test_mtm_sees_recovered_intratrade_loss_and_full_holding_duration():
    bars, signals = inputs()
    result = run(bars, signals)
    assert result.summary["max_drawdown"] == pytest.approx(-0.2)
    assert result.summary["compounded_return"] == pytest.approx(0.01)
    assert result.trades.holding_hours.iloc[0] == 2
    assert len(result.equity) == len(bars)
    assert result.trades.entry_time.iloc[0] == signals.decision_time.iloc[0]


def test_gap_through_stop_fills_at_open_not_unreachable_stop():
    bars, signals = inputs()
    bars.loc[1, ["low", "close"]] = [99.0, 100.0]
    bars.loc[2, ["high", "low", "close"]] = [85.0, 75.0, 82.0]
    r = run(
        bars,
        signals,
        contract=replace(
            DEFAULT_PHASE2_CONTRACT,
            take_profit_atr=20,
            stop_loss_atr=5,
            max_holding_bars=2,
        ),
    )
    assert r.trades.exit_price.iloc[0] == 80
    assert r.trades.exit_reason.iloc[0] == "stop_loss_gap_open"
    assert r.trades.holding_hours.iloc[0] == 1


@pytest.mark.parametrize("field", ["atr_14", "close", "prob_long"])
def test_nonfinite_inputs_fail_closed(field):
    bars, signals = inputs()
    target = signals if field == "prob_long" else bars
    target.loc[0, field] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run(bars, signals)


def test_right_edge_is_marked_open_not_a_completed_timeout():
    bars, signals = inputs()
    r = run(bars.iloc[:2], signals)
    assert r.summary["trade_count"] == 0
    assert r.summary["censored_position_count"] == 1
    assert r.trades.exit_reason.iloc[0] == "end_of_data_censored"
    assert r.trades.exit_fee_return.iloc[0] == 0
    assert r.summary["compounded_return"] == pytest.approx(-0.2)


def test_exact_fees_and_slippage_reconcile_to_reference_notional_cashflows():
    scenario = CostScenario("test", 4, 5, 2, 3, 0)
    r = net_long_return(scenario, entry_price=100, exit_price=110, holding_hours=2)
    cash_pnl = (
        r["exit_fill_price"]
        - r["entry_fill_price"]
        - 0.0004 * r["entry_fill_price"]
        - 0.0005 * r["exit_fill_price"]
    ) / 100
    assert r["net_return"] == pytest.approx(cash_pnl)
    assert r["gross_return"] - r["net_return"] == pytest.approx(
        sum(
            r[k]
            for k in [
                "entry_fee_return",
                "exit_fee_return",
                "entry_slippage_return",
                "exit_slippage_return",
                "funding_return",
            ]
        )
    )


def test_funding_is_paid_at_events_not_prorated_hours():
    bars, signals = inputs()
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T02:00Z", "2026-01-01T03:00Z"]),
            "funding_rate": [0.001, 0.05],
            "mark_price": [120.0, 100.0],
        }
    )
    r = run(bars, signals, funding_events=events)
    assert r.trades.funding_return.iloc[0] == pytest.approx(0.0012)
    assert r.summary["funding_basis"] == "historical_events"


def test_adapter_exposes_true_open_close_and_decision_times():
    bars, _ = inputs()
    source = bars.rename(columns={"bar_open_time": "timestamp"}).assign(prob_long=0.8)
    b, s, _ = phase2_inputs_from_predictions(
        source, candidate_id="x", threshold=0.5, split="all"
    )
    assert (b.bar_close_time - b.bar_open_time).eq(pd.Timedelta(hours=1)).all()
    assert s.decision_time.equals(b.bar_close_time)


def test_raw_validation_rejects_impossible_ohlc(synthetic_klines):
    frame = synthetic_klines(20, "1h")
    frame.loc[3, "low"] = frame.loc[3, "high"] + 1
    with pytest.raises(ValueError, match="OHLC"):
        validate_full_kline_frame(frame, "1h")


def test_labels_and_sequences_never_bridge_missing_hours(synthetic_klines):
    frame = synthetic_klines(24, "1h").drop(index=5).reset_index(drop=True)
    frame["atr_14"] = 1000.0
    labels = add_long_only_labels(frame)
    assert ((labels.exit_timestamp - labels.timestamp) == pd.Timedelta(hours=10)).all()
    dataset = SequenceDataset(
        np.ones((23, 2)),
        np.zeros(23),
        np.zeros(23),
        seq_len=4,
        timestamps=frame.timestamp,
    )
    for end in dataset.end_positions:
        assert frame.timestamp.iloc[end] - frame.timestamp.iloc[
            end - 3
        ] == pd.Timedelta(hours=3)


def test_recorded_window_cannot_expand_with_same_model_hash(tmp_path):
    path = tmp_path / "evaluation.json"
    original = {
        "rows": 1524,
        "data_start": "2026-06-13T02:00Z",
        "data_end": "2026-08-15T13:00Z",
    }
    row = {
        **original,
        "candidate_id": "x",
        "evidence_passed": False,
        "manifest_hash": "hash",
    }
    path.write_text(json.dumps({"rows": [row]}))
    kwargs = dict(
        primary_id="x",
        recorded_status="failed",
        expected_manifest_hash="hash",
        recorded_outcome=original,
    )
    assert _recorded_evaluation_matches(path, **kwargs)
    row["rows"] = 2524
    path.write_text(json.dumps({"rows": [row]}))
    assert not _recorded_evaluation_matches(path, **kwargs)


def test_forward_ledger_is_idempotent_and_rejects_history_mutation(tmp_path):
    bars, _ = inputs()
    frame = bars.rename(columns={"bar_open_time": "timestamp"}).assign(
        candidate_id="x", manifest_hash="hash", prob_long=0.7
    )
    path = tmp_path / "forward_predictions.jsonl"
    kwargs = dict(candidate_id="x", manifest_hash="hash")
    assert (
        append_forward_predictions(path, frame.iloc[:2], **kwargs)["appended_rows"] == 2
    )
    assert append_forward_predictions(path, frame, **kwargs)["appended_rows"] == 1
    assert append_forward_predictions(path, frame, **kwargs)["appended_rows"] == 0
    frame.loc[0, "prob_long"] = 0.9
    with pytest.raises(ValueError, match="changed"):
        append_forward_predictions(path, frame, **kwargs)
    assert len(read_forward_ledger(path)) == 3
    with pytest.raises(ValueError, match="different model"):
        append_forward_predictions(
            path,
            frame.assign(manifest_hash="new"),
            candidate_id="x",
            manifest_hash="new",
        )


def test_funding_missing_history_cannot_masquerade_as_zero_cost():
    bars, signals = inputs()
    old = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01T00:00Z"]),
                        "funding_rate": [.001], "mark_price": [100.]})
    with pytest.raises(ValueError, match="coverage is incomplete"):
        run(bars, signals, funding_events=old)


def test_no_training_after_closed_swa_and_snapshot_end_is_closed():
    with pytest.raises(ValueError, match="Training is paused"):
        validate_training_research_contract(load_config("config.yaml"))
    assert closed_boundary("2026-08-30T09:42Z", "4h") == pd.Timestamp(
        "2026-08-30T08:00Z"
    )
