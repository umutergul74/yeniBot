from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from yenibot.automation.phase2_forward import main as phase2_forward_main
from yenibot.phase2.confirmation import canonical_forward_lock_hash
from yenibot.phase2.confirmation import filter_clean_confirmation_inputs
from yenibot.phase2.confirmation import load_phase2_forward_lock
from yenibot.phase2.confirmation import locked_strategy_contracts
from yenibot.phase2.adapter import attach_phase2_market_columns
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.risk import Phase2RiskPolicy


def _pending_gate(path: Path) -> Phase2Gate:
    return Phase2Gate(
        report_dir=path,
        ready_for_phase2=False,
        report_consistency_passed=True,
        future_oos_evaluation_completed=False,
        future_oos_candidate_passed=False,
        promotion_allowed=False,
        blockers=("future_unseen_oos_not_ready",),
    )


def _bars(start: str, *, periods: int = 5, atr: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_close_time": pd.date_range(start, periods=periods, freq="1h", tz="UTC"),
            "open": [100.0] * periods,
            "high": [100.2] + [102.5] * (periods - 1),
            "low": [99.8] + [99.5] * (periods - 1),
            "close": [100.0] + [102.0] * (periods - 1),
            "atr_14": [atr] * periods,
        }
    )


def _input_manifest(lock: dict) -> dict:
    frozen = lock["frozen_model"]
    return {
        "candidate_manifest": {
            "candidate_id": frozen["candidate_id"],
            "manifest_hash": frozen["candidate_manifest_hash"],
            "source_run_id": frozen["source_run_id"],
            "threshold": {"value": frozen["threshold"]},
        },
        "result": {
            "candidate_id": frozen["candidate_id"],
            "threshold": frozen["threshold"],
        },
    }


def _write_bundle(
    root: Path,
    *,
    decision_time: str,
) -> Path:
    lock = load_phase2_forward_lock()
    source = root / "source" / "phase2_sandbox"
    source.mkdir(parents=True)
    bars = _bars(decision_time, atr=0.8)
    signals = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp(decision_time)],
            "prob_long": [0.60],
            "candidate_id": [lock["frozen_model"]["candidate_id"]],
            "threshold": [lock["frozen_model"]["threshold"]],
        }
    )
    bars.to_csv(source / "phase2_bars.csv", index=False)
    signals.to_csv(source / "phase2_signals.csv", index=False)
    (source / "phase2_input_manifest.json").write_text(
        json.dumps(_input_manifest(lock)),
        encoding="utf-8",
    )
    bundle = root / "phase2_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root / "source"))
    return bundle


def _write_phase1_future_bundle(root: Path) -> Path:
    lock = load_phase2_forward_lock()
    source = root / "phase1_source"
    source.mkdir(parents=True)
    frozen = lock["frozen_model"]
    manifest = {
        "candidate_id": frozen["candidate_id"],
        "manifest_hash": frozen["candidate_manifest_hash"],
        "source_run_id": frozen["source_run_id"],
        "threshold": {"value": frozen["threshold"]},
    }
    (source / "frozen_candidate_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    timestamps = pd.date_range("2026-07-01", periods=5, freq="1h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "candidate_id": [frozen["candidate_id"]] * len(timestamps),
            "prob_long": [0.60] * len(timestamps),
            "threshold": [frozen["threshold"]] * len(timestamps),
            "open": [100.0] * len(timestamps),
            "high": [100.2, 102.0, 100.2, 102.0, 100.2],
            "low": [99.8] * len(timestamps),
            "close": [100.0, 101.5, 100.0, 101.5, 100.0],
            "atr_14": [0.8] * len(timestamps),
        }
    ).to_csv(source / "future_oos_predictions.csv", index=False)
    bundle = root / "phase1_future_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source))
    return bundle


def test_committed_forward_lock_is_self_verified_and_bounded() -> None:
    lock = load_phase2_forward_lock()
    assert lock["lock_hash"] == canonical_forward_lock_hash(lock)
    assert lock["success_gates"]["automatic_promotion_allowed"] is False
    assert lock["confirmation_window"]["minimum_trade_count"] == 75
    assert set(locked_strategy_contracts(lock)) == {
        "primary_balanced",
        "challenger_return",
    }


def test_clean_boundary_excludes_seen_window_rows() -> None:
    lock = load_phase2_forward_lock()
    bars = pd.concat(
        [
            _bars("2026-05-01", periods=2),
            _bars("2026-07-01", periods=2),
        ],
        ignore_index=True,
    )
    signals = pd.DataFrame(
        {
            "decision_time": ["2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "prob_long": [0.60, 0.60],
            "candidate_id": ["control_recent3_equal_v2"] * 2,
            "threshold": [lock["frozen_model"]["threshold"]] * 2,
        }
    )
    result = filter_clean_confirmation_inputs(bars, signals, lock=lock)
    assert len(result.signals) == 1
    assert result.boundary_report["excluded_pre_anchor_signal_count"] == 1
    assert result.boundary_report["fit_operations_performed"] == 0
    assert result.signals["decision_time"].min() > pd.Timestamp(
        lock["confirmation_window"]["decision_time_start_exclusive"]
    )


def test_prediction_artifact_can_receive_causal_market_columns() -> None:
    timestamps = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    predictions = pd.DataFrame(
        {"timestamp": timestamps, "prob_long": [0.4, 0.6, 0.7]}
    )
    context = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "atr_14": [1.0, 1.1, 1.2],
        }
    )
    result = attach_phase2_market_columns(predictions, context)
    assert {"open", "high", "low", "close", "atr_14"}.issubset(result.columns)
    assert result["atr_14"].tolist() == [1.0, 1.1, 1.2]


def test_fixed_fractional_risk_sizes_from_initial_stop(tmp_path: Path) -> None:
    bars = _bars("2026-07-01", atr=1.0)
    signals = pd.DataFrame(
        {"decision_time": [bars.loc[0, "bar_close_time"]], "prob_long": [0.70]}
    )
    contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        threshold=0.50,
        take_profit_atr=2.0,
        stop_loss_atr=4.0,
        max_holding_bars=2,
    )
    policy = Phase2RiskPolicy()
    result = run_long_only_backtest(
        bars,
        signals,
        gate=_pending_gate(tmp_path),
        contract=contract,
        risk_policy=policy,
    )
    trade = result.trades.iloc[0]
    assert trade["stop_distance_fraction"] == pytest.approx(0.04)
    assert trade["risk_sizing_loss_fraction"] == pytest.approx(0.0412)
    assert trade["position_notional_fraction"] == pytest.approx(0.0025 / 0.0412)
    assert trade["portfolio_return"] == pytest.approx(
        trade["net_return"] * trade["position_notional_fraction"]
    )
    assert result.summary["return_basis"] == "risk_sized_portfolio_return"


@pytest.mark.parametrize(
    ("daily_limit", "drawdown_limit", "expected_daily", "expected_drawdown"),
    [
        (0.002, 0.50, 1, 0),
        (0.50, 0.002, 0, 1),
    ],
)
def test_risk_guardrails_block_new_entries_after_realized_loss(
    tmp_path: Path,
    daily_limit: float,
    drawdown_limit: float,
    expected_daily: int,
    expected_drawdown: int,
) -> None:
    bars = _bars("2026-07-01", periods=6, atr=1.0)
    bars.loc[1:, "high"] = 100.2
    bars.loc[1:, "low"] = 98.0
    bars.loc[1:, "close"] = 99.0
    signals = pd.DataFrame(
        {
            "decision_time": [
                bars.loc[0, "bar_close_time"],
                bars.loc[2, "bar_close_time"],
            ],
            "prob_long": [0.70, 0.70],
        }
    )
    contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        threshold=0.50,
        take_profit_atr=5.0,
        stop_loss_atr=1.0,
        max_holding_bars=1,
    )
    policy = Phase2RiskPolicy(
        risk_fraction_per_trade=0.01,
        max_notional_fraction=0.25,
        daily_realized_loss_limit_fraction=daily_limit,
        max_realized_drawdown_fraction=drawdown_limit,
    )
    result = run_long_only_backtest(
        bars,
        signals,
        gate=_pending_gate(tmp_path),
        contract=contract,
        risk_policy=policy,
    )
    assert len(result.trades) == 1
    assert result.summary["risk_daily_loss_skip_count"] == expected_daily
    assert result.summary["risk_drawdown_halt_skip_count"] == expected_drawdown


def test_forward_runner_waits_when_bundle_contains_only_seen_rows(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path, decision_time="2026-05-01T00:00:00Z")
    output = tmp_path / "output"
    assert phase2_forward_main(
        ["--bundle", str(bundle), "--output-dir", str(output)]
    ) == 0
    report = json.loads(
        (output / "phase2_forward_decision.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "waiting_for_post_anchor_data"
    assert report["boundary"]["accepted_signal_count"] == 0
    assert report["fit_operations_performed"] == 0
    assert report["candidate_results"] == []


def test_forward_runner_applies_locked_candidates_and_risk_policy(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path, decision_time="2026-07-01T00:00:00Z")
    output = tmp_path / "output"
    assert phase2_forward_main(
        ["--bundle", str(bundle), "--output-dir", str(output)]
    ) == 0
    report = json.loads(
        (output / "phase2_forward_decision.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "historical_audit_only_lock_requires_new_preregistration"
    assert {item["role"] for item in report["candidate_results"]} == {
        "primary_balanced",
        "challenger_return",
    }
    primary = next(
        item
        for item in report["candidate_results"]
        if item["role"] == "primary_balanced"
    )
    assert primary["base"]["return_basis"] == "risk_sized_portfolio_return"
    assert primary["gate"]["promotion_allowed"] is False
    ledger = pd.read_csv(
        output
        / "candidates"
        / "primary_balanced"
        / "base"
        / "phase2_trade_ledger.csv"
    )
    assert {
        "position_notional_fraction",
        "equity_before",
        "equity_after",
        "portfolio_return",
    }.issubset(ledger.columns)


def test_forward_runner_consumes_phase1_future_oos_predictions(
    tmp_path: Path,
) -> None:
    bundle = _write_phase1_future_bundle(tmp_path)
    output = tmp_path / "output"
    assert phase2_forward_main(
        ["--bundle", str(bundle), "--output-dir", str(output)]
    ) == 0
    report = json.loads(
        (output / "phase2_forward_decision.json").read_text(encoding="utf-8")
    )
    assert report["boundary"]["source_audit"]["source_mode"] == (
        "future_oos_predictions"
    )
    assert report["boundary"]["accepted_signal_count"] == 5
    assert len(report["candidate_results"]) == 2
