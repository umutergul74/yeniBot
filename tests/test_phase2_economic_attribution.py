from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.economic_attribution import EconomicAttributionSpec
from yenibot.phase2.economic_attribution import run_economic_attribution
from yenibot.phase2.economic_attribution import write_economic_attribution
from yenibot.phase2.readiness import Phase2Gate


def _gate(tmp_path: Path) -> Phase2Gate:
    return Phase2Gate(
        report_dir=tmp_path,
        ready_for_phase2=False,
        report_consistency_passed=True,
        future_oos_evaluation_completed=True,
        future_oos_candidate_passed=False,
        promotion_allowed=False,
        blockers=("future_oos_failed",),
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = 130
    close_times = pd.date_range(
        "2026-01-01 01:00", periods=periods, freq="1h", tz="UTC"
    )
    trend = 100.0 + np.arange(periods) * 0.03
    wave = np.sin(np.arange(periods) / 4.0) * 0.35
    close = trend + wave
    bars = pd.DataFrame(
        {
            "bar_close_time": close_times,
            "open": close - 0.05,
            "high": close + 0.55,
            "low": close - 0.55,
            "close": close,
            "atr_14": np.full(periods, 0.50),
        }
    )
    decisions = close_times[:120]
    scores = np.linspace(0.20, 0.80, len(decisions))
    signals = pd.DataFrame(
        {
            "decision_time": decisions,
            "prob_long": scores,
            "candidate_id": DEFAULT_PHASE2_CONTRACT.candidate_id,
            "threshold": DEFAULT_PHASE2_CONTRACT.threshold,
            "fold": np.repeat([1, 2, 3], 40),
            "split": "test",
            "label": (scores > 0.50).astype(int),
            "forward_return": (scores - 0.50) / 100.0,
            "tb_return": (scores - 0.50) / 100.0,
        }
    )
    return bars, signals


def _spec() -> EconomicAttributionSpec:
    return EconomicAttributionSpec(
        permutations=20,
        seed=17,
        minimum_input_rows=20,
        minimum_completed_trades=1,
    )


def test_economic_attribution_is_deterministic_and_non_promotable(
    tmp_path: Path,
) -> None:
    bars, signals = _inputs()

    first = run_economic_attribution(
        bars,
        signals,
        gate=_gate(tmp_path),
        contract=DEFAULT_PHASE2_CONTRACT,
        spec=_spec(),
    )
    second = run_economic_attribution(
        bars,
        signals,
        gate=_gate(tmp_path),
        contract=DEFAULT_PHASE2_CONTRACT,
        spec=_spec(),
    )

    assert len(first.null_trials) == 20
    assert first.report["promotion_allowed"] is False
    assert first.report["live_trading_allowed"] is False
    assert first.report["model_or_strategy_refit_performed"] is False
    assert first.report["contract_hash"] == second.report["contract_hash"]
    pd.testing.assert_frame_equal(first.null_trials, second.null_trials)
    expected_selected = int(
        (signals.prob_long >= DEFAULT_PHASE2_CONTRACT.threshold).sum()
    )
    assert first.null_trials.selected_signal_count.eq(expected_selected).all()
    assert set(first.score_bands.score_decile) == set(range(1, 11))

    output_dir = tmp_path / "attribution"
    write_economic_attribution(output_dir, first)
    assert (output_dir / "phase2_economic_attribution.json").exists()
    assert (output_dir / "phase2_rank_destroyed_null_trials.csv").exists()
    assert (output_dir / "phase2_economic_attribution.md").exists()


def test_economic_attribution_rejects_mixed_or_non_test_splits(tmp_path: Path) -> None:
    bars, signals = _inputs()
    signals.loc[0, "split"] = "val"

    with pytest.raises(ValueError, match="one explicit untouched split"):
        run_economic_attribution(
            bars,
            signals,
            gate=_gate(tmp_path),
            contract=DEFAULT_PHASE2_CONTRACT,
            spec=_spec(),
        )


def test_economic_attribution_rejects_threshold_and_candidate_drift(
    tmp_path: Path,
) -> None:
    bars, signals = _inputs()
    signals["threshold"] = DEFAULT_PHASE2_CONTRACT.threshold + 0.01

    with pytest.raises(ValueError, match="Frozen threshold mismatch"):
        run_economic_attribution(
            bars,
            signals,
            gate=_gate(tmp_path),
            contract=DEFAULT_PHASE2_CONTRACT,
            spec=_spec(),
        )

    _, signals = _inputs()
    drifted_contract = replace(DEFAULT_PHASE2_CONTRACT, candidate_id="other")
    with pytest.raises(ValueError, match="Candidate identity mismatch"):
        run_economic_attribution(
            bars,
            signals,
            gate=_gate(tmp_path),
            contract=drifted_contract,
            spec=_spec(),
        )


def test_economic_attribution_rejects_duplicate_decision_times(tmp_path: Path) -> None:
    bars, signals = _inputs()
    signals.loc[1, "decision_time"] = signals.loc[0, "decision_time"]

    with pytest.raises(ValueError, match="ambiguous timestamps"):
        run_economic_attribution(
            bars,
            signals,
            gate=_gate(tmp_path),
            contract=DEFAULT_PHASE2_CONTRACT,
            spec=_spec(),
        )


def test_walk_forward_embargo_gaps_are_segmented_not_traded(tmp_path: Path) -> None:
    bars, signals = _inputs()
    segments = []
    for fold, start in ((1, 0), (2, 40), (3, 80)):
        segment = bars.iloc[start : start + 40].copy()
        shift = pd.Timedelta(hours=(fold - 1) * 64)
        segment["bar_close_time"] = segment["bar_close_time"] + shift
        segments.append(segment)
        mask = signals["fold"].eq(fold)
        signals.loc[mask, "decision_time"] = signals.loc[mask, "decision_time"] + shift
    gapped_bars = pd.concat(segments, ignore_index=True)

    result = run_economic_attribution(
        gapped_bars,
        signals,
        gate=_gate(tmp_path),
        contract=DEFAULT_PHASE2_CONTRACT,
        spec=_spec(),
    )

    diagnostics = result.report["input_diagnostics"]
    assert diagnostics["fold_segmented_evaluation"] is True
    assert diagnostics["inter_fold_gap_count"] == 2
    assert diagnostics["within_fold_bar_gap_count"] == 0
    assert result.report["actual"]["base"]["fold_count"] == 3
    assert result.report["actual"]["base"]["data_contract_complete"] is True
