import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yenibot.phase2.prequential_utility as module
from yenibot.automation.phase2_prequential_utility import (
    _load_locked_spec,
    main,
)
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.engine import Phase2BacktestResult, run_long_only_backtest
from yenibot.phase2.economic_attribution import _fold_outcomes
from yenibot.phase2.prequential_utility import (
    build_prequential_signals,
    oof_opportunity_targets,
    select_common_cohort,
)
from yenibot.phase2.readiness import load_phase2_gate


def _inputs():
    bar_parts, signal_parts = [], []
    for fold in range(4):
        n = 120
        t = np.arange(n)
        close = 100 + 0.02 * t + 0.8 * np.sin(t / 8 + fold)
        times = pd.date_range(f"2024-0{fold + 1}-01", periods=n, freq="h", tz="UTC")
        bar_parts.append(
            pd.DataFrame(
                {
                    "bar_open_time": times,
                    "bar_close_time": times + pd.Timedelta(hours=1),
                    "open": close - 0.02,
                    "high": close + 0.18,
                    "low": close - 0.18,
                    "close": close,
                    "atr_14": 0.21 + 0.02 * np.cos(t / 13),
                }
            )
        )
        signal_parts.append(
            pd.DataFrame(
                {
                    "decision_time": times + pd.Timedelta(hours=1),
                    "fold": fold,
                    "split": "test",
                    "prob_long": 0.5 + 0.4 * np.sin(t / 9),
                    "candidate_id": "fixture",
                    "label": t % 2,
                    "forward_return": 0.001,
                    "tb_return": 0.002,
                }
            )
        )
    spec = {
        "warmup_folds": [0, 1],
        "evaluation_fold_first": 2,
        "evaluation_fold_last": 3,
        "minimum_mature_history_rows": 100,
        "ridge_alpha": 10,
        "candidate_id": "fixture_candidate",
        "context_control_id": "fixture_atr",
    }
    return (
        pd.concat(bar_parts, ignore_index=True),
        pd.concat(signal_parts, ignore_index=True),
        spec,
    )


def test_all_fit_members_are_strictly_past_with_original_split_provenance():
    bars, signals, spec = _inputs()
    candidate, control, fits, targets = build_prequential_signals(
        bars, signals, spec=spec
    )
    assert list(candidate.fold.unique()) == [2, 3]
    assert len(candidate) == len(control) == 240
    assert set(targets.source_split) == {"test"}
    for record in fits:
        assert record["training_max_fold"] == record["fold"] - 1
        assert pd.Timestamp(record["training_outcome_max"]) < pd.Timestamp(
            record["test_start"]
        )
        mask = targets.fit_eligible & targets.fold.lt(record["fold"])
        assert record["eligible_history_rows"] == int(mask.sum())
        assert record["fit_operations"] == 2
        assert record["earlier_oof_test_outcomes_used"] is True
        assert record["current_or_future_fold_outcomes_used"] is False
        assert record["atr_only_fit"]["coefficients"][0] == 0
        assert record["atr_only_fit"]["coefficients"][2] == 0
    assert targets.loc[~targets.eligible, "adverse_net_target"].isna().all()


def test_current_and_future_outcomes_cannot_change_current_fit_or_earlier_decisions():
    bars, signals, spec = _inputs()
    first, _, fits, _ = build_prequential_signals(bars, signals, spec=spec)
    changed_bars, changed_signals = bars.copy(), signals.copy()
    changed_signals[["label", "forward_return", "tb_return"]] = [0, 999.0, -999.0]
    # Keep the first 60 decisions in fold 2 unmodified; destroy later price paths.
    late = ((signals.fold == 2) & (signals.index >= 300)) | signals.fold.eq(3)
    changed_bars.loc[late, ["open", "high", "low", "close", "atr_14"]] *= 2.0
    changed_signals.loc[late, "prob_long"] = 0.01
    second, _, second_fits, _ = build_prequential_signals(
        changed_bars, changed_signals, spec=spec
    )
    assert fits[0]["candidate_fit"] == second_fits[0]["candidate_fit"]
    assert fits[0]["atr_only_fit"] == second_fits[0]["atr_only_fit"]
    assert (
        fits[0]["training_membership_sha256"]
        == second_fits[0]["training_membership_sha256"]
    )
    np.testing.assert_array_equal(first.utility_score[:60], second.utility_score[:60])


def test_past_outcomes_can_inform_later_fit_and_equal_boundary_is_excluded(monkeypatch):
    bars, signals, spec = _inputs()
    _, _, fits, targets = build_prequential_signals(bars, signals, spec=spec)
    changed = targets.copy()
    idx = changed.index[changed.eligible & changed.fold.eq(0)][0]
    changed.loc[idx, "outcome_time_conservative"] = signals.loc[
        signals.fold.eq(2), "decision_time"
    ].min()
    changed.loc[changed.eligible & changed.fold.eq(1), "adverse_net_target"] += 0.05
    monkeypatch.setattr(
        module, "oof_opportunity_targets", lambda *args, **kwargs: changed.copy()
    )
    _, _, after, _ = build_prequential_signals(bars, signals, spec=spec)
    assert after[0]["eligible_history_rows"] == fits[0]["eligible_history_rows"] - 1
    assert after[0]["candidate_fit"] != fits[0]["candidate_fit"]
    assert after[1]["eligible_history_rows"] == fits[1]["eligible_history_rows"]


def test_atr_control_does_not_consume_score_values():
    bars, signals, spec = _inputs()
    _, before, fits, _ = build_prequential_signals(bars, signals, spec=spec)
    changed = signals.copy()
    changed["prob_long"] = 1.0 - changed.prob_long
    _, after, new_fits, _ = build_prequential_signals(bars, changed, spec=spec)
    np.testing.assert_array_equal(before.utility_score, after.utility_score)
    assert [r["atr_only_fit"] for r in fits] == [r["atr_only_fit"] for r in new_fits]


@pytest.mark.parametrize("fold", [0, 2])
def test_fold_local_exit_indices_match_single_trade_reference(tmp_path, fold):
    bars, signals, _ = _inputs()
    targets = oof_opportunity_targets(bars, signals)
    fold_bars = bars.loc[signals.fold.eq(fold)].reset_index(drop=True)
    selected = targets.loc[targets.fold.eq(fold)].reset_index(drop=True)
    contract = DEFAULT_PHASE2_CONTRACT
    for i in (0, 51, 116, 119):
        signal = pd.DataFrame(
            {"decision_time": [selected.decision_time.iloc[i]], "prob_long": [1.0]}
        )
        actual = run_long_only_backtest(
            fold_bars,
            signal,
            gate=load_phase2_gate(tmp_path),
            contract=contract,
            cost_scenario=contract.cost_scenarios[-1],
        )
        row = selected.iloc[i]
        if actual.trades.empty or actual.trades.trade_status.iloc[0] == "censored":
            assert not row.eligible
        else:
            assert row.eligible
            assert row.adverse_net_target == pytest.approx(
                actual.trades.net_return.iloc[0], abs=1e-12
            )
            assert (
                row.outcome_time_conservative
                == fold_bars.bar_close_time.iloc[row.exit_index_within_fold]
            )


@pytest.mark.parametrize(
    "case",
    ["short_history", "missing_fold", "duplicate_time", "wrong_split", "wrong_clock"],
)
def test_contract_violations_fail_closed_instead_of_dropping_folds(case):
    bars, signals, spec = _inputs()
    if case == "short_history":
        spec["minimum_mature_history_rows"] = 10000
    elif case == "missing_fold":
        keep = signals.fold.ne(1)
        bars, signals = bars.loc[keep], signals.loc[keep]
    elif case == "duplicate_time":
        signals.loc[1, "decision_time"] = signals.decision_time.iloc[0]
    elif case == "wrong_split":
        signals.loc[0, "split"] = "val"
    else:
        bars.loc[0, "bar_open_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        build_prequential_signals(bars, signals, spec=spec)


def test_common_cohort_keeps_zero_trade_folds_and_rejects_missing_rows():
    _, signals, _ = _inputs()
    keys = signals.loc[signals.fold.ge(2), ["fold", "decision_time"]]
    selected = select_common_cohort(signals, keys, folds=[2, 3])
    assert len(selected) == 240
    outcomes = _fold_outcomes(
        Phase2BacktestResult(pd.DataFrame(), pd.DataFrame(), {}, {}),
        selected,
        contract=DEFAULT_PHASE2_CONTRACT,
    )
    assert list(outcomes.fold) == [2, 3]
    assert outcomes.trade_count.sum() == 0
    with pytest.raises(ValueError, match="same full evaluation cohort"):
        select_common_cohort(signals.drop(index=300), keys, folds=[2, 3])


def test_locked_spec_and_existing_output_cannot_be_silently_changed(tmp_path):
    spec = Path("configs/prequential_oof_utility_v1.json")
    _load_locked_spec(spec)
    modified = json.loads(spec.read_text(encoding="utf-8"))
    modified["entry_utility_cutoff"] = -0.001
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="Pinned research protocol"):
        _load_locked_spec(changed)
    changed.write_text(
        spec.read_text(encoding="utf-8"), encoding="utf-8", newline="\r\n"
    )
    _load_locked_spec(changed)
    output = tmp_path / "completed"
    output.mkdir()
    (output / "checkpoint.json").write_text("{}")
    args = ["--output-dir", str(output)]
    for arg in ("scope-dir", "report-dir", "q80-dir", "validation-probe-dir"):
        args.extend([f"--{arg}", str(tmp_path)])
    with pytest.raises(FileExistsError, match="overwrite"):
        main(args)
