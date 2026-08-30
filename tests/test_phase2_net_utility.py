from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.net_utility import (
    build_net_utility_signals,
    fit_ridge_payoff,
    paired_fold_block_intervals,
    predict_ridge_payoff,
    utility_to_score,
    validation_opportunity_targets,
)
from yenibot.phase2.readiness import load_phase2_gate


def _frame():
    parts = []
    for fold in range(2):
        for split, day, count in (("val", 1, 240), ("test", 16, 36)):
            t = np.arange(count)
            close = 100 + 0.03 * t + 0.9 * np.sin(t / 11)
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            f"2024-0{fold + 1}-{day:02d}",
                            periods=count,
                            freq="h",
                            tz="UTC",
                        ),
                        "fold": fold,
                        "split": split,
                        "prob_long": 0.5 + 0.4 * np.sin(t / 9),
                        "open": close - 0.04,
                        "high": close + 0.3,
                        "low": close - 0.3,
                        "close": close,
                        "atr_14": 0.22,
                        "label": (t % 3 == 0).astype(int),
                        "forward_return": 0.002,
                        "tb_return": 0.001,
                    }
                )
            )
    return (
        pd.concat(parts, ignore_index=True),
        {
            "fold_count": 2,
            "validation_rows_per_fold": 240,
            "test_rows_per_fold": 36,
            "label_maturity_hours": 10,
            "historical_cutoff": "2024-03-01T00:00:00Z",
            "source_run_id": "fixture",
        },
        {"ridge_alpha": 10.0, "minimum_fit_rows": 200},
    )


def test_ridge_matches_reference_and_handles_constant_features():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(250, 3))
    x[:, 2] = 2.0
    y = 0.004 * x[:, 0] + rng.normal(0, 0.005, len(x))
    fit = fit_ridge_payoff(x, y, alpha=10)
    scaler = StandardScaler().fit(x)
    expected = Ridge(alpha=10).fit(scaler.transform(x), y).predict(scaler.transform(x))
    np.testing.assert_allclose(predict_ridge_payoff(fit, x), expected, atol=1e-12)
    assert fit["scale"][2] == 1.0


def test_zero_negative_and_missing_utility_abstain():
    utility = np.array([-1.0, -0.001, 0.0, -0.0, np.nan, np.inf, 1e-30, 0.001])
    score = utility_to_score(utility)
    assert np.isfinite(score).all()
    np.testing.assert_array_equal(score >= 0.5, [False] * 6 + [True, True])


def test_test_labels_future_scores_and_other_folds_do_not_change_fit_or_earlier_actions():
    frame, source, spec = _frame()
    first, fits, _ = build_net_utility_signals(
        frame, source_spec=source, utility_spec=spec
    )
    changed = frame.copy()
    changed.loc[changed.split.eq("test"), ["label", "forward_return", "tb_return"]] = [
        1,
        -999.0,
        999.0,
    ]
    changed.loc[changed.fold.eq(1), "prob_long"] = 0.75
    last_test = changed.index[changed.fold.eq(0) & changed.split.eq("test")][-1]
    changed.loc[last_test, "prob_long"] = 0.001
    second, second_fits, _ = build_net_utility_signals(
        changed, source_spec=source, utility_spec=spec
    )
    assert fits[0]["fit"] == second_fits[0]["fit"]
    np.testing.assert_array_equal(first.utility_score[:35], second.utility_score[:35])
    assert fits[0]["outcome_max"] < fits[0]["test_start"]


@pytest.mark.parametrize("case", ["ordinary", "same_bar", "gap"])
def test_opportunity_targets_match_reference_single_trades(tmp_path: Path, case):
    frame, _, _ = _frame()
    val = frame.loc[frame.fold.eq(0) & frame.split.eq("val")].copy()
    contract = DEFAULT_PHASE2_CONTRACT
    if case == "same_bar":
        contract = replace(contract, take_profit_atr=0.5, stop_loss_atr=0.5)
    elif case == "gap":
        val = val.drop(index=100)
    val = val.reset_index(drop=True)
    targets = validation_opportunity_targets(
        val, test_start=pd.Timestamp("2024-01-16", tz="UTC"), contract=contract
    )
    bars = val[["timestamp", "open", "high", "low", "close", "atr_14"]].rename(
        columns={"timestamp": "bar_open_time"}
    )
    bars["bar_close_time"] = bars.bar_open_time + pd.Timedelta(hours=1)
    for idx in (0, 50, 96, len(val) - 5, len(val) - 1):
        signals = pd.DataFrame(
            {"decision_time": [bars.bar_close_time.iloc[idx]], "prob_long": [1.0]}
        )
        actual = run_long_only_backtest(
            bars,
            signals,
            gate=load_phase2_gate(tmp_path),
            contract=contract,
            cost_scenario=contract.cost_scenarios[-1],
        )
        row = targets.iloc[idx]
        if actual.trades.empty or actual.trades.trade_status.iloc[0] == "censored":
            assert not row.eligible
        else:
            assert row.eligible
            assert row.adverse_net_target == pytest.approx(
                actual.trades.net_return.iloc[0], abs=1e-12
            )
    assert not targets.eligible.iloc[-1]


def test_insufficient_fit_and_incomplete_outcomes_abstain():
    frame, source, spec = _frame()
    spec["minimum_fit_rows"] = 10000
    signals, fits, targets = build_net_utility_signals(
        frame, source_spec=source, utility_spec=spec
    )
    assert not signals.utility_action.any()
    assert not any(f["fit_performed"] for f in fits)
    assert targets.loc[~targets.eligible, "adverse_net_target"].isna().all()
    val = frame.loc[frame.fold.eq(0) & frame.split.eq("val")]
    boundary = val.timestamp.iloc[100]
    cut = validation_opportunity_targets(val, test_start=boundary)
    assert (cut.loc[cut.eligible, "outcome_time_conservative"] < boundary).all()


def test_paired_fold_uncertainty_uses_aligned_folds_not_hourly_samples():
    reference = pd.DataFrame({"fold": range(18), "compounded_net_return": 0.01})
    candidate = reference.copy()
    candidate["compounded_net_return"] += 0.02
    args = dict(block_lengths=[3, 6], replicates=200, seed=9)
    intervals, pairs = paired_fold_block_intervals(candidate, reference, **args)
    assert len(pairs) == 18
    assert all(i["lower_bound_positive"] for i in intervals)
    identical, _ = paired_fold_block_intervals(reference, reference, **args)
    assert not any(i["lower_bound_positive"] for i in identical)
    with pytest.raises(ValueError, match="complete aligned"):
        paired_fold_block_intervals(candidate.iloc[1:], reference, **args)
