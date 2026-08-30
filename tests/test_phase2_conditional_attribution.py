from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from yenibot.phase2.conditional_attribution import (
    FrozenPayoffReplay,
    conditional_groups,
    conditional_tail_summary,
    draw_conditional_mapping,
    lag_one_within_fold,
    mapping_diagnostics,
)
from yenibot.phase2.net_utility import predict_ridge_payoff
from yenibot.automation.phase2_conditional_attribution import (
    load_trial_checkpoint,
    output_lease,
    save_trial_checkpoint,
)


def test_resume_keeps_exact_trial_identity_and_rejects_partial_pairs(tmp_path):
    path = tmp_path / "checkpoint.json"
    rows = [
        {
            "ratio": 1.1,
            "method": "ordinary",
            "seed": 7,
            "trial": trial,
            "cost": cost,
            "compounded_return": 0.01,
            "mean_net_return": 0.001,
        }
        for trial in range(2)
        for cost in ("base", "adverse")
    ]
    kwargs = dict(ratio=1.1, method="ordinary", seed=7)
    save_trial_checkpoint(path, rows, **kwargs)
    assert load_trial_checkpoint(path, **kwargs) == rows
    with pytest.raises(ValueError, match="another conditional variant"):
        load_trial_checkpoint(path, ratio=1.05, method="ordinary", seed=7)
    save_trial_checkpoint(path, rows[:-1], **kwargs)
    with pytest.raises(ValueError, match="Incomplete/duplicate"):
        load_trial_checkpoint(path, **kwargs)
    s, _ = fixture()
    groups, _, _ = conditional_groups(s, ratio=1.1)
    first = np.random.default_rng(7)
    full = [
        draw_conditional_mapping(len(s), groups, method="ordinary", rng=first)
        for _ in range(4)
    ]
    resumed = np.random.default_rng(7)
    for i in range(4):
        draw = draw_conditional_mapping(len(s), groups, method="ordinary", rng=resumed)
        if i >= 2:
            np.testing.assert_array_equal(draw, full[i])


def test_output_lease_blocks_concurrent_runs_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        with output_lease(tmp_path):
            with pytest.raises(OSError):
                with output_lease(tmp_path):
                    pass
            raise RuntimeError("interrupted")
    with output_lease(tmp_path):
        pass


def fixture():
    parts = []
    fits = []
    for fold, month in [(2, 1), (3, 2)]:
        n = 80
        t = np.arange(n)
        a = 0.01 * (1 + 0.01 * np.sin(t / 4))
        a[40:] *= 1.5
        s = pd.DataFrame(
            {
                "decision_time": pd.date_range(
                    f"2023-{month:02}-01", periods=n, freq="h", tz="UTC"
                ),
                "fold": fold,
                "frozen_score_percentile": t / n,
                "decision_atr_close_fraction": a,
                "utility_action": True,
                "label": 0,
                "forward_return": 0.0,
            }
        )
        fit = {
            "center": [0.5, 0.01, 0.005],
            "scale": [0.2, 0.004, 0.003],
            "coefficients": [0.01, 0.02, 0.03],
            "intercept": -0.001,
        }
        fits.append(
            {
                "fold": fold,
                "training_max_fold": fold - 1,
                "training_outcome_max": (
                    s.decision_time.min() - pd.Timedelta(hours=48)
                ).isoformat(),
                "test_start": s.decision_time.min().isoformat(),
                "candidate_fit": fit,
            }
        )
        parts.append(s)
    return pd.concat(parts, ignore_index=True), fits


@pytest.mark.parametrize("ratio", [1.10, 1.05])
@pytest.mark.parametrize("method", ["ordinary", "circular"])
def test_mapping_preserves_context_and_conditional_score_marginals(ratio, method):
    s, fits = fixture()
    groups, audit, _ = conditional_groups(s, ratio=ratio)
    original = s.copy(deep=True)
    mapping = draw_conditional_mapping(
        len(s), groups, method=method, rng=np.random.default_rng(17)
    )
    assert sorted(mapping) == list(range(len(s)))
    for group in groups:
        assert set(mapping[group]) == set(group)
        np.testing.assert_array_equal(
            np.sort(s.frozen_score_percentile.iloc[mapping[group]]),
            np.sort(s.frozen_score_percentile.iloc[group]),
        )
        assert s.fold.iloc[group].nunique() == 1
        if method == "circular":
            assert any(
                np.array_equal(mapping[group], np.roll(group, k))
                for k in range(len(group))
            )
    replay = FrozenPayoffReplay(s, fits)
    predicted, scores = replay.predict(s.frozen_score_percentile.to_numpy()[mapping])
    diag = mapping_diagnostics(s, mapping, scores)
    assert diag["maximum_donor_recipient_atr_ratio"] <= ratio + 1e-12
    assert audit["coverage_sufficient"]
    pd.testing.assert_frame_equal(s, original)
    # Interaction uses the recipient's ATR, never the donor's future context.
    for record in fits:
        idx = s.index[s.fold.eq(record["fold"])]
        p = s.frozen_score_percentile.to_numpy()[mapping[idx]]
        a = s.decision_atr_close_fraction.to_numpy()[idx]
        expected = predict_ridge_payoff(
            record["candidate_fit"], np.column_stack([p, a, p * a])
        )
        np.testing.assert_array_equal(predicted[idx], expected)


def test_atr_only_payoff_is_unchanged_by_score_perturbation():
    s, fits = fixture()
    for r in fits:
        r["candidate_fit"]["coefficients"] = [0.0, 0.02, 0.0]
    replay = FrozenPayoffReplay(s, fits)
    original, _ = replay.predict(s.frozen_score_percentile)
    for ratio in (1.1, 1.05):
        groups, _, _ = conditional_groups(s, ratio=ratio)
        mapping = draw_conditional_mapping(
            len(s), groups, method="ordinary", rng=np.random.default_rng(1)
        )
        after, _ = replay.predict(s.frozen_score_percentile.to_numpy()[mapping])
        np.testing.assert_array_equal(after, original)
    summary = conditional_tail_summary(
        pd.DataFrame({"compounded_return": [0.2] * 500}),
        {"compounded_return": 0.2},
        statistic="compounded_return",
    )
    assert summary["upper_tail_monte_carlo_p"] == 1.0


def test_labels_and_forward_returns_do_not_enter_groups_or_frozen_predictions():
    s, fits = fixture()
    original = FrozenPayoffReplay(s, fits).predict(s.frozen_score_percentile)[0]
    s["label"] = 1
    s["forward_return"] = 999.0
    changed = FrozenPayoffReplay(s, fits).predict(s.frozen_score_percentile)[0]
    np.testing.assert_array_equal(changed, original)


def test_constant_and_singleton_groups_are_retained_and_flagged():
    s, _ = fixture()
    s["frozen_score_percentile"] = 0.5
    s.loc[0, "decision_atr_close_fraction"] = 0.1
    groups, audit, table = conditional_groups(s, ratio=1.05)
    assert sum(len(g) for g in groups) == len(s)
    assert audit["singleton_rows"] == 1
    assert audit["uninformative_rows"] == len(s)
    assert audit["selected_row_exchangeable_fraction"] == 0
    assert not audit["coverage_sufficient"]
    assert table.rows.sum() == len(s)


@pytest.mark.parametrize(
    "failure", ["duplicate_fit", "future_fit", "bad_scale", "bad_atr"]
)
def test_invalid_fit_sources_cannot_silently_produce_abstention(failure):
    s, fits = fixture()
    fits = deepcopy(fits)
    if failure == "duplicate_fit":
        fits.append(fits[0])
    elif failure == "future_fit":
        fits[0]["training_max_fold"] = 2
    elif failure == "bad_scale":
        fits[0]["candidate_fit"]["scale"][1] = 0.0
    else:
        s.loc[0, "decision_atr_close_fraction"] = np.nan
    with pytest.raises(ValueError):
        FrozenPayoffReplay(s, fits)


def test_tail_probability_includes_numerical_ties_and_requires_complete_draws():
    frame = pd.DataFrame({"mean_net_return": np.zeros(500)})
    observed = {"mean_net_return": 0.01}
    assert conditional_tail_summary(frame, observed, statistic="mean_net_return")[
        "upper_tail_monte_carlo_p"
    ] == pytest.approx(1 / 501)
    frame["mean_net_return"] = 0.01 - 1e-15
    assert (
        conditional_tail_summary(frame, observed, statistic="mean_net_return")[
            "upper_tail_monte_carlo_p"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="exactly 500"):
        conditional_tail_summary(
            frame.iloc[:499], observed, statistic="mean_net_return"
        )


def test_lag_statistics_never_bridge_fold_gaps_or_invent_constant_correlation():
    assert lag_one_within_fold([0, 1, 0, 1], [2, 2, 3, 3]) is None
    assert lag_one_within_fold([1, 1, 1], [2, 2, 2]) is None
    assert lag_one_within_fold(
        [0, 1, 2, 9, 10, 11], [2, 2, 2, 3, 3, 3]
    ) == pytest.approx(1.0)
