"""Past-OOF payoff fitting; strictly expanding, with no current-fold outcome use."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.execution_cache import build_fold_execution_cache
from yenibot.phase2.net_utility import (
    _features,
    fit_ridge_payoff,
    predict_ridge_payoff,
    utility_to_score,
)


def _aligned_inputs(bars, signals):
    bars, signals = bars.copy(), signals.copy()
    if bars.empty or len(bars) != len(signals):
        raise ValueError("OOF requires one decision per observed bar")
    for frame, columns in (
        (bars, ["bar_open_time", "bar_close_time"]),
        (signals, ["decision_time"]),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    bars = bars.sort_values("bar_close_time").reset_index(drop=True)
    signals = signals.sort_values("decision_time").reset_index(drop=True)
    if (
        signals.decision_time.isna().any()
        or signals.decision_time.duplicated().any()
        or not np.array_equal(bars.bar_close_time, signals.decision_time)
        or not (bars.bar_close_time - bars.bar_open_time)
        .eq(pd.Timedelta(hours=1))
        .all()
    ):
        raise ValueError("OOF bar/decision identity or hourly clock mismatch")
    if signals.split.isna().any() or set(signals.split.unique()) != {"test"}:
        raise ValueError("Keep original OOF test split provenance")
    fold = pd.to_numeric(signals.fold, errors="raise")
    if not np.isfinite(fold).all() or not fold.eq(np.floor(fold)).all():
        raise ValueError("OOF fold IDs must be finite integers")
    signals["fold"] = fold.astype(int)
    if not signals.fold.is_monotonic_increasing:
        raise ValueError("OOF folds must be chronologically ordered")
    scores = pd.to_numeric(signals.prob_long, errors="raise")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("OOF percentile must be finite and in [0, 1]")
    signals["prob_long"] = scores
    return bars, signals


def oof_opportunity_targets(bars, signals, *, contract=DEFAULT_PHASE2_CONTRACT):
    """All potential outcomes, not a portfolio; fit-time masking is mandatory.

    Caching future targets is a computational convenience, never permission to
    consume them before their outcome time. Exit indices are LOCAL to each fold.
    """
    bars, signals = _aligned_inputs(bars, signals)
    contract = replace(contract, score_column="prob_long")
    adverse = next(c for c in contract.cost_scenarios if c.name == "adverse")
    cache = build_fold_execution_cache(
        bars, signals, contract=contract, scenario=adverse
    )
    outcome = pd.Series(pd.NaT, index=signals.index, dtype="datetime64[ns, UTC]")
    for indices in cache.folds:
        group = signals.iloc[indices]
        fold_bars = bars.loc[
            bars.bar_close_time.between(
                group.decision_time.min(), group.decision_time.max()
            )
        ].reset_index(drop=True)
        valid = indices[cache.entry_indices[indices] >= 0]
        outcome.loc[valid] = fold_bars.bar_close_time.iloc[
            cache.exit_indices[valid]
        ].to_numpy()
    eligible = (cache.entry_indices >= 0) & ~cache.censored & ~cache.data_gaps
    eligible &= outcome.notna().to_numpy() & np.isfinite(cache.net_returns)
    reasons = np.full(len(signals), "eligible", dtype=object)
    reasons[cache.entry_indices < 0] = "no_eligible_next_entry"
    reasons[cache.censored] = "censored_outcome"
    reasons[cache.data_gaps] = "data_gap_outcome"
    return pd.DataFrame(
        {
            "decision_time": signals.decision_time,
            "fold": signals.fold,
            "source_split": signals.split,
            "entry_index_within_fold": cache.entry_indices,
            "exit_index_within_fold": cache.exit_indices,
            "outcome_time_conservative": outcome,
            "eligible": eligible,
            "exclusion_reason": reasons,
            "adverse_net_target": np.where(eligible, cache.net_returns, np.nan),
        }
    )


def select_common_cohort(signals, reference_keys, *, folds):
    """Never select a policy's active rows or discard its zero-trade folds."""
    selected = signals.loc[signals.fold.isin(folds)].copy()
    keys = ["fold", "decision_time"]
    for frame in (selected, reference_keys):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Common cohort has missing/duplicate identity")
    selected["decision_time"] = pd.to_datetime(selected.decision_time, utc=True)
    reference = reference_keys[keys].copy()
    reference["decision_time"] = pd.to_datetime(reference.decision_time, utc=True)
    selected = selected.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    if sorted(selected.fold.unique()) != list(folds) or not selected[keys].equals(
        reference
    ):
        raise ValueError("Policies must cover exactly the same full evaluation cohort")
    return selected


def build_prequential_signals(
    bars,
    signals,
    *,
    spec: dict[str, Any],
    on_fold: Callable[[dict[str, Any]], None] | None = None,
):
    bars, signals = _aligned_inputs(bars, signals)
    first, last = spec["evaluation_fold_first"], spec["evaluation_fold_last"]
    if spec["warmup_folds"] != list(range(first)) or sorted(
        signals.fold.unique()
    ) != list(range(last + 1)):
        raise ValueError("Fixed warmup/evaluation fold coverage differs from contract")
    targets = oof_opportunity_targets(bars, signals)
    x = _features(bars, signals.prob_long.to_numpy(dtype=float))
    context_x = np.column_stack([np.zeros(len(x)), x[:, 1], np.zeros(len(x))])
    targets["fit_eligible"] = targets.eligible & np.isfinite(x).all(axis=1)
    targets["frozen_score_percentile"] = x[:, 0]
    targets["decision_atr_close_fraction"] = x[:, 1]
    targets["score_atr_product"] = x[:, 2]
    fits, parts, controls = [], [], []
    for fold in range(first, last + 1):
        test_mask = signals.fold.eq(fold).to_numpy()
        start = signals.loc[test_mask, "decision_time"].min()
        # Both constraints matter: outcomes must be past AND from an earlier fold.
        train = (
            targets.fit_eligible
            & targets.fold.lt(fold)
            & targets.outcome_time_conservative.lt(start)
        ).to_numpy()
        if int(train.sum()) < spec["minimum_mature_history_rows"]:
            raise ValueError(
                f"Insufficient mature history at fixed evaluation fold {fold}"
            )
        y = targets.loc[train, "adverse_net_target"].to_numpy(dtype=float)
        fit = fit_ridge_payoff(x[train], y, alpha=spec["ridge_alpha"])
        context_fit = fit_ridge_payoff(context_x[train], y, alpha=spec["ridge_alpha"])
        membership = targets.loc[
            train, ["fold", "decision_time", "outcome_time_conservative"]
        ]
        record = {
            "fold": fold,
            "test_start": start.isoformat(),
            "eligible_history_rows": int(train.sum()),
            "training_max_fold": int(targets.loc[train, "fold"].max()),
            "training_outcome_max": targets.loc[train, "outcome_time_conservative"]
            .max()
            .isoformat(),
            "training_membership_sha256": hashlib.sha256(
                membership.to_csv(index=False, lineterminator="\n").encode("utf-8")
            ).hexdigest(),
            "training_role": "past_oof_calibration_train",
            "earlier_oof_test_outcomes_used": True,
            "current_or_future_fold_outcomes_used": False,
            "candidate_fit": fit,
            "atr_only_fit": context_fit,
            "fit_operations": 2,
        }
        for destination, selected_fit, features, candidate in (
            (parts, fit, x, spec["candidate_id"]),
            (controls, context_fit, context_x, spec["context_control_id"]),
        ):
            prediction = predict_ridge_payoff(selected_fit, features[test_mask])
            part = (
                signals.loc[test_mask]
                .copy()
                .rename(columns={"prob_long": "frozen_score_percentile"})
            )
            part["candidate_id"] = candidate
            part["predicted_adverse_net_return"] = prediction
            part["utility_score"] = utility_to_score(prediction)
            part["utility_action"] = np.isfinite(prediction) & (prediction > 0)
            part["decision_atr_close_fraction"] = features[test_mask, 1]
            destination.append(part)
            record[f"{candidate}_action_rows"] = int(part.utility_action.sum())
        fits.append(record)
        if on_fold is not None:
            on_fold(record)
    return (
        pd.concat(parts, ignore_index=True),
        pd.concat(controls, ignore_index=True),
        fits,
        targets,
    )
