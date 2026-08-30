"""One validation-only ridge payoff probe; immutable execution and no test fit."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT, Phase2StrategyContract
from yenibot.phase2.execution_cache import build_fold_execution_cache
from yenibot.phase2.full_oof import validation_cdf_test_scores


def utility_to_score(predicted: np.ndarray) -> np.ndarray:
    """Monotone encoding for the engine, NOT a probability; zero must abstain."""
    predicted = np.asarray(predicted, dtype=float)
    positive = np.isfinite(predicted) & (predicted > 0)
    score = (
        0.5 + np.arctan(np.where(np.isfinite(predicted), predicted, 0) / 0.001) / np.pi
    )
    score[positive] = np.maximum(score[positive], np.nextafter(0.5, 1.0))
    score[~positive] = np.minimum(score[~positive], np.nextafter(0.5, 0.0))
    score[~np.isfinite(predicted)] = 0.0
    return score


def _features(frame: pd.DataFrame, percentile: np.ndarray) -> np.ndarray:
    # No next-open, future OHLC, test label or forward return enters these inputs.
    atr = pd.to_numeric(frame.atr_14, errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame.close, errors="coerce").to_numpy(dtype=float)
    fraction = np.divide(atr, close, out=np.full(len(frame), np.nan), where=close > 0)
    fraction[atr <= 0] = np.nan
    return np.column_stack([percentile, fraction, percentile * fraction])


def fit_ridge_payoff(x: np.ndarray, y: np.ndarray, *, alpha: float) -> dict[str, Any]:
    """Minimize centered squared error + alpha * ||standardized coefficient||^2."""
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and positive")
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),) or not len(x):
        raise ValueError("Ridge requires aligned three-feature observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge fit inputs must be finite")
    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardized = (x - center) / scale
    intercept = float(y.mean())
    coefficients = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(3),
        standardized.T @ (y - intercept),
    )
    return {
        "alpha": float(alpha),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "fit_rows": len(y),
        "fit_target_mean": intercept,
    }


def predict_ridge_payoff(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    predictions = np.full(len(x), np.nan)
    valid = np.isfinite(x).all(axis=1)
    predictions[valid] = (
        (x[valid] - np.asarray(fit["center"])) / np.asarray(fit["scale"])
    ) @ np.asarray(fit["coefficients"]) + fit["intercept"]
    return predictions


def validation_opportunity_targets(
    validation: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    contract: Phase2StrategyContract = DEFAULT_PHASE2_CONTRACT,
) -> pd.DataFrame:
    """Overlapping training opportunities; NOT a realizable trade portfolio."""
    frame = validation.sort_values("timestamp").reset_index(drop=True)
    if frame.empty or set(frame.split.unique()) != {"val"}:
        raise ValueError("Opportunity targets accept validation rows only")
    bars = frame[["timestamp", "open", "high", "low", "close", "atr_14"]].rename(
        columns={"timestamp": "bar_open_time"}
    )
    bars["bar_close_time"] = pd.to_datetime(
        bars.bar_open_time, utc=True
    ) + pd.Timedelta(hours=1)
    signals = pd.DataFrame(
        {
            "decision_time": bars.bar_close_time,
            "prob_long": 1.0,
            "fold": frame.fold,
        }
    )
    # The target contract is deliberately independent of the fitted action score.
    target_contract = replace(contract, score_column="prob_long")
    adverse = next(c for c in contract.cost_scenarios if c.name == "adverse")
    cache = build_fold_execution_cache(
        bars, signals, contract=target_contract, scenario=adverse
    )
    valid_entry = cache.entry_indices >= 0
    exit_times = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    exit_times.loc[valid_entry] = bars.bar_close_time.iloc[
        cache.exit_indices[valid_entry]
    ].to_numpy()
    # Bar-close proxy is conservative for a gap-open fill; all observations
    # must also end within validation, never in test or the embargo.
    eligible = valid_entry & ~cache.censored & ~cache.data_gaps
    eligible &= exit_times.lt(pd.Timestamp(test_start)).to_numpy()
    reasons = np.full(len(frame), "eligible", dtype=object)
    reasons[~valid_entry] = "no_eligible_next_entry"
    reasons[cache.censored] = "censored_outcome"
    reasons[cache.data_gaps] = "data_gap_outcome"
    reasons[valid_entry & ~exit_times.lt(pd.Timestamp(test_start)).to_numpy()] = (
        "outcome_not_mature_before_test"
    )
    return pd.DataFrame(
        {
            "timestamp": frame.timestamp,
            "fold": frame.fold,
            "decision_time": signals.decision_time,
            "entry_index": cache.entry_indices,
            "exit_index": cache.exit_indices,
            "outcome_time_conservative": exit_times,
            "eligible": eligible,
            "exclusion_reason": reasons,
            "adverse_net_target": np.where(eligible, cache.net_returns, np.nan),
        }
    )


def build_net_utility_signals(
    frame: pd.DataFrame,
    *,
    source_spec: dict[str, Any],
    utility_spec: dict[str, Any],
    on_fold: Callable[[dict[str, Any], pd.DataFrame], None] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Fit exactly once per eligible fold, then freeze its test action function."""
    test, _ = validation_cdf_test_scores(frame, spec=source_spec)
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["fold"] = pd.to_numeric(frame.fold).astype(int)
    frame["prob_long"] = pd.to_numeric(frame.prob_long)
    candidate_id = f"{source_spec['source_run_id']}_validation_net_utility_hurdle_v1"
    fits, targets, parts = [], [], []
    for fold, fold_test in test.groupby("fold", sort=True):
        val = (
            frame.loc[frame.fold.eq(fold) & frame.split.eq("val")]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        reference = np.sort(val.prob_long.to_numpy(dtype=float))
        val_percentile = np.searchsorted(
            reference, val.prob_long.to_numpy(), side="right"
        ) / len(val)
        target = validation_opportunity_targets(
            val, test_start=fold_test.timestamp.min()
        )
        x = _features(val, val_percentile)
        valid = target.eligible.to_numpy() & np.isfinite(x).all(axis=1)
        target["fit_eligible"] = valid
        target["validation_score_percentile"] = val_percentile
        target["decision_atr_close_fraction"] = x[:, 1]
        target["score_atr_product"] = x[:, 2]
        test_x = _features(fold_test, fold_test.prob_long.to_numpy())
        fit_record = {
            "fold": int(fold),
            "validation_rows": len(val),
            "eligible_fit_rows": int(valid.sum()),
            "excluded_rows": int((~valid).sum()),
            "validation_end": pd.Timestamp(val.timestamp.max()).isoformat(),
            "test_start": pd.Timestamp(fold_test.timestamp.min()).isoformat(),
            "outcome_max": target.loc[valid, "outcome_time_conservative"]
            .max()
            .isoformat()
            if valid.any()
            else None,
            "fit_performed": False,
            "test_labels_used_in_fit": False,
        }
        if valid.sum() < utility_spec["minimum_fit_rows"]:
            predicted = np.full(len(fold_test), np.nan)
            fit_record["abstain_reason"] = (
                "insufficient_eligible_validation_opportunities"
            )
        else:
            fit = fit_ridge_payoff(
                x[valid],
                target.loc[valid, "adverse_net_target"].to_numpy(),
                alpha=utility_spec["ridge_alpha"],
            )
            predicted = predict_ridge_payoff(fit, test_x)
            fit_record.update(fit_performed=True, fit=fit)
        part = fold_test[
            [
                "timestamp",
                "fold",
                "split",
                "raw_prob_long",
                "prob_long",
                "label",
                "forward_return",
                "tb_return",
            ]
        ].copy()
        part = part.rename(
            columns={
                "timestamp": "source_bar_open_time",
                "prob_long": "frozen_score_percentile",
            }
        )
        part["decision_time"] = part.source_bar_open_time + pd.Timedelta(hours=1)
        part["candidate_id"] = candidate_id
        part["predicted_adverse_net_return"] = predicted
        part["utility_score"] = utility_to_score(predicted)
        part["utility_action"] = np.isfinite(predicted) & (predicted > 0)
        part["decision_atr_close_fraction"] = test_x[:, 1]
        fit_record["test_action_count"] = int(part.utility_action.sum())
        fit_record["test_invalid_prediction_count"] = int(
            (~np.isfinite(predicted)).sum()
        )
        fits.append(fit_record)
        targets.append(target)
        parts.append(part)
        if on_fold is not None:
            on_fold(fit_record, target)
    return (
        pd.concat(parts, ignore_index=True),
        fits,
        pd.concat(targets, ignore_index=True),
    )


def paired_fold_block_intervals(
    candidate, reference, *, block_lengths, replicates, seed
):
    """Paired fold units, not overlapping hourly labels; no independence claim."""
    merged = (
        candidate[["fold", "compounded_net_return"]]
        .merge(
            reference[["fold", "compounded_net_return"]],
            on="fold",
            how="outer",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )
        .sort_values("fold")
    )
    if merged.isna().any().any() or len(merged) < max(block_lengths) * 2:
        raise ValueError("Paired bootstrap requires complete aligned fold sets")
    delta = (
        merged.compounded_net_return_candidate - merged.compounded_net_return_reference
    ).to_numpy()
    if not np.isfinite(delta).all() or replicates < 100:
        raise ValueError(
            "Paired bootstrap needs finite fold deltas and >=100 replicates"
        )
    rows = []
    for length in block_lengths:
        if length < 1:
            raise ValueError("Block lengths must be positive")
        rng = np.random.default_rng(seed + int(length))
        starts = rng.integers(
            0,
            len(delta) - length + 1,
            size=(replicates, int(np.ceil(len(delta) / length))),
        )
        indices = (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[
            :, : len(delta)
        ]
        means = delta[indices].mean(axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
        rows.append(
            {
                "block_length_folds": int(length),
                "fold_count": len(delta),
                "replicates": replicates,
                "seed": seed + int(length),
                "paired_mean_fold_return_delta": float(delta.mean()),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "lower_bound_positive": bool(lower > 0),
            }
        )
    merged["paired_delta"] = delta
    return rows, merged
