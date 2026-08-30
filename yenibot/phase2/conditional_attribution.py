"""Conditional score perturbations with fixed ATR context and frozen payoff fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yenibot.phase2.net_utility import predict_ridge_payoff, utility_to_score


def conditional_groups(signals, *, ratio):
    if ratio not in (1.10, 1.05):
        raise ValueError("Only the two locked conditioning grids are supported")
    s = signals.copy()
    if not s.index.equals(pd.RangeIndex(len(s))) or s.empty:
        raise ValueError("Conditional signals need a nonempty positional index")
    s["decision_time"] = pd.to_datetime(s.decision_time, utc=True, errors="raise")
    a = s.decision_atr_close_fraction.to_numpy(dtype=float)
    p = s.frozen_score_percentile.to_numpy(dtype=float)
    if (
        s[["fold", "decision_time"]].isna().any().any()
        or s.decision_time.duplicated().any()
        or not s.decision_time.is_monotonic_increasing
        or not np.isfinite(a).all()
        or (a <= 0).any()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid conditional score/time/ATR data")
    s["atr_bin"] = np.floor(np.log(a) / np.log(ratio)).astype(int)
    s["calendar_month"] = s.decision_time.dt.strftime("%Y-%m")
    groups, records = [], []
    eligible = np.zeros(len(s), dtype=bool)
    for key, group in s.groupby(["fold", "calendar_month", "atr_bin"], sort=True):
        indices = group.index.to_numpy(dtype=int)
        groups.append(indices)
        distinct = int(group.frozen_score_percentile.nunique())
        eligible[indices] = len(indices) >= 2 and distinct >= 2
        records.append(
            {
                "fold": int(key[0]),
                "month": key[1],
                "atr_bin": int(key[2]),
                "rows": len(indices),
                "distinct_scores": distinct,
                "min_atr": float(a[indices].min()),
                "max_atr": float(a[indices].max()),
            }
        )
    selected = s.utility_action.to_numpy(dtype=bool)
    if not selected.any():
        raise ValueError("No originally selected rows for conditional attribution")
    audit = {
        "ratio": ratio,
        "groups": len(groups),
        "singleton_rows": sum(r["rows"] for r in records if r["rows"] < 2),
        "uninformative_rows": int((~eligible).sum()),
        "selected_row_exchangeable_fraction": float(eligible[selected].mean()),
        "coverage_sufficient": bool(eligible[selected].mean() >= 0.9),
        "maximum_within_group_atr_ratio": max(
            r["max_atr"] / r["min_atr"] for r in records
        ),
    }
    return tuple(groups), audit, pd.DataFrame(records)


def draw_conditional_mapping(count, groups, *, method, rng):
    mapping = np.arange(count)
    if method not in ("ordinary", "circular"):
        raise ValueError("Unknown conditional method")
    for indices in groups:
        mapping[indices] = (
            rng.permutation(indices)
            if method == "ordinary"
            else np.roll(indices, rng.integers(len(indices)))
        )
    return mapping


class FrozenPayoffReplay:
    """No estimator fit: evaluate only the already pinned per-fold parameters."""

    def __init__(self, signals, fits):
        self.atr = signals.decision_atr_close_fraction.to_numpy(dtype=float).copy()
        if (
            not signals.index.equals(pd.RangeIndex(len(signals)))
            or not np.isfinite(self.atr).all()
            or (self.atr <= 0).any()
        ):
            raise ValueError(
                "Frozen replay needs positional rows and positive finite ATR"
            )
        by_fold = {r["fold"]: r for r in fits}
        if len(by_fold) != len(fits) or set(by_fold) != set(signals.fold.unique()):
            raise ValueError("Frozen fit/cohort identities differ")
        self.parts = []
        for fold, group in signals.groupby("fold", sort=True):
            record = by_fold[fold]
            start = pd.to_datetime(group.decision_time, utc=True).min()
            if (
                record["training_max_fold"] >= fold
                or pd.Timestamp(record["training_outcome_max"]) >= start
                or pd.Timestamp(record["test_start"]) != start
            ):
                raise ValueError("Frozen fit violates historical time boundary")
            fit = record["candidate_fit"]
            vectors = [
                np.asarray(fit[k], dtype=float)
                for k in ("center", "scale", "coefficients")
            ]
            if (
                any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors)
                or (vectors[1] <= 0).any()
                or not np.isfinite(fit["intercept"])
            ):
                raise ValueError("Invalid frozen ridge parameters")
            self.parts.append((group.index.to_numpy(dtype=int), fit))

    def predict(self, score):
        p = np.asarray(score, dtype=float)
        if (
            p.shape != self.atr.shape
            or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()
        ):
            raise ValueError("Need one finite percentile per original row")
        x = np.column_stack([p, self.atr, p * self.atr])
        predicted = np.empty(len(p))
        for indices, fit in self.parts:
            predicted[indices] = predict_ridge_payoff(fit, x[indices])
        return predicted, utility_to_score(predicted)


def lag_one_within_fold(values, folds):
    x = np.asarray(values, dtype=float)
    same_fold = np.asarray(folds)[1:] == np.asarray(folds)[:-1]
    a, b = x[:-1][same_fold], x[1:][same_fold]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def mapping_diagnostics(signals, mapping, transformed_scores):
    a = signals.decision_atr_close_fraction.to_numpy(dtype=float)
    p = signals.frozen_score_percentile.to_numpy(dtype=float)
    selected = signals.utility_action.to_numpy(dtype=bool)
    ratio = np.maximum(a / a[mapping], a[mapping] / a)
    return {
        "changed_score_fraction": float(np.mean(p[mapping] != p)),
        "changed_original_selected_score_fraction": float(
            np.mean(p[mapping][selected] != p[selected])
        ),
        "maximum_donor_recipient_atr_ratio": float(ratio.max()),
        "selected_donor_recipient_atr_ratio_p95": float(
            np.quantile(ratio[selected], 0.95)
        ),
        "raw_score_lag_one": lag_one_within_fold(p[mapping], signals.fold),
        "action_lag_one": lag_one_within_fold(transformed_scores >= 0.5, signals.fold),
    }


def conditional_tail_summary(trials, actual, *, statistic):
    values = trials[statistic].to_numpy(dtype=float)
    observed = float(actual[statistic])
    if len(values) != 500 or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError("Conditional v1 requires exactly 500 finite trials")
    extreme = (values >= observed) | np.isclose(
        values, observed, rtol=1e-12, atol=1e-14
    )
    return {
        "actual": observed,
        "null_median": float(np.median(values)),
        "null_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "actual_minus_null_median": float(observed - np.median(values)),
        "upper_tail_monte_carlo_p": float((1 + extreme.sum()) / (len(values) + 1)),
    }
