"""Compact root-cause diagnostics for frozen future-OOS evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from yenibot.experiment.common import _rank_ic_for_frame

__all__ = [
    "empty_future_oos_diagnostic_frames",
    "empty_future_oos_model_metrics",
    "future_oos_diagnostic_frames",
    "future_oos_failure_summary",
    "future_oos_model_metrics",
    "future_oos_failure_markdown",
]

_METRIC_COLUMNS = [
    "rows",
    "rank_ic",
    "label_prevalence",
    "pred_long_rate",
    "precision",
    "recall",
    "f1",
    "prauc",
    "prauc_lift_vs_prevalence",
    "precision_lift_vs_prevalence",
    "top_10_lift",
    "top_10_forward_return",
    "selected_forward_return",
    "score_mean",
    "score_std",
]
_TEMPORAL_BLOCK_COLUMNS = [
    "candidate_id",
    "block_id",
    "data_start",
    "data_end",
    "threshold",
    *_METRIC_COLUMNS,
]
_SCORE_BAND_COLUMNS = [
    "candidate_id",
    "score_decile",
    "band",
    "rows",
    "score_min",
    "score_mean",
    "score_max",
    "label_rate",
    "label_lift",
    "mean_forward_return",
    "mean_tb_return",
    "frozen_threshold_selection_rate",
]
_REGIME_METRIC_COLUMNS = [
    "candidate_id",
    "regime",
    "threshold",
    *_METRIC_COLUMNS,
]
_ENSEMBLE_DISAGREEMENT_COLUMNS = [
    "candidate_id",
    "rows",
    "model_count_min",
    "model_count_max",
    "prob_long_model_std_mean",
    "prob_long_model_std_p90",
    "prob_long_model_range_mean",
    "error_model_std_mean",
    "correct_model_std_mean",
    "selected_model_std_mean",
    "not_selected_model_std_mean",
    "high_disagreement_error_rate",
]
_MODEL_METRIC_COLUMNS = [
    "candidate_id",
    "profile",
    "model_fold",
    "threshold",
    *_METRIC_COLUMNS,
]


def empty_future_oos_diagnostic_frames() -> dict[str, pd.DataFrame]:
    """Return header-preserving empty frames for every compact OOS report."""

    return {
        "temporal_blocks": pd.DataFrame(columns=_TEMPORAL_BLOCK_COLUMNS),
        "score_bands": pd.DataFrame(columns=_SCORE_BAND_COLUMNS),
        "regime_metrics": pd.DataFrame(columns=_REGIME_METRIC_COLUMNS),
        "ensemble_disagreement": pd.DataFrame(
            columns=_ENSEMBLE_DISAGREEMENT_COLUMNS
        ),
    }


def empty_future_oos_model_metrics() -> pd.DataFrame:
    """Return an empty per-model report that remains readable as CSV."""

    return pd.DataFrame(columns=_MODEL_METRIC_COLUMNS)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def _metrics(frame: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    clean = frame.copy().replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["prob_long", "label", "forward_return"])
    if clean.empty:
        return {"rows": 0}
    labels = clean["label"].astype(int)
    scores = pd.to_numeric(clean["prob_long"], errors="coerce")
    returns = pd.to_numeric(clean["forward_return"], errors="coerce")
    selected = scores >= threshold
    prevalence = float(labels.mean())
    pred_rate = float(selected.mean())
    precision = float(precision_score(labels, selected, zero_division=0))
    recall = float(recall_score(labels, selected, zero_division=0))
    f1 = float(f1_score(labels, selected, zero_division=0))
    prauc = (
        float(average_precision_score(labels, scores))
        if labels.nunique(dropna=True) > 1
        else np.nan
    )
    top_count = max(1, int(np.ceil(len(clean) * 0.10)))
    top = clean.assign(_score=scores).nlargest(top_count, "_score")
    top_label_rate = float(pd.to_numeric(top["label"], errors="coerce").mean())
    return {
        "rows": int(len(clean)),
        "rank_ic": _rank_ic_for_frame(clean),
        "label_prevalence": prevalence,
        "pred_long_rate": pred_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prauc": prauc,
        "prauc_lift_vs_prevalence": _safe_ratio(prauc, prevalence),
        "precision_lift_vs_prevalence": _safe_ratio(precision, prevalence),
        "top_10_lift": _safe_ratio(top_label_rate, prevalence),
        "top_10_forward_return": float(
            pd.to_numeric(top["forward_return"], errors="coerce").mean()
        ),
        "selected_forward_return": (
            float(returns[selected].mean()) if bool(selected.any()) else np.nan
        ),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=0)),
    }


def _temporal_blocks(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    block_hours: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in predictions.groupby("candidate_id", sort=False):
        frame = candidate.sort_values("timestamp").copy()
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        start = timestamps.min()
        block_id = ((timestamps - start) / pd.Timedelta(hours=block_hours)).astype(int)
        for index, part in frame.groupby(block_id, sort=True):
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "block_id": int(index),
                    "data_start": pd.to_datetime(part["timestamp"], utc=True).min(),
                    "data_end": pd.to_datetime(part["timestamp"], utc=True).max(),
                    "threshold": threshold,
                    **_metrics(part, threshold=threshold),
                }
            )
    return pd.DataFrame(rows, columns=_TEMPORAL_BLOCK_COLUMNS)


def _score_bands(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in predictions.groupby("candidate_id", sort=False):
        frame = candidate.copy().replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["prob_long", "label", "forward_return"])
        if frame.empty:
            continue
        ranks = frame["prob_long"].rank(method="first", pct=True)
        frame["score_decile"] = np.minimum((ranks * 10).astype(int), 9)
        base_rate = float(frame["label"].mean())
        for decile, part in frame.groupby("score_decile", sort=True):
            label_rate = float(pd.to_numeric(part["label"], errors="coerce").mean())
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "score_decile": int(decile),
                    "band": f"{int(decile) * 10:02d}-{(int(decile) + 1) * 10:02d}%",
                    "rows": int(len(part)),
                    "score_min": float(part["prob_long"].min()),
                    "score_mean": float(part["prob_long"].mean()),
                    "score_max": float(part["prob_long"].max()),
                    "label_rate": label_rate,
                    "label_lift": _safe_ratio(label_rate, base_rate),
                    "mean_forward_return": float(
                        pd.to_numeric(part["forward_return"], errors="coerce").mean()
                    ),
                    "mean_tb_return": (
                        float(pd.to_numeric(part["tb_return"], errors="coerce").mean())
                        if "tb_return" in part.columns
                        else np.nan
                    ),
                    "frozen_threshold_selection_rate": float(
                        (pd.to_numeric(part["prob_long"], errors="coerce") >= threshold).mean()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=_SCORE_BAND_COLUMNS)


def _regime_metrics(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    regime_columns = sorted(
        column for column in predictions.columns if column.startswith("regime_prob_")
    )
    if not regime_columns:
        return pd.DataFrame(columns=_REGIME_METRIC_COLUMNS)
    frame = predictions.copy()
    frame["regime"] = (
        frame[regime_columns]
        .apply(pd.to_numeric, errors="coerce")
        .idxmax(axis=1)
        .str.replace("regime_prob_", "", regex=False)
    )
    rows: list[dict[str, Any]] = []
    for (candidate_id, regime), part in frame.groupby(
        ["candidate_id", "regime"],
        sort=True,
    ):
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "regime": str(regime),
                "threshold": threshold,
                **_metrics(part, threshold=threshold),
            }
        )
    return pd.DataFrame(rows, columns=_REGIME_METRIC_COLUMNS)


def _ensemble_disagreement(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    required = {"prob_long_model_std", "prob_long_model_min", "prob_long_model_max"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame(columns=_ENSEMBLE_DISAGREEMENT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for candidate_id, part in predictions.groupby("candidate_id", sort=False):
        selected = pd.to_numeric(part["prob_long"], errors="coerce") >= threshold
        errors = selected.astype(int) != pd.to_numeric(part["label"], errors="coerce").astype(int)
        model_std = pd.to_numeric(part["prob_long_model_std"], errors="coerce")
        model_range = (
            pd.to_numeric(part["prob_long_model_max"], errors="coerce")
            - pd.to_numeric(part["prob_long_model_min"], errors="coerce")
        )
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "rows": int(len(part)),
                "model_count_min": int(
                    pd.to_numeric(part["model_fold_count"], errors="coerce").min()
                ),
                "model_count_max": int(
                    pd.to_numeric(part["model_fold_count"], errors="coerce").max()
                ),
                "prob_long_model_std_mean": float(model_std.mean()),
                "prob_long_model_std_p90": float(model_std.quantile(0.90)),
                "prob_long_model_range_mean": float(model_range.mean()),
                "error_model_std_mean": float(model_std[errors].mean()),
                "correct_model_std_mean": float(model_std[~errors].mean()),
                "selected_model_std_mean": float(model_std[selected].mean()),
                "not_selected_model_std_mean": float(model_std[~selected].mean()),
                "high_disagreement_error_rate": float(
                    errors[model_std >= model_std.quantile(0.90)].mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=_ENSEMBLE_DISAGREEMENT_COLUMNS)


def future_oos_diagnostic_frames(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    block_hours: int = 168,
) -> dict[str, pd.DataFrame]:
    """Build compact diagnostics without fitting or changing the frozen policy."""

    return {
        "temporal_blocks": _temporal_blocks(
            predictions,
            threshold=threshold,
            block_hours=block_hours,
        ),
        "score_bands": _score_bands(predictions, threshold=threshold),
        "regime_metrics": _regime_metrics(predictions, threshold=threshold),
        "ensemble_disagreement": _ensemble_disagreement(
            predictions,
            threshold=threshold,
        ),
    }


def future_oos_model_metrics(
    raw_predictions: pd.DataFrame,
    *,
    candidate_id: str,
    profile: str,
    threshold: float,
) -> pd.DataFrame:
    """Summarize each frozen component model on the same untouched OOS rows."""

    if raw_predictions.empty or "model_fold" not in raw_predictions.columns:
        return empty_future_oos_model_metrics()
    rows = []
    for model_fold, part in raw_predictions.groupby("model_fold", sort=True):
        rows.append(
            {
                "candidate_id": candidate_id,
                "profile": profile,
                "model_fold": int(model_fold),
                "threshold": threshold,
                **_metrics(part, threshold=threshold),
            }
        )
    return pd.DataFrame(rows, columns=_MODEL_METRIC_COLUMNS)


def future_oos_failure_summary(
    evaluation_row: dict[str, Any],
    *,
    temporal_blocks: pd.DataFrame,
    ensemble_disagreement: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Classify the failure mechanism without proposing same-window tuning."""

    failed_gates = {
        item for item in str(evaluation_row.get("failed_gates", "")).split(";") if item
    }
    ranking_failed = bool(
        failed_gates.intersection({"rank_ic", "rank_ic_lower_ci", "prauc_lift", "top_10_lift"})
    )
    payoff_failed = bool(
        failed_gates.intersection({"top_10_forward_return", "selected_forward_return"})
    )
    threshold_failed = bool(
        failed_gates.intersection({"precision_lift", "f1_skill", "pred_long_rate"})
    )
    if ranking_failed and payoff_failed:
        mechanism = "ranking_and_payoff_breakdown_not_threshold_only"
    elif ranking_failed:
        mechanism = "ranking_breakdown"
    elif payoff_failed:
        mechanism = "economic_payoff_breakdown"
    elif threshold_failed:
        mechanism = "threshold_transfer_breakdown"
    else:
        mechanism = "no_failure_or_unclassified"
    temporal_positive = (
        float((pd.to_numeric(temporal_blocks["rank_ic"], errors="coerce") > 0).mean())
        if not temporal_blocks.empty and "rank_ic" in temporal_blocks.columns
        else np.nan
    )
    fold_age_rank_ic_correlation = np.nan
    recent_model_rank_ic_mean = np.nan
    old_model_rank_ic_mean = np.nan
    recency_signal = "unavailable"
    if (
        not model_metrics.empty
        and {"model_fold", "rank_ic"}.issubset(model_metrics.columns)
    ):
        model_frame = model_metrics.copy()
        model_frame["model_fold"] = pd.to_numeric(
            model_frame["model_fold"],
            errors="coerce",
        )
        model_frame["rank_ic"] = pd.to_numeric(
            model_frame["rank_ic"],
            errors="coerce",
        )
        model_frame = model_frame.dropna(subset=["model_fold", "rank_ic"]).sort_values(
            "model_fold"
        )
        if len(model_frame) >= 4:
            fold_age_rank_ic_correlation = float(
                model_frame["model_fold"].corr(
                    model_frame["rank_ic"],
                    method="spearman",
                )
            )
            quartile = max(1, len(model_frame) // 4)
            old_model_rank_ic_mean = float(model_frame.head(quartile)["rank_ic"].mean())
            recent_model_rank_ic_mean = float(
                model_frame.tail(quartile)["rank_ic"].mean()
            )
            delta = recent_model_rank_ic_mean - old_model_rank_ic_mean
            if fold_age_rank_ic_correlation >= 0.25 and delta > 0:
                recency_signal = "newer_models_outperform_older_models_diagnostic_only"
            elif fold_age_rank_ic_correlation <= -0.25 and delta < 0:
                recency_signal = "older_models_outperform_newer_models_diagnostic_only"
            else:
                recency_signal = "no_clear_monotonic_model_recency_edge"
    return {
        "candidate_id": evaluation_row.get("candidate_id"),
        "evaluation_passed": bool(evaluation_row.get("evidence_passed", False)),
        "failed_gates": sorted(failed_gates),
        "primary_failure_mechanism": mechanism,
        "ranking_failed": ranking_failed,
        "payoff_failed": payoff_failed,
        "threshold_failed": threshold_failed,
        "temporal_positive_rank_ic_fraction": temporal_positive,
        "ensemble_disagreement_available": not ensemble_disagreement.empty,
        "model_fold_rank_ic_spearman": fold_age_rank_ic_correlation,
        "recent_model_rank_ic_mean": recent_model_rank_ic_mean,
        "old_model_rank_ic_mean": old_model_rank_ic_mean,
        "recency_signal": recency_signal,
        "same_window_tuning_allowed": False,
        "candidate_status": (
            "retired_after_failed_future_oos"
            if not bool(evaluation_row.get("evidence_passed", False))
            else "passed_future_oos"
        ),
        "next_research_hypothesis": (
            "causal_rolling_retraining_and_recency_aware_ensemble"
            if ranking_failed
            else "validation_only_threshold_transfer"
            if threshold_failed
            else "economic_label_and_payoff_alignment"
        ),
        "new_future_oos_anchor_required": not bool(
            evaluation_row.get("evidence_passed", False)
        ),
    }


def future_oos_failure_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Future OOS Failure Analysis",
        "",
        f"- Candidate: `{summary.get('candidate_id')}`",
        f"- Status: `{summary.get('candidate_status')}`",
        f"- Primary mechanism: `{summary.get('primary_failure_mechanism')}`",
        f"- Failed gates: `{';'.join(summary.get('failed_gates', [])) or 'none'}`",
        f"- Temporal positive Rank IC fraction: `{summary.get('temporal_positive_rank_ic_fraction')}`",
        f"- Model recency signal: `{summary.get('recency_signal')}`",
        f"- Model-fold/Rank-IC Spearman: `{summary.get('model_fold_rank_ic_spearman')}`",
        f"- Next research hypothesis: `{summary.get('next_research_hypothesis')}`",
        f"- Same-window tuning allowed: `{summary.get('same_window_tuning_allowed')}`",
        f"- New future-OOS anchor required: `{summary.get('new_future_oos_anchor_required')}`",
        "",
        "The failed future-OOS window may be used for diagnosis and future hypothesis formation only. "
        "It must not be reused to promote, threshold-tune, or weight-tune the failed candidate.",
    ]
    return "\n".join(lines) + "\n"
