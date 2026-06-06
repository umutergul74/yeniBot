"""Score separation and bad-fold signature diagnostics."""

from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
    _mean_for_mask,
    _score_ks_statistic,
)

from yenibot.experiment.folds import (
    _entry_threshold_policy_frame,
)

from yenibot.experiment.training import (
    _test_predictions,
)

__all__ = [
    '_score_quantile',
    '_score_separation_forensics_frame',
    '_bad_fold_signature_frame',
    '_score_separation_markdown',
]

def _score_quantile(part: pd.DataFrame, label_value: int, quantile: float) -> float:
    values = pd.to_numeric(part.loc[part["label"].astype(int) == int(label_value), "prob_long"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(float(quantile))) if not values.empty else np.nan

def _score_separation_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "count",
        "label_long_rate",
        "rank_ic",
        "rank_ic_bucket",
        "score_gap_pos_minus_neg",
        "score_gap_z",
        "score_ks",
        "prob_long_pos_mean",
        "prob_long_neg_mean",
        "prob_long_pos_p25",
        "prob_long_pos_p50",
        "prob_long_pos_p75",
        "prob_long_neg_p25",
        "prob_long_neg_p50",
        "prob_long_neg_p75",
        "prob_long_std",
        "prob_long_iqr",
        "official_threshold",
        "official_f1",
        "official_precision",
        "official_recall",
        "official_pred_long_rate",
        "selected_threshold",
        "selected_f1",
        "selected_pred_long_rate",
        "constrained_threshold",
        "constrained_f1",
        "constrained_pred_long_rate",
        "top_10_lift_vs_base",
        "top_10_forward_return",
        "primary_issue",
    ]
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "label", "prob_long"}.issubset(predictions.columns):
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty:
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        score_bands = diagnostics.get("score_band_by_fold")
        top10_by_id: dict[int, dict[str, Any]] = {}
        if (
            isinstance(score_bands, pd.DataFrame)
            and not score_bands.empty
            and {"fold", "band"}.issubset(score_bands.columns)
        ):
            top10 = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            top10_by_id = {int(row["fold"]): row.to_dict() for _, row in top10.dropna(subset=["fold"]).iterrows()}

        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, part in test_predictions.groupby("fold"):
            fold_id = int(fold)
            part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if part.empty:
                continue
            labels = part["label"].astype(int)
            scores = pd.to_numeric(part["prob_long"], errors="coerce")
            pos_scores = scores.loc[labels == 1]
            neg_scores = scores.loc[labels == 0]
            pos_mean = float(pos_scores.mean()) if not pos_scores.empty else np.nan
            neg_mean = float(neg_scores.mean()) if not neg_scores.empty else np.nan
            score_std = float(scores.std(ddof=0)) if scores.notna().any() else np.nan
            score_gap = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan
            rank_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            top10_row = top10_by_id.get(fold_id, {})
            rank_ic_value = _float(rank_row, "rank_ic")
            official_f1 = _float(threshold_row, "test_f1_at_official_threshold", _float(threshold_row, "test_f1_at_constrained_threshold"))
            top10_lift = _float(top10_row, "lift_vs_base")
            top10_return = _float(top10_row, "mean_forward_return")
            if np.isfinite(rank_ic_value) and rank_ic_value < 0:
                issue = "negative_rank_ic"
            elif np.isfinite(score_gap) and score_gap <= 0:
                issue = "score_reversal"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                issue = "official_f1_gap"
            elif np.isfinite(top10_lift) and top10_lift < 1.0:
                issue = "top_10_label_lift_gap"
            elif np.isfinite(top10_return) and top10_return <= 0.0:
                issue = "top_10_payoff_gap"
            else:
                issue = "ok"
            if np.isfinite(rank_ic_value) and rank_ic_value < 0:
                bucket = "negative_rank_ic"
            elif np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic:
                bucket = "below_target_rank_ic"
            else:
                bucket = "rank_ic_ok"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "start": str(part["timestamp"].min()) if "timestamp" in part.columns else str(rank_row.get("start", "")),
                    "end": str(part["timestamp"].max()) if "timestamp" in part.columns else str(rank_row.get("end", "")),
                    "count": int(len(part)),
                    "label_long_rate": float(labels.mean()) if len(labels) else np.nan,
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    "score_gap_pos_minus_neg": score_gap,
                    "score_gap_z": float(score_gap / score_std) if np.isfinite(score_gap) and np.isfinite(score_std) and score_std > 0 else np.nan,
                    "score_ks": _score_ks_statistic(pos_scores, neg_scores),
                    "prob_long_pos_mean": pos_mean,
                    "prob_long_neg_mean": neg_mean,
                    "prob_long_pos_p25": _score_quantile(part, 1, 0.25),
                    "prob_long_pos_p50": _score_quantile(part, 1, 0.50),
                    "prob_long_pos_p75": _score_quantile(part, 1, 0.75),
                    "prob_long_neg_p25": _score_quantile(part, 0, 0.25),
                    "prob_long_neg_p50": _score_quantile(part, 0, 0.50),
                    "prob_long_neg_p75": _score_quantile(part, 0, 0.75),
                    "prob_long_std": score_std,
                    "prob_long_iqr": float(scores.quantile(0.75) - scores.quantile(0.25)) if scores.notna().any() else np.nan,
                    "official_threshold": _float(threshold_row, "official_threshold"),
                    "official_f1": official_f1,
                    "official_precision": _float(threshold_row, "test_precision_at_official_threshold"),
                    "official_recall": _float(threshold_row, "test_recall_at_official_threshold"),
                    "official_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_official_threshold"),
                    "selected_threshold": _float(threshold_row, "selected_threshold"),
                    "selected_f1": _float(threshold_row, "test_f1_at_selected_threshold"),
                    "selected_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_selected_threshold"),
                    "constrained_threshold": _float(threshold_row, "constrained_threshold"),
                    "constrained_f1": _float(threshold_row, "test_f1_at_constrained_threshold"),
                    "constrained_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
                    "top_10_lift_vs_base": top10_lift,
                    "top_10_forward_return": top10_return,
                    "primary_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "rank_ic"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )

def _bad_fold_signature_frame(score_forensics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "bad_definition",
        "fold_count",
        "bad_fold_count",
        "good_fold_count",
        "bad_rank_ic_mean",
        "good_rank_ic_mean",
        "bad_score_gap_mean",
        "good_score_gap_mean",
        "score_gap_delta_bad_minus_good",
        "bad_score_ks_mean",
        "good_score_ks_mean",
        "bad_label_long_rate_mean",
        "good_label_long_rate_mean",
        "label_long_rate_delta_bad_minus_good",
        "bad_official_f1_mean",
        "good_official_f1_mean",
        "official_f1_delta_bad_minus_good",
        "bad_official_pred_long_rate_mean",
        "good_official_pred_long_rate_mean",
        "bad_top_10_lift_mean",
        "good_top_10_lift_mean",
        "bad_top_10_forward_return_mean",
        "good_top_10_forward_return_mean",
        "bad_primary_issue_counts",
        "likely_signature",
        "recommended_next_action",
    ]
    if score_forensics.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in score_forensics.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank_values = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank_values < 0.0
        if int(bad_mask.sum()) < 3:
            bad_mask = rank_values < target_rank_ic
            definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        else:
            definition = "negative_rank_ic"
        good_mask = rank_values >= target_rank_ic
        bad_count = int(bad_mask.sum())
        good_count = int(good_mask.sum())
        if bad_count == 0:
            continue
        bad_score_gap = _mean_for_mask(part, bad_mask, "score_gap_pos_minus_neg")
        good_score_gap = _mean_for_mask(part, good_mask, "score_gap_pos_minus_neg")
        score_gap_delta = bad_score_gap - good_score_gap if np.isfinite(bad_score_gap) and np.isfinite(good_score_gap) else np.nan
        bad_label_rate = _mean_for_mask(part, bad_mask, "label_long_rate")
        good_label_rate = _mean_for_mask(part, good_mask, "label_long_rate")
        label_delta = bad_label_rate - good_label_rate if np.isfinite(bad_label_rate) and np.isfinite(good_label_rate) else np.nan
        bad_f1 = _mean_for_mask(part, bad_mask, "official_f1")
        good_f1 = _mean_for_mask(part, good_mask, "official_f1")
        f1_delta = bad_f1 - good_f1 if np.isfinite(bad_f1) and np.isfinite(good_f1) else np.nan
        bad_top_lift = _mean_for_mask(part, bad_mask, "top_10_lift_vs_base")
        good_top_lift = _mean_for_mask(part, good_mask, "top_10_lift_vs_base")
        bad_top_return = _mean_for_mask(part, bad_mask, "top_10_forward_return")
        good_top_return = _mean_for_mask(part, good_mask, "top_10_forward_return")
        issue_counts = part.loc[bad_mask, "primary_issue"].astype(str).value_counts().to_dict()
        signatures: list[str] = []
        if np.isfinite(score_gap_delta) and score_gap_delta < -0.01:
            signatures.append("score_separation_compresses_or_reverses")
        if np.isfinite(bad_top_lift) and bad_top_lift < 1.0:
            signatures.append("top_score_label_lift_fails")
        if np.isfinite(bad_top_return) and bad_top_return <= 0.0:
            signatures.append("top_score_payoff_reverses")
        if np.isfinite(label_delta) and abs(label_delta) >= 0.05:
            signatures.append("label_distribution_shift")
        if np.isfinite(f1_delta) and f1_delta < -0.05:
            signatures.append("official_threshold_f1_collapses")
        if not signatures:
            signatures.append("rank_ic_variance_without_single_score_signature")
        if "score_separation_compresses_or_reverses" in signatures:
            action = "inspect_bad_fold_feature_drift_and_add_only_pre_registered_score_separation_features"
        elif "top_score_payoff_reverses" in signatures:
            action = "do_not_promote_score_band_policy_until_future_oos_confirms_payoff"
        elif "label_distribution_shift" in signatures:
            action = "review_label_regime_balance_before_feature_changes"
        elif "official_threshold_f1_collapses" in signatures:
            action = "focus_on_fold_specific_score_separation_not_threshold_smoothing"
        else:
            action = "use_fold_level_feature_importance_or_future_oos_before_new_profile_search"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "bad_definition": definition,
                "fold_count": int(part["fold"].nunique()),
                "bad_fold_count": bad_count,
                "good_fold_count": good_count,
                "bad_rank_ic_mean": _mean_for_mask(part, bad_mask, "rank_ic"),
                "good_rank_ic_mean": _mean_for_mask(part, good_mask, "rank_ic"),
                "bad_score_gap_mean": bad_score_gap,
                "good_score_gap_mean": good_score_gap,
                "score_gap_delta_bad_minus_good": score_gap_delta,
                "bad_score_ks_mean": _mean_for_mask(part, bad_mask, "score_ks"),
                "good_score_ks_mean": _mean_for_mask(part, good_mask, "score_ks"),
                "bad_label_long_rate_mean": bad_label_rate,
                "good_label_long_rate_mean": good_label_rate,
                "label_long_rate_delta_bad_minus_good": label_delta,
                "bad_official_f1_mean": bad_f1,
                "good_official_f1_mean": good_f1,
                "official_f1_delta_bad_minus_good": f1_delta,
                "bad_official_pred_long_rate_mean": _mean_for_mask(part, bad_mask, "official_pred_long_rate"),
                "good_official_pred_long_rate_mean": _mean_for_mask(part, good_mask, "official_pred_long_rate"),
                "bad_top_10_lift_mean": bad_top_lift,
                "good_top_10_lift_mean": good_top_lift,
                "bad_top_10_forward_return_mean": bad_top_return,
                "good_top_10_forward_return_mean": good_top_return,
                "bad_primary_issue_counts": json.dumps(issue_counts, sort_keys=True),
                "likely_signature": ";".join(signatures),
                "recommended_next_action": action,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "bad_fold_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _score_separation_markdown(score_forensics: pd.DataFrame, bad_signature: pd.DataFrame) -> str:
    lines = ["# Bad Fold Score-Separation Forensics", ""]
    if score_forensics.empty and bad_signature.empty:
        lines.append("No score-separation rows were produced.")
        return "\n".join(lines)
    lines.append(
        "These diagnostics explain whether weak folds are caused by score separation, label-rate shifts, "
        "threshold behavior, or top-score payoff reversal. They are diagnostics only."
    )
    if not bad_signature.empty:
        lines.append("")
        lines.append("## Bad Fold Signatures")
        display_cols = [
            "candidate",
            "fold_scope",
            "bad_fold_count",
            "bad_rank_ic_mean",
            "bad_score_gap_mean",
            "good_score_gap_mean",
            "bad_top_10_lift_mean",
            "bad_top_10_forward_return_mean",
            "likely_signature",
            "recommended_next_action",
        ]
        visible = bad_signature[[column for column in display_cols if column in bad_signature.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not score_forensics.empty:
        lines.append("")
        lines.append(f"Fold-level rows: {len(score_forensics)}. See `score_separation_forensics.csv` for detail.")
    return "\n".join(lines)
