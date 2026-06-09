"""Causal classification policy, skill baselines, and validation charter diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
    _safe_average_precision,
    _write_json,
)

from yenibot.experiment.folds import (
    _entry_official_threshold_source,
    _entry_threshold_policy_frame,
)

from yenibot.experiment.rank_ic import (
    _score_quantile_threshold,
)

__all__ = [
    '_selection_stats_for_mask',
    '_classification_metrics_for_mask',
    '_threshold_score_quantile_review_frames',
    '_threshold_score_quantile_review_markdown',
    '_write_threshold_score_quantile_review',
    '_causal_score_quantile_mask',
    '_causal_threshold_policy_frames',
    '_causal_threshold_policy_markdown',
    '_write_causal_threshold_policy',
    '_classification_skill_frames',
    '_classification_skill_markdown',
    '_write_classification_skill',
    '_validation_charter_review_frame',
    '_validation_charter_review_markdown',
    '_write_validation_charter_review',
    '_validation_charter_proposal_frame',
    '_validation_charter_proposal_markdown',
    '_write_validation_charter_proposal',
]

def _selection_stats_for_mask(frame: pd.DataFrame, selected_mask: pd.Series) -> dict[str, float]:
    if frame.empty or "label" not in frame.columns:
        return {
            "actual_long_rate": np.nan,
            "label_lift_vs_base": np.nan,
            "mean_forward_return": np.nan,
            "mean_tb_return": np.nan,
        }
    labels = pd.to_numeric(frame["label"], errors="coerce")
    base_rate = float(labels.mean()) if labels.notna().any() else np.nan
    selected = frame.loc[selected_mask.fillna(False)].copy()
    if selected.empty:
        return {
            "actual_long_rate": 0.0,
            "label_lift_vs_base": 0.0 if np.isfinite(base_rate) and base_rate > 0 else np.nan,
            "mean_forward_return": np.nan,
            "mean_tb_return": np.nan,
        }
    actual_rate = float(pd.to_numeric(selected["label"], errors="coerce").mean())
    return {
        "actual_long_rate": actual_rate,
        "label_lift_vs_base": float(actual_rate / base_rate) if np.isfinite(base_rate) and base_rate > 0 else np.nan,
        "mean_forward_return": (
            float(pd.to_numeric(selected["forward_return"], errors="coerce").mean())
            if "forward_return" in selected.columns
            else np.nan
        ),
        "mean_tb_return": (
            float(pd.to_numeric(selected["tb_return"], errors="coerce").mean())
            if "tb_return" in selected.columns
            else np.nan
        ),
    }

def _classification_metrics_for_mask(labels: pd.Series, selected_mask: pd.Series) -> dict[str, float]:
    y_true = labels.astype(int)
    y_pred = selected_mask.fillna(False).astype(bool)
    tp = int(((y_true == 1) & y_pred).sum())
    fp = int(((y_true == 0) & y_pred).sum())
    fn = int(((y_true == 1) & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }

def _threshold_score_quantile_review_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold",
        "selection_rate_source",
        "target_selection_rate",
        "realized_selection_rate",
        "score_quantile_threshold",
        "official_f1",
        "official_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_actual_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "selection_guard",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold_count",
        "target_selection_rate_mean",
        "realized_selection_rate_mean",
        "score_quantile_threshold_mean",
        "test_f1_mean",
        "test_precision_mean",
        "test_recall_mean",
        "test_pred_long_rate_mean",
        "test_label_lift_vs_base_mean",
        "test_mean_forward_return_mean",
        "test_mean_tb_return_mean",
        "official_f1_mean",
        "f1_delta_vs_official",
        "official_pred_long_rate_mean",
        "pred_long_rate_delta_vs_official",
        "f1_pass_fold_rate",
        "precision_pass_fold_rate",
        "pred_rate_pass_fold_rate",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "policy_passed_diagnostic",
        "diagnostic_outcome",
        "recommended_action",
        "selection_guard",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    rates = _cfg(config, ["validation", "score_quantile_review", "selection_rates"], [0.30, 0.40, 0.50, 0.60, 0.70])
    fixed_rates: list[float] = []
    for value in rates:
        try:
            rate_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(rate_value):
            fixed_rates.append(rate_value)
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
        threshold_metrics = _entry_threshold_policy_frame(entry)
        metric_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold_raw, fold_part in predictions.groupby("fold"):
            fold = int(fold_raw)
            if "split" in fold_part.columns:
                validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
                test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            else:
                validation = pd.DataFrame()
                test = fold_part.copy()
            test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if test.empty:
                continue
            metrics_row = metric_by_fold.get(fold, {})
            official_f1 = _float(metrics_row, "test_f1_at_official_threshold")
            official_rate = _float(metrics_row, "test_pred_long_rate_at_official_threshold")
            policies: list[dict[str, Any]] = []
            for rate in fixed_rates:
                policies.append(
                    {
                        "policy_name": f"fixed_top_{int(round(rate * 100)):02d}",
                        "policy_type": "score_quantile_fixed_rate",
                        "target_selection_rate": rate,
                        "selection_rate_source": "configured_fixed_rate",
                    }
                )
            if not validation.empty:
                for source_name, threshold_column in [
                    ("validation_selected_rate", "selected_threshold"),
                    ("validation_constrained_rate", "constrained_threshold"),
                    ("validation_official_rate", "official_threshold"),
                ]:
                    threshold = _float(metrics_row, threshold_column)
                    if np.isfinite(threshold):
                        val_rate = float((pd.to_numeric(validation["prob_long"], errors="coerce") >= threshold).mean())
                        policies.append(
                            {
                                "policy_name": f"{source_name}_quantile",
                                "policy_type": "score_quantile_validation_rate",
                                "target_selection_rate": val_rate,
                                "selection_rate_source": source_name,
                            }
                        )
            for policy in policies:
                rate = float(policy["target_selection_rate"])
                if not np.isfinite(rate):
                    continue
                rate = float(max(0.0, min(1.0, rate)))
                threshold = _score_quantile_threshold(test, rate)
                if not np.isfinite(threshold) and not np.isinf(threshold):
                    continue
                scores = pd.to_numeric(test["prob_long"], errors="coerce")
                selected_mask = scores >= threshold
                class_metrics = _classification_metrics_for_mask(test["label"], selected_mask)
                selection_stats = _selection_stats_for_mask(test, selected_mask)
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "policy_name": policy["policy_name"],
                        "policy_type": policy["policy_type"],
                        "fold": fold,
                        "selection_rate_source": policy["selection_rate_source"],
                        "target_selection_rate": rate,
                        "realized_selection_rate": class_metrics["pred_long_rate"],
                        "score_quantile_threshold": threshold,
                        "official_f1": official_f1,
                        "official_pred_long_rate": official_rate,
                        "test_f1": class_metrics["f1"],
                        "test_precision": class_metrics["precision"],
                        "test_recall": class_metrics["recall"],
                        "test_pred_long_rate": class_metrics["pred_long_rate"],
                        "test_actual_long_rate": selection_stats["actual_long_rate"],
                        "test_label_lift_vs_base": selection_stats["label_lift_vs_base"],
                        "test_mean_forward_return": selection_stats["mean_forward_return"],
                        "test_mean_tb_return": selection_stats["mean_tb_return"],
                        "selection_guard": "uses_current_test_score_distribution_no_labels_diagnostic_only",
                    }
                )
    if not rows:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)

    by_fold = (
        pd.DataFrame(rows, columns=by_fold_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "policy_name", "fold"])
        .reset_index(drop=True)
    )
    summaries: list[dict[str, Any]] = []
    group_cols = ["candidate", "candidate_type", "fold_scope", "policy_name", "policy_type"]
    for keys, part in by_fold.groupby(group_cols, dropna=False):
        candidate, candidate_type, fold_scope, policy_name, policy_type = keys
        f1 = pd.to_numeric(part["test_f1"], errors="coerce")
        precision = pd.to_numeric(part["test_precision"], errors="coerce")
        pred_rate = pd.to_numeric(part["test_pred_long_rate"], errors="coerce")
        lift = pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce")
        fwd = pd.to_numeric(part["test_mean_forward_return"], errors="coerce")
        official_f1 = pd.to_numeric(part["official_f1"], errors="coerce")
        official_rate = pd.to_numeric(part["official_pred_long_rate"], errors="coerce")
        f1_pass = f1 >= min_long_f1
        precision_pass = precision >= min_precision
        pred_pass = pred_rate <= max_pred_long_rate
        f1_pass_rate = float(f1_pass.mean()) if f1.notna().any() else np.nan
        precision_pass_rate = float(precision_pass.mean()) if precision.notna().any() else np.nan
        pred_pass_rate = float(pred_pass.mean()) if pred_rate.notna().any() else np.nan
        positive_lift_rate = float((lift > 1.0).mean()) if lift.notna().any() else np.nan
        positive_fwd_rate = float((fwd > 0.0).mean()) if fwd.notna().any() else np.nan
        policy_passed = bool(
            np.isfinite(f1_pass_rate)
            and f1_pass_rate >= 0.75
            and np.isfinite(pred_pass_rate)
            and pred_pass_rate >= 0.75
            and np.isfinite(precision_pass_rate)
            and precision_pass_rate >= 0.75
        )
        f1_mean = float(f1.mean()) if f1.notna().any() else np.nan
        official_f1_mean = float(official_f1.mean()) if official_f1.notna().any() else np.nan
        f1_delta = f1_mean - official_f1_mean if np.isfinite(f1_mean) and np.isfinite(official_f1_mean) else np.nan
        if policy_passed and np.isfinite(f1_delta) and f1_delta > 0.005:
            outcome = "score_scale_transfer_candidate"
            action = "pre_register_score_quantile_policy_for_future_oos; not deployable from current holdout"
        elif np.isfinite(f1_mean) and f1_mean >= min_long_f1 and np.isfinite(float(pred_rate.mean())) and float(pred_rate.mean()) > max_pred_long_rate:
            outcome = "quantile_policy_too_broad"
            action = "do_not_promote; selection rate must be lower or score separation must improve"
        elif np.isfinite(f1_mean) and f1_mean < min_long_f1:
            outcome = "score_separation_not_solved_by_scale_normalization"
            action = "prioritize score-reversal/gating hypothesis over threshold-only work"
        else:
            outcome = "monitor_no_clear_threshold_fix"
            action = "diagnostic_only"
        summaries.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "fold_count": int(part["fold"].nunique()),
                "target_selection_rate_mean": float(pd.to_numeric(part["target_selection_rate"], errors="coerce").mean()),
                "realized_selection_rate_mean": float(pd.to_numeric(part["realized_selection_rate"], errors="coerce").mean()),
                "score_quantile_threshold_mean": float(pd.to_numeric(part["score_quantile_threshold"], errors="coerce").mean()),
                "test_f1_mean": f1_mean,
                "test_precision_mean": float(precision.mean()) if precision.notna().any() else np.nan,
                "test_recall_mean": float(pd.to_numeric(part["test_recall"], errors="coerce").mean()),
                "test_pred_long_rate_mean": float(pred_rate.mean()) if pred_rate.notna().any() else np.nan,
                "test_label_lift_vs_base_mean": float(lift.mean()) if lift.notna().any() else np.nan,
                "test_mean_forward_return_mean": float(fwd.mean()) if fwd.notna().any() else np.nan,
                "test_mean_tb_return_mean": float(pd.to_numeric(part["test_mean_tb_return"], errors="coerce").mean()),
                "official_f1_mean": official_f1_mean,
                "f1_delta_vs_official": f1_delta,
                "official_pred_long_rate_mean": float(official_rate.mean()) if official_rate.notna().any() else np.nan,
                "pred_long_rate_delta_vs_official": (
                    float(pred_rate.mean() - official_rate.mean())
                    if pred_rate.notna().any() and official_rate.notna().any()
                    else np.nan
                ),
                "f1_pass_fold_rate": f1_pass_rate,
                "precision_pass_fold_rate": precision_pass_rate,
                "pred_rate_pass_fold_rate": pred_pass_rate,
                "positive_lift_fold_rate": positive_lift_rate,
                "positive_forward_return_fold_rate": positive_fwd_rate,
                "policy_passed_diagnostic": policy_passed,
                "diagnostic_outcome": outcome,
                "recommended_action": action,
                "selection_guard": ";".join(sorted({str(value) for value in part["selection_guard"].dropna()})),
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "policy_passed_diagnostic", "test_f1_mean"],
            ascending=[True, True, True, False, False],
        )
        .reset_index(drop=True)
    )
    return summary, by_fold

def _threshold_score_quantile_review_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Threshold Score-Quantile Review", ""]
    lines.append(
        "This diagnostic asks whether normalizing scores by within-fold score quantiles would close the "
        "official F1 gap. It uses no labels to form the selection mask, but it does use the current test "
        "score distribution, so it is diagnostic-only and not a deployable Phase 1 policy."
    )
    if summary.empty:
        lines.extend(["", "No score-quantile threshold rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "test_f1_mean",
        "test_pred_long_rate_mean",
        "f1_delta_vs_official",
        "f1_pass_fold_rate",
        "positive_forward_return_fold_rate",
        "diagnostic_outcome",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append(f"By-fold rows: {len(by_fold)}. See `threshold_score_quantile_by_fold.csv`.")
    return "\n".join(lines)

def _write_threshold_score_quantile_review(path: Path, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "threshold_score_quantile_review.csv", index=False)
    by_fold.to_csv(path / "threshold_score_quantile_by_fold.csv", index=False)
    (path / "threshold_score_quantile_review.md").write_text(
        _threshold_score_quantile_review_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "threshold_score_quantile_review.json",
        {
            "summary": summary.to_dict(orient="records"),
            "by_fold": by_fold.to_dict(orient="records"),
        },
    )

def _causal_score_quantile_mask(
    validation_scores: pd.Series,
    test_scores: pd.Series,
    *,
    selection_rate: float,
    rolling_window: int,
    min_history: int,
) -> tuple[pd.Series, pd.Series]:
    history = (
        pd.to_numeric(validation_scores, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tolist()
    )
    if rolling_window > 0:
        history = history[-rolling_window:]
    selections: list[bool] = []
    thresholds: list[float] = []
    for value in pd.to_numeric(test_scores, errors="coerce").tolist():
        if len(history) < min_history or not np.isfinite(value):
            thresholds.append(np.nan)
            selections.append(False)
        else:
            threshold = float(np.quantile(np.asarray(history, dtype=float), 1.0 - selection_rate))
            thresholds.append(threshold)
            selections.append(bool(value >= threshold))
        if np.isfinite(value):
            history.append(float(value))
            if rolling_window > 0 and len(history) > rolling_window:
                history = history[-rolling_window:]
    return (
        pd.Series(selections, index=test_scores.index, dtype=bool),
        pd.Series(thresholds, index=test_scores.index, dtype=float),
    )

def _causal_threshold_policy_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "fold",
        "selection_rate_source",
        "target_selection_rate",
        "realized_selection_rate",
        "mean_causal_threshold",
        "threshold_std",
        "history_rows",
        "rolling_window",
        "min_history",
        "official_f1",
        "official_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_actual_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "selection_guard",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "fold_count",
        "target_selection_rate_mean",
        "realized_selection_rate_mean",
        "test_f1_mean",
        "test_precision_mean",
        "test_recall_mean",
        "test_pred_long_rate_mean",
        "test_label_lift_vs_base_mean",
        "test_mean_forward_return_mean",
        "test_mean_tb_return_mean",
        "official_f1_mean",
        "f1_delta_vs_official",
        "f1_pass_fold_rate",
        "precision_pass_fold_rate",
        "pred_rate_pass_fold_rate",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "causal_policy_passed_cv",
        "diagnostic_outcome",
        "recommended_action",
        "selection_guard",
    ]
    cfg = _cfg(config, ["validation", "causal_threshold_policy"], {}) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)
    rolling_window = max(0, int(cfg.get("rolling_window", 1080)))
    min_history = max(20, int(cfg.get("min_history", 256)))
    configured_rates = cfg.get("selection_rates", [0.55, 0.60, 0.65, 0.70]) or []
    fixed_rates = sorted(
        {
            float(value)
            for value in configured_rates
            if isinstance(value, (int, float)) and np.isfinite(float(value)) and 0.0 < float(value) < 1.0
        }
    )
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    rows: list[dict[str, Any]] = []

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "split", "label", "prob_long"}.issubset(predictions.columns):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        metric_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold_raw, fold_part in predictions.groupby("fold"):
            fold = int(fold_raw)
            validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
            test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            if validation.empty or test.empty:
                continue
            sort_column = "timestamp" if "timestamp" in test.columns else None
            if sort_column:
                validation = validation.sort_values(sort_column)
                test = test.sort_values(sort_column)
            validation = validation.replace([np.inf, -np.inf], np.nan).dropna(subset=["prob_long"]).copy()
            test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if len(validation) < min_history or test.empty:
                continue
            metrics_row = metric_by_fold.get(fold, {})
            policies = [
                {
                    "policy_name": f"causal_fixed_top_{int(round(rate * 100)):02d}",
                    "target_selection_rate": rate,
                    "selection_rate_source": "configured_fixed_rate",
                }
                for rate in fixed_rates
            ]
            for source_name, threshold_column in [
                ("validation_constrained_rate", "constrained_threshold"),
                ("validation_official_rate", "official_threshold"),
            ]:
                threshold = _float(metrics_row, threshold_column)
                if np.isfinite(threshold):
                    val_rate = float(
                        (
                            pd.to_numeric(validation["prob_long"], errors="coerce")
                            >= threshold
                        ).mean()
                    )
                    if 0.0 < val_rate < 1.0:
                        policies.append(
                            {
                                "policy_name": f"causal_{source_name}",
                                "target_selection_rate": val_rate,
                                "selection_rate_source": source_name,
                            }
                        )
            for policy in policies:
                rate = float(policy["target_selection_rate"])
                selected_mask, thresholds = _causal_score_quantile_mask(
                    validation["prob_long"],
                    test["prob_long"],
                    selection_rate=rate,
                    rolling_window=rolling_window,
                    min_history=min_history,
                )
                class_metrics = _classification_metrics_for_mask(test["label"], selected_mask)
                selection_stats = _selection_stats_for_mask(test, selected_mask)
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "policy_name": policy["policy_name"],
                        "fold": fold,
                        "selection_rate_source": policy["selection_rate_source"],
                        "target_selection_rate": rate,
                        "realized_selection_rate": class_metrics["pred_long_rate"],
                        "mean_causal_threshold": float(thresholds.mean()) if thresholds.notna().any() else np.nan,
                        "threshold_std": float(thresholds.std(ddof=0)) if thresholds.notna().any() else np.nan,
                        "history_rows": int(len(validation)),
                        "rolling_window": rolling_window,
                        "min_history": min_history,
                        "official_f1": _float(metrics_row, "test_f1_at_official_threshold"),
                        "official_pred_long_rate": _float(
                            metrics_row,
                            "test_pred_long_rate_at_official_threshold",
                        ),
                        "test_f1": class_metrics["f1"],
                        "test_precision": class_metrics["precision"],
                        "test_recall": class_metrics["recall"],
                        "test_pred_long_rate": class_metrics["pred_long_rate"],
                        "test_actual_long_rate": selection_stats["actual_long_rate"],
                        "test_label_lift_vs_base": selection_stats["label_lift_vs_base"],
                        "test_mean_forward_return": selection_stats["mean_forward_return"],
                        "test_mean_tb_return": selection_stats["mean_tb_return"],
                        "selection_guard": "causal_past_scores_only_no_test_labels",
                    }
                )
    if not rows:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)
    by_fold = (
        pd.DataFrame(rows, columns=by_fold_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "policy_name", "fold"])
        .reset_index(drop=True)
    )
    summaries: list[dict[str, Any]] = []
    for keys, part in by_fold.groupby(
        ["candidate", "candidate_type", "fold_scope", "policy_name"],
        dropna=False,
    ):
        candidate, candidate_type, fold_scope, policy_name = keys
        f1 = pd.to_numeric(part["test_f1"], errors="coerce")
        precision = pd.to_numeric(part["test_precision"], errors="coerce")
        pred_rate = pd.to_numeric(part["test_pred_long_rate"], errors="coerce")
        lift = pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce")
        fwd = pd.to_numeric(part["test_mean_forward_return"], errors="coerce")
        official_f1 = pd.to_numeric(part["official_f1"], errors="coerce")
        f1_pass_rate = float((f1 >= min_long_f1).mean()) if f1.notna().any() else np.nan
        precision_pass_rate = float((precision >= min_precision).mean()) if precision.notna().any() else np.nan
        pred_pass_rate = float((pred_rate <= max_pred_long_rate).mean()) if pred_rate.notna().any() else np.nan
        policy_passed = bool(
            np.isfinite(f1_pass_rate)
            and f1_pass_rate >= 0.75
            and np.isfinite(precision_pass_rate)
            and precision_pass_rate >= 0.75
            and np.isfinite(pred_pass_rate)
            and pred_pass_rate >= 0.75
        )
        f1_mean = float(f1.mean()) if f1.notna().any() else np.nan
        official_f1_mean = float(official_f1.mean()) if official_f1.notna().any() else np.nan
        f1_delta = f1_mean - official_f1_mean if np.isfinite(f1_mean) and np.isfinite(official_f1_mean) else np.nan
        if policy_passed and np.isfinite(f1_delta) and f1_delta > 0.005:
            outcome = "causal_threshold_transfer_candidate"
            action = "pre_register_for_future_unseen_oos; do_not_promote_from_seen_holdout"
        elif np.isfinite(f1_mean) and f1_mean >= min_long_f1 and float(pred_rate.mean()) > max_pred_long_rate:
            outcome = "causal_policy_too_broad"
            action = "reject_policy; do_not_relax_pred_long_rate_guardrail"
        elif np.isfinite(f1_mean) and f1_mean < min_long_f1:
            outcome = "causal_threshold_transfer_insufficient"
            action = "score_separation_remains_primary_f1_blocker"
        else:
            outcome = "monitor_no_clear_causal_threshold_fix"
            action = "diagnostic_only"
        summaries.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "fold_scope": str(fold_scope),
                "policy_name": str(policy_name),
                "fold_count": int(part["fold"].nunique()),
                "target_selection_rate_mean": float(
                    pd.to_numeric(part["target_selection_rate"], errors="coerce").mean()
                ),
                "realized_selection_rate_mean": float(pred_rate.mean()) if pred_rate.notna().any() else np.nan,
                "test_f1_mean": f1_mean,
                "test_precision_mean": float(precision.mean()) if precision.notna().any() else np.nan,
                "test_recall_mean": float(pd.to_numeric(part["test_recall"], errors="coerce").mean()),
                "test_pred_long_rate_mean": float(pred_rate.mean()) if pred_rate.notna().any() else np.nan,
                "test_label_lift_vs_base_mean": float(lift.mean()) if lift.notna().any() else np.nan,
                "test_mean_forward_return_mean": float(fwd.mean()) if fwd.notna().any() else np.nan,
                "test_mean_tb_return_mean": float(
                    pd.to_numeric(part["test_mean_tb_return"], errors="coerce").mean()
                ),
                "official_f1_mean": official_f1_mean,
                "f1_delta_vs_official": f1_delta,
                "f1_pass_fold_rate": f1_pass_rate,
                "precision_pass_fold_rate": precision_pass_rate,
                "pred_rate_pass_fold_rate": pred_pass_rate,
                "positive_lift_fold_rate": float((lift > 1.0).mean()) if lift.notna().any() else np.nan,
                "positive_forward_return_fold_rate": float((fwd > 0.0).mean()) if fwd.notna().any() else np.nan,
                "causal_policy_passed_cv": policy_passed,
                "diagnostic_outcome": outcome,
                "recommended_action": action,
                "selection_guard": "causal_past_scores_only_no_test_labels",
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "causal_policy_passed_cv", "test_f1_mean"],
            ascending=[True, True, True, False, False],
        )
        .reset_index(drop=True)
    )
    return summary, by_fold

def _causal_threshold_policy_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Causal Threshold Policy Review", ""]
    lines.append(
        "Each test decision uses only validation scores and earlier test scores. Test labels never set "
        "the threshold. These policies remain CV diagnostics until pre-registered future unseen OOS confirms them."
    )
    if summary.empty:
        lines.extend(["", "No causal threshold policy rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "test_f1_mean",
        "test_pred_long_rate_mean",
        "f1_delta_vs_official",
        "f1_pass_fold_rate",
        "positive_forward_return_fold_rate",
        "causal_policy_passed_cv",
        "diagnostic_outcome",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append(f"Fold-level rows: {len(by_fold)}. See `causal_threshold_policy_by_fold.csv`.")
    return "\n".join(lines)

def _write_causal_threshold_policy(path: Path, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "causal_threshold_policy_summary.csv", index=False)
    by_fold.to_csv(path / "causal_threshold_policy_by_fold.csv", index=False)
    (path / "causal_threshold_policy.md").write_text(
        _causal_threshold_policy_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "causal_threshold_policy.json",
        {"summary": summary.to_dict(orient="records"), "by_fold": by_fold.to_dict(orient="records")},
    )

def _classification_skill_frames(
    entries: list[dict[str, Any]],
    causal_threshold_by_fold: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold",
        "count",
        "label_prevalence",
        "average_precision",
        "prauc_lift_vs_prevalence",
        "threshold",
        "f1",
        "precision",
        "recall",
        "pred_long_rate",
        "always_long_f1",
        "rate_matched_random_f1",
        "max_rate_random_f1",
        "f1_skill_vs_always_long",
        "f1_skill_vs_rate_matched_random",
        "precision_lift_vs_prevalence",
        "selected_actual_long_rate",
        "selected_label_lift_vs_base",
        "selected_mean_forward_return",
        "selected_mean_tb_return",
        "f1_target",
        "max_pred_long_rate",
        "f1_target_exceeds_always_long",
        "f1_target_exceeds_max_rate_random",
        "skill_evidence_passed",
        "selection_guard",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold_count",
        "label_prevalence_mean",
        "average_precision_mean",
        "prauc_lift_vs_prevalence_mean",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "pred_long_rate_mean",
        "always_long_f1_mean",
        "rate_matched_random_f1_mean",
        "max_rate_random_f1_mean",
        "f1_skill_vs_always_long_mean",
        "f1_skill_vs_rate_matched_random_mean",
        "precision_lift_vs_prevalence_mean",
        "positive_f1_skill_vs_rate_random_fold_rate",
        "positive_forward_return_fold_rate",
        "selected_mean_forward_return_mean",
        "selected_mean_tb_return_mean",
        "f1_target",
        "max_pred_long_rate",
        "f1_target_exceeds_always_long_baseline",
        "f1_target_exceeds_max_rate_random_baseline",
        "skill_evidence_passed",
        "classification_conclusion",
        "recommended_governance_action",
    ]
    cfg = _cfg(config, ["validation", "classification_skill"], {}) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)
    f1_target = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_prauc_lift = float(cfg.get("min_prauc_lift_vs_prevalence", 1.05))
    min_precision_lift = float(cfg.get("min_precision_lift_vs_prevalence", 1.05))
    min_f1_skill = float(cfg.get("min_f1_skill_vs_rate_random", 0.0))
    min_positive_fwd_rate = float(cfg.get("min_positive_forward_return_fold_rate", 0.60))
    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        candidate: str,
        candidate_type: str,
        fold_scope: str,
        policy_name: str,
        policy_type: str,
        fold: int,
        test: pd.DataFrame,
        scores: pd.Series,
        threshold: float,
        class_metrics: dict[str, float],
        selection_stats: dict[str, float],
        selection_guard: str,
    ) -> None:
        labels = pd.to_numeric(test["label"], errors="coerce")
        valid = labels.notna() & pd.to_numeric(scores, errors="coerce").notna()
        labels = labels.loc[valid].astype(int)
        clean_scores = pd.to_numeric(scores.loc[valid], errors="coerce")
        count = int(len(labels))
        if count == 0:
            return
        prevalence = float(labels.mean())
        average_precision = _safe_average_precision(labels, clean_scores)
        pred_rate = float(class_metrics["pred_long_rate"])
        always_long_f1 = float(2.0 * prevalence / (1.0 + prevalence)) if prevalence > 0 else 0.0
        rate_random_denominator = prevalence + pred_rate
        rate_matched_random_f1 = (
            float(2.0 * prevalence * pred_rate / rate_random_denominator)
            if rate_random_denominator > 0
            else 0.0
        )
        max_rate_denominator = prevalence + max_pred_long_rate
        max_rate_random_f1 = (
            float(2.0 * prevalence * max_pred_long_rate / max_rate_denominator)
            if max_rate_denominator > 0
            else 0.0
        )
        precision_lift = (
            float(class_metrics["precision"] / prevalence)
            if prevalence > 0
            else np.nan
        )
        prauc_lift = float(average_precision / prevalence) if prevalence > 0 and np.isfinite(average_precision) else np.nan
        f1_skill_rate_random = float(class_metrics["f1"] - rate_matched_random_f1)
        skill_passed = bool(
            np.isfinite(prauc_lift)
            and prauc_lift >= min_prauc_lift
            and np.isfinite(precision_lift)
            and precision_lift >= min_precision_lift
            and f1_skill_rate_random > min_f1_skill
            and pred_rate <= max_pred_long_rate
            and np.isfinite(selection_stats["mean_forward_return"])
            and selection_stats["mean_forward_return"] > 0.0
        )
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "fold": int(fold),
                "count": count,
                "label_prevalence": prevalence,
                "average_precision": average_precision,
                "prauc_lift_vs_prevalence": prauc_lift,
                "threshold": threshold,
                "f1": float(class_metrics["f1"]),
                "precision": float(class_metrics["precision"]),
                "recall": float(class_metrics["recall"]),
                "pred_long_rate": pred_rate,
                "always_long_f1": always_long_f1,
                "rate_matched_random_f1": rate_matched_random_f1,
                "max_rate_random_f1": max_rate_random_f1,
                "f1_skill_vs_always_long": float(class_metrics["f1"] - always_long_f1),
                "f1_skill_vs_rate_matched_random": f1_skill_rate_random,
                "precision_lift_vs_prevalence": precision_lift,
                "selected_actual_long_rate": selection_stats["actual_long_rate"],
                "selected_label_lift_vs_base": selection_stats["label_lift_vs_base"],
                "selected_mean_forward_return": selection_stats["mean_forward_return"],
                "selected_mean_tb_return": selection_stats["mean_tb_return"],
                "f1_target": f1_target,
                "max_pred_long_rate": max_pred_long_rate,
                "f1_target_exceeds_always_long": bool(f1_target > always_long_f1),
                "f1_target_exceeds_max_rate_random": bool(f1_target > max_rate_random_f1),
                "skill_evidence_passed": skill_passed,
                "selection_guard": selection_guard,
            }
        )

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "split", "label", "prob_long"}.issubset(predictions.columns):
            continue
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        threshold_metrics = _entry_threshold_policy_frame(entry)
        official_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        calibrated = (entry.get("diagnostics", {}) or {}).get("calibrated_predictions")
        for fold_raw, part in predictions.groupby("fold"):
            fold = int(fold_raw)
            test = part.loc[part["split"].astype(str) == "test"].copy()
            test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"])
            if test.empty:
                continue
            metric_row = official_by_fold.get(fold, {})
            uses_calibration = bool(metric_row.get("official_threshold_uses_calibration", False))
            score_column = "prob_long"
            if uses_calibration and isinstance(calibrated, pd.DataFrame) and not calibrated.empty:
                calibrated_fold = calibrated.loc[
                    pd.to_numeric(calibrated["fold"], errors="coerce").eq(fold)
                ].copy()
                if not calibrated_fold.empty and "prob_long_calibrated" in calibrated_fold.columns:
                    test = calibrated_fold.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=["label", "prob_long_calibrated"]
                    )
                    score_column = "prob_long_calibrated"
            threshold = _float(metric_row, "official_threshold")
            if np.isfinite(threshold):
                selected_mask = pd.to_numeric(test[score_column], errors="coerce") >= threshold
                class_metrics = _classification_metrics_for_mask(test["label"], selected_mask)
                selection_stats = _selection_stats_for_mask(test, selected_mask)
            else:
                class_metrics = {
                    "f1": _float(metric_row, "test_f1_at_official_threshold"),
                    "precision": _float(metric_row, "test_precision_at_official_threshold"),
                    "recall": _float(metric_row, "test_recall_at_official_threshold"),
                    "pred_long_rate": _float(metric_row, "test_pred_long_rate_at_official_threshold"),
                }
                if not all(np.isfinite(value) for value in class_metrics.values()):
                    continue
                selection_stats = {
                    "actual_long_rate": np.nan,
                    "label_lift_vs_base": np.nan,
                    "mean_forward_return": np.nan,
                    "mean_tb_return": np.nan,
                }
            append_row(
                candidate=candidate,
                candidate_type=candidate_type,
                fold_scope=fold_scope,
                policy_name="official_threshold",
                policy_type="official",
                fold=fold,
                test=test,
                scores=test[score_column],
                threshold=threshold,
                class_metrics=class_metrics,
                selection_stats=selection_stats,
                selection_guard=str(
                    metric_row.get("official_threshold_source")
                    or _entry_official_threshold_source(entry)
                    or "validation_official_threshold"
                ),
            )

    if not causal_threshold_by_fold.empty:
        entry_lookup = {
            (str(entry.get("profile", "")), str(entry.get("fold_scope", ""))): entry
            for entry in entries
        }
        for _, policy_row in causal_threshold_by_fold.iterrows():
            candidate = str(policy_row.get("candidate", ""))
            fold_scope = str(policy_row.get("fold_scope", ""))
            entry = entry_lookup.get((candidate, fold_scope))
            if entry is None:
                continue
            predictions = entry.get("predictions")
            fold = int(policy_row["fold"])
            test = predictions.loc[
                (pd.to_numeric(predictions["fold"], errors="coerce").eq(fold))
                & predictions["split"].astype(str).eq("test")
            ].copy()
            test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"])
            if test.empty:
                continue
            class_metrics = {
                "f1": _float(policy_row.to_dict(), "test_f1"),
                "precision": _float(policy_row.to_dict(), "test_precision"),
                "recall": _float(policy_row.to_dict(), "test_recall"),
                "pred_long_rate": _float(policy_row.to_dict(), "test_pred_long_rate"),
            }
            selection_stats = {
                "actual_long_rate": _float(policy_row.to_dict(), "test_actual_long_rate"),
                "label_lift_vs_base": _float(policy_row.to_dict(), "test_label_lift_vs_base"),
                "mean_forward_return": _float(policy_row.to_dict(), "test_mean_forward_return"),
                "mean_tb_return": _float(policy_row.to_dict(), "test_mean_tb_return"),
            }
            append_row(
                candidate=candidate,
                candidate_type=str(policy_row.get("candidate_type", _diagnostic_candidate_type(fold_scope))),
                fold_scope=fold_scope,
                policy_name=str(policy_row.get("policy_name", "")),
                policy_type="causal_threshold",
                fold=fold,
                test=test,
                scores=test["prob_long"],
                threshold=_float(policy_row.to_dict(), "mean_causal_threshold"),
                class_metrics=class_metrics,
                selection_stats=selection_stats,
                selection_guard=str(policy_row.get("selection_guard", "causal_past_scores_only_no_test_labels")),
            )

    if not rows:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)
    by_fold = (
        pd.DataFrame(rows, columns=by_fold_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "policy_type", "policy_name", "fold"])
        .reset_index(drop=True)
    )
    summaries: list[dict[str, Any]] = []
    for keys, part in by_fold.groupby(
        ["candidate", "candidate_type", "fold_scope", "policy_name", "policy_type"],
        dropna=False,
    ):
        candidate, candidate_type, fold_scope, policy_name, policy_type = keys
        numeric = {
            column: pd.to_numeric(part[column], errors="coerce")
            for column in [
                "label_prevalence",
                "average_precision",
                "prauc_lift_vs_prevalence",
                "f1",
                "precision",
                "recall",
                "pred_long_rate",
                "always_long_f1",
                "rate_matched_random_f1",
                "max_rate_random_f1",
                "f1_skill_vs_always_long",
                "f1_skill_vs_rate_matched_random",
                "precision_lift_vs_prevalence",
                "selected_mean_forward_return",
                "selected_mean_tb_return",
            ]
        }
        target_above_always = bool(f1_target > float(numeric["always_long_f1"].mean()))
        target_above_max_rate_random = bool(f1_target > float(numeric["max_rate_random_f1"].mean()))
        positive_fwd_rate = float((numeric["selected_mean_forward_return"] > 0.0).mean())
        skill_pass_rate = float(part["skill_evidence_passed"].astype(bool).mean())
        aggregate_skill_passed = bool(
            numeric["prauc_lift_vs_prevalence"].mean() >= min_prauc_lift
            and numeric["precision_lift_vs_prevalence"].mean() >= min_precision_lift
            and numeric["f1_skill_vs_rate_matched_random"].mean() > min_f1_skill
            and positive_fwd_rate >= min_positive_fwd_rate
            and skill_pass_rate >= 0.50
        )
        if not target_above_always:
            conclusion = "standalone_f1_target_below_always_long_no_skill_baseline"
            action = "retain_official_gate; require_pred_rate_guardrail_and_skill_normalized_companion_metrics"
        elif not target_above_max_rate_random:
            conclusion = "f1_target_below_guardrail_matched_random_baseline"
            action = "formal_charter_review_required_before_using_f1_as_readiness_evidence"
        elif aggregate_skill_passed:
            conclusion = "classification_skill_present_under_guardrails"
            action = "require_future_unseen_oos_confirmation; do_not_promote_from_seen_holdout"
        else:
            conclusion = "classification_skill_weak_or_inconsistent_under_guardrails"
            action = "improve_score_separation_or_payoff_alignment; do_not_optimize_raw_f1_alone"
        summaries.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "fold_scope": str(fold_scope),
                "policy_name": str(policy_name),
                "policy_type": str(policy_type),
                "fold_count": int(part["fold"].nunique()),
                "label_prevalence_mean": float(numeric["label_prevalence"].mean()),
                "average_precision_mean": float(numeric["average_precision"].mean()),
                "prauc_lift_vs_prevalence_mean": float(numeric["prauc_lift_vs_prevalence"].mean()),
                "f1_mean": float(numeric["f1"].mean()),
                "precision_mean": float(numeric["precision"].mean()),
                "recall_mean": float(numeric["recall"].mean()),
                "pred_long_rate_mean": float(numeric["pred_long_rate"].mean()),
                "always_long_f1_mean": float(numeric["always_long_f1"].mean()),
                "rate_matched_random_f1_mean": float(numeric["rate_matched_random_f1"].mean()),
                "max_rate_random_f1_mean": float(numeric["max_rate_random_f1"].mean()),
                "f1_skill_vs_always_long_mean": float(numeric["f1_skill_vs_always_long"].mean()),
                "f1_skill_vs_rate_matched_random_mean": float(
                    numeric["f1_skill_vs_rate_matched_random"].mean()
                ),
                "precision_lift_vs_prevalence_mean": float(
                    numeric["precision_lift_vs_prevalence"].mean()
                ),
                "positive_f1_skill_vs_rate_random_fold_rate": float(
                    (numeric["f1_skill_vs_rate_matched_random"] > 0.0).mean()
                ),
                "positive_forward_return_fold_rate": positive_fwd_rate,
                "selected_mean_forward_return_mean": float(
                    numeric["selected_mean_forward_return"].mean()
                ),
                "selected_mean_tb_return_mean": float(numeric["selected_mean_tb_return"].mean()),
                "f1_target": f1_target,
                "max_pred_long_rate": max_pred_long_rate,
                "f1_target_exceeds_always_long_baseline": target_above_always,
                "f1_target_exceeds_max_rate_random_baseline": target_above_max_rate_random,
                "skill_evidence_passed": aggregate_skill_passed,
                "classification_conclusion": conclusion,
                "recommended_governance_action": action,
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "policy_type", "policy_name"])
        .reset_index(drop=True)
    )
    return summary, by_fold

def _classification_skill_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Classification Skill Audit", ""]
    lines.append(
        "Raw F1 is compared with two no-skill references: always predicting long and randomly selecting "
        "the same fraction of rows. PRAUC is normalized by label prevalence. These are companion diagnostics; "
        "the official Phase 1 F1 and prediction-rate gates remain unchanged."
    )
    if summary.empty:
        lines.extend(["", "No classification skill rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "f1_mean",
        "always_long_f1_mean",
        "rate_matched_random_f1_mean",
        "f1_skill_vs_rate_matched_random_mean",
        "prauc_lift_vs_prevalence_mean",
        "precision_lift_vs_prevalence_mean",
        "positive_forward_return_fold_rate",
        "skill_evidence_passed",
        "classification_conclusion",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append(f"Fold-level rows: {len(by_fold)}. See `classification_skill_by_fold.csv`.")
    return "\n".join(lines)

def _write_classification_skill(
    path: Path,
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "classification_skill_summary.csv", index=False)
    by_fold.to_csv(path / "classification_skill_by_fold.csv", index=False)
    (path / "classification_skill.md").write_text(
        _classification_skill_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "classification_skill.json",
        {"summary": summary.to_dict(orient="records"), "by_fold": by_fold.to_dict(orient="records")},
    )

def _validation_charter_review_frame(
    *,
    control_profile: str,
    rank_ic_evidence: pd.DataFrame,
    classification_skill_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "criterion",
        "control_profile",
        "active_charter",
        "criterion_role",
        "review_status",
        "official_target",
        "observed_value",
        "official_gate_passed",
        "reference_baseline",
        "reference_value",
        "statistical_validity",
        "charter_review_recommended",
        "proposed_companion_evidence",
        "automatic_gate_change_allowed",
        "governance_decision",
    ]
    charter = _cfg(config, ["validation", "charter"], {}) or {}
    active_charter = str(charter.get("active_version", "v3_legacy"))
    evidence_charter_active = active_charter != "v3_legacy"
    rows: list[dict[str, Any]] = []
    rank = pd.DataFrame()
    if not rank_ic_evidence.empty:
        rank = rank_ic_evidence.loc[
            (rank_ic_evidence["candidate"].astype(str) == str(control_profile))
            & rank_ic_evidence["fold_scope"].astype(str).eq("full")
        ]
    if not rank.empty:
        row = rank.iloc[0]
        observed_std = _float(row.to_dict(), "observed_std_rank_ic")
        target_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
        target_below_noise = bool(row.get("target_below_noise_floor_all_blocks", False))
        random_positive = bool(row.get("random_effects_positive_all_blocks", False))
        rows.append(
            {
                "criterion": "rank_ic_std",
                "control_profile": control_profile,
                "active_charter": active_charter,
                "criterion_role": "monitor" if evidence_charter_active else "gate",
                "review_status": (
                    "resolved_by_active_evidence_charter"
                    if evidence_charter_active
                    else "formal_review_pending"
                ),
                "official_target": (
                    f"legacy monitor < {target_std}"
                    if evidence_charter_active
                    else f"< {target_std}"
                ),
                "observed_value": observed_std,
                "official_gate_passed": bool(np.isfinite(observed_std) and observed_std < target_std),
                "reference_baseline": "multi_block_bootstrap_noise_floor_max",
                "reference_value": _float(row.to_dict(), "max_noise_floor_std"),
                "statistical_validity": (
                    "absolute_target_below_measured_noise_floor_across_all_blocks"
                    if target_below_noise
                    else "absolute_target_above_at_least_one_measured_noise_floor"
                ),
                "charter_review_recommended": (
                    False
                    if evidence_charter_active
                    else target_below_noise and random_positive
                ),
                "proposed_companion_evidence": (
                    "random_effects_ci; positive_fold_sign_test; between_fold_tau; heterogeneity_i2; future_unseen_oos"
                ),
                "automatic_gate_change_allowed": False,
                "governance_decision": (
                    "resolved_active_charter_uses_random_effects_and_sign_test;legacy_std_monitor_only"
                    if evidence_charter_active
                    else "formal_review_recommended_keep_official_gate_unchanged"
                    if target_below_noise and random_positive
                    else "retain_current_gate_and_monitor"
                ),
            }
        )

    classification = pd.DataFrame()
    if not classification_skill_summary.empty:
        classification = classification_skill_summary.loc[
            (classification_skill_summary["candidate"].astype(str) == str(control_profile))
            & classification_skill_summary["fold_scope"].astype(str).eq("full")
            & classification_skill_summary["policy_name"].astype(str).eq("official_threshold")
        ]
    if not classification.empty:
        row = classification.iloc[0]
        observed_f1 = _float(row.to_dict(), "f1_mean")
        target_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
        pred_rate = _float(row.to_dict(), "pred_long_rate_mean")
        max_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
        target_above_always = bool(row.get("f1_target_exceeds_always_long_baseline", False))
        target_above_max_rate_random = bool(row.get("f1_target_exceeds_max_rate_random_baseline", False))
        rows.append(
            {
                "criterion": "long_f1",
                "control_profile": control_profile,
                "active_charter": active_charter,
                "criterion_role": "monitor" if evidence_charter_active else "gate",
                "review_status": (
                    "resolved_by_active_evidence_charter"
                    if evidence_charter_active
                    else "formal_review_pending"
                ),
                "official_target": (
                    f"legacy monitor > {target_f1}; active pred_long_rate <= {max_rate}"
                    if evidence_charter_active
                    else f"> {target_f1}; pred_long_rate <= {max_rate}"
                ),
                "observed_value": observed_f1,
                "official_gate_passed": bool(
                    np.isfinite(observed_f1)
                    and observed_f1 > target_f1
                    and np.isfinite(pred_rate)
                    and pred_rate <= max_rate
                ),
                "reference_baseline": "always_long_f1_mean",
                "reference_value": _float(row.to_dict(), "always_long_f1_mean"),
                "statistical_validity": (
                    "standalone_f1_target_not_above_always_long_baseline_but_rate_guardrail_adds_constraint"
                    if not target_above_always
                    else (
                        "target_not_above_max_rate_random_baseline"
                        if not target_above_max_rate_random
                        else "target_above_reported_no_skill_baselines"
                    )
                ),
                "charter_review_recommended": (
                    False
                    if evidence_charter_active
                    else not target_above_always or not target_above_max_rate_random
                ),
                "proposed_companion_evidence": (
                    "f1_skill_vs_rate_matched_random; prauc_lift_vs_prevalence; "
                    "precision_lift_vs_prevalence; positive_forward_return_fold_rate"
                ),
                "automatic_gate_change_allowed": False,
                "governance_decision": (
                    "resolved_active_charter_uses_skill_normalized_metrics;legacy_raw_f1_monitor_only"
                    if evidence_charter_active
                    else "formal_review_recommended_keep_official_gate_unchanged"
                    if not target_above_always or not target_above_max_rate_random
                    else "retain_current_gate_with_skill_companions"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _validation_charter_review_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Validation Charter Review", ""]
    lines.append(
        "This report audits whether legacy Phase 1 thresholds remain statistically discriminative and records "
        "how the explicitly active charter treats them. It never changes a gate automatically."
    )
    if frame.empty:
        lines.extend(["", "No validation charter rows were produced."])
        return "\n".join(lines)
    lines.append("")
    lines.append("| " + " | ".join(frame.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in frame.columns) + " |")
    return "\n".join(lines)

def _write_validation_charter_review(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "validation_charter_review.csv", index=False)
    (path / "validation_charter_review.md").write_text(
        _validation_charter_review_markdown(frame),
        encoding="utf-8",
    )
    _write_json(
        path / "validation_charter_review.json",
        {
            "formal_revision_recommended": bool(
                not frame.empty and frame["charter_review_recommended"].astype(bool).any()
            ),
            "review_status": (
                str(frame["review_status"].iloc[0])
                if not frame.empty and "review_status" in frame.columns
                else "not_produced"
            ),
            "active_charter": (
                str(frame["active_charter"].iloc[0])
                if not frame.empty and "active_charter" in frame.columns
                else "unknown"
            ),
            "automatic_gate_change_allowed": False,
            "rows": frame.to_dict(orient="records"),
        },
    )

def _validation_charter_proposal_frame(
    *,
    control_profile: str,
    rank_ic_evidence: pd.DataFrame,
    classification_skill_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "proposal_version",
        "proposal_status",
        "active_for_phase1_readiness",
        "criterion",
        "criterion_role",
        "comparison",
        "proposed_target",
        "observed_value",
        "evidence_passed",
        "evidence_source",
        "rationale",
        "official_gate_unchanged",
    ]
    proposal = _cfg(config, ["validation", "charter_proposal"], {}) or {}
    if not bool(proposal.get("enabled", True)):
        return pd.DataFrame(columns=columns)
    version = str(proposal.get("version", "v4_draft"))
    status = str(proposal.get("status", "proposed_not_active"))
    charter = _cfg(config, ["validation", "charter"], {}) or {}
    active_version = str(charter.get("active_version", "v3_legacy"))
    charter_versions = charter.get("versions", {}) or {}
    active_definition = charter_versions.get(active_version, {}) or {}
    active_for_phase1 = (
        version == active_version
        and status == "active"
        and str(active_definition.get("status", "")) == "active"
    )
    official_gate_unchanged = active_version == "v3_legacy"
    rows: list[dict[str, Any]] = []

    rank = pd.DataFrame()
    if not rank_ic_evidence.empty:
        rank = rank_ic_evidence.loc[
            (rank_ic_evidence["candidate"].astype(str) == str(control_profile))
            & rank_ic_evidence["fold_scope"].astype(str).eq("full")
        ]
    if not rank.empty:
        item = rank.iloc[0].to_dict()
        rank_rules = [
            (
                "mean_rank_ic",
                "gate",
                ">=",
                float(proposal.get("min_mean_rank_ic", _cfg(config, ["validation", "target_rank_ic"], 0.03))),
                _float(item, "observed_mean_rank_ic"),
                "Aggregate out-of-sample monotonic signal.",
            ),
            (
                "positive_fold_fraction",
                "gate",
                ">=",
                float(
                    proposal.get(
                        "min_positive_fold_fraction",
                        _cfg(config, ["validation", "min_positive_ic_fraction"], 0.75),
                    )
                ),
                _float(item, "positive_fold_fraction"),
                "Signal should remain positive across market windows.",
            ),
            (
                "positive_fold_sign_test_pvalue",
                "gate",
                "<=",
                float(proposal.get("max_positive_fold_sign_test_pvalue", 0.01)),
                _float(item, "positive_fold_sign_test_pvalue"),
                "Fold positivity should be unlikely under a no-skill sign process.",
            ),
            (
                "random_effects_positive_all_blocks",
                "gate",
                "==",
                1.0,
                float(bool(item.get("random_effects_positive_all_blocks", False))),
                "Random-effects lower confidence bounds must stay positive across block assumptions.",
            ),
            (
                "rank_ic_std",
                "monitor",
                "monitor_only",
                np.nan,
                _float(item, "observed_std_rank_ic"),
                "Absolute fold std remains visible, but the active evidence charter does not use an infeasible below-noise target as a blocker.",
            ),
        ]
        for criterion, role, comparison, target, observed, rationale in rank_rules:
            if comparison == ">=":
                passed = bool(np.isfinite(observed) and observed >= target)
            elif comparison == "<=":
                passed = bool(np.isfinite(observed) and observed <= target)
            elif comparison == "==":
                passed = bool(np.isfinite(observed) and observed == target)
            else:
                passed = np.nan
            rows.append(
                {
                    "proposal_version": version,
                    "proposal_status": status,
                    "active_for_phase1_readiness": active_for_phase1,
                    "criterion": criterion,
                    "criterion_role": role,
                    "comparison": comparison,
                    "proposed_target": target,
                    "observed_value": observed,
                    "evidence_passed": passed,
                    "evidence_source": "rank_ic_aggregate_evidence.csv",
                    "rationale": rationale,
                    "official_gate_unchanged": official_gate_unchanged,
                }
            )

    classification = pd.DataFrame()
    if not classification_skill_summary.empty:
        classification = classification_skill_summary.loc[
            (classification_skill_summary["candidate"].astype(str) == str(control_profile))
            & classification_skill_summary["fold_scope"].astype(str).eq("full")
            & classification_skill_summary["policy_name"].astype(str).eq("official_threshold")
        ]
    if not classification.empty:
        item = classification.iloc[0].to_dict()
        skill_cfg = _cfg(config, ["validation", "classification_skill"], {}) or {}
        class_rules = [
            (
                "prauc_lift_vs_prevalence",
                "gate",
                ">=",
                float(
                    proposal.get(
                        "min_prauc_lift_vs_prevalence",
                        skill_cfg.get("min_prauc_lift_vs_prevalence", 1.05),
                    )
                ),
                _float(item, "prauc_lift_vs_prevalence_mean"),
                "PRAUC must exceed the class-prevalence no-skill baseline.",
            ),
            (
                "precision_lift_vs_prevalence",
                "gate",
                ">=",
                float(
                    proposal.get(
                        "min_precision_lift_vs_prevalence",
                        skill_cfg.get("min_precision_lift_vs_prevalence", 1.05),
                    )
                ),
                _float(item, "precision_lift_vs_prevalence_mean"),
                "Selected longs must be more precise than unconditional prevalence.",
            ),
            (
                "f1_skill_vs_rate_matched_random",
                "gate",
                ">",
                float(
                    proposal.get(
                        "min_f1_skill_vs_rate_random",
                        skill_cfg.get("min_f1_skill_vs_rate_random", 0.0),
                    )
                ),
                _float(item, "f1_skill_vs_rate_matched_random_mean"),
                "F1 must add value over random selection at the same prediction rate.",
            ),
            (
                "positive_f1_skill_fold_fraction",
                "gate",
                ">=",
                float(proposal.get("min_positive_f1_skill_fold_fraction", 0.75)),
                _float(item, "positive_f1_skill_vs_rate_random_fold_rate"),
                "Rate-normalized F1 skill should persist across folds.",
            ),
            (
                "positive_forward_return_fold_fraction",
                "gate",
                ">=",
                float(
                    proposal.get(
                        "min_positive_forward_return_fold_fraction",
                        skill_cfg.get("min_positive_forward_return_fold_rate", 0.60),
                    )
                ),
                _float(item, "positive_forward_return_fold_rate"),
                "Selected rows should have positive realized forward return in most folds.",
            ),
            (
                "prediction_long_rate",
                "gate",
                "<=",
                float(
                    proposal.get(
                        "max_pred_long_rate",
                        _cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70),
                    )
                ),
                _float(item, "pred_long_rate_mean"),
                "Classification skill cannot come from predicting nearly every row as long.",
            ),
            (
                "raw_long_f1",
                "monitor",
                "monitor_only",
                float(_cfg(config, ["validation", "min_long_f1"], 0.45)),
                _float(item, "f1_mean"),
                "Raw F1 stays reported, while the active evidence charter evaluates it beside rate-matched no-skill baselines.",
            ),
        ]
        for criterion, role, comparison, target, observed, rationale in class_rules:
            if comparison == ">=":
                passed = bool(np.isfinite(observed) and observed >= target)
            elif comparison == ">":
                passed = bool(np.isfinite(observed) and observed > target)
            elif comparison == "<=":
                passed = bool(np.isfinite(observed) and observed <= target)
            else:
                passed = np.nan
            rows.append(
                {
                    "proposal_version": version,
                    "proposal_status": status,
                    "active_for_phase1_readiness": active_for_phase1,
                    "criterion": criterion,
                    "criterion_role": role,
                    "comparison": comparison,
                    "proposed_target": target,
                    "observed_value": observed,
                    "evidence_passed": passed,
                    "evidence_source": "classification_skill_summary.csv",
                    "rationale": rationale,
                    "official_gate_unchanged": official_gate_unchanged,
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _validation_charter_proposal_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Validation Charter Proposal", ""]
    if frame.empty:
        lines.extend(["", "No proposal rows were produced."])
        return "\n".join(lines)
    active = bool(frame["active_for_phase1_readiness"].map(bool).all())
    if active:
        lines.append(
            "This evidence charter is active through an explicit reviewed config and documentation commit. "
            "It does not authorize Phase 2 by itself; future unseen OOS requirements remain mandatory."
        )
    else:
        lines.append(
            "This is an inactive governance draft. It organizes statistically interpretable companion evidence "
            "but does not alter the official Phase 1 gates or authorize Phase 2."
        )
    lines.extend(["", "| " + " | ".join(frame.columns) + " |"])
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in frame.columns) + " |")
    gate_rows = frame.loc[frame["criterion_role"].astype(str).eq("gate")]
    passed = bool(not gate_rows.empty and gate_rows["evidence_passed"].astype(bool).all())
    lines.extend(
        [
            "",
            f"- Evidence gates passed: `{passed}`",
            f"- Active for Phase 1 readiness: `{active}`",
            f"- Legacy official gates unchanged: `{bool(frame['official_gate_unchanged'].map(bool).all())}`",
        ]
    )
    return "\n".join(lines)

def _write_validation_charter_proposal(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "validation_charter_proposal.csv", index=False)
    (path / "validation_charter_proposal.md").write_text(
        _validation_charter_proposal_markdown(frame),
        encoding="utf-8",
    )
    gate_rows = (
        frame.loc[frame["criterion_role"].astype(str).eq("gate")]
        if not frame.empty and "criterion_role" in frame.columns
        else pd.DataFrame()
    )
    active = bool(
        not frame.empty
        and "active_for_phase1_readiness" in frame.columns
        and frame["active_for_phase1_readiness"].map(bool).all()
    )
    official_gate_unchanged = bool(
        not frame.empty
        and "official_gate_unchanged" in frame.columns
        and frame["official_gate_unchanged"].map(bool).all()
    )
    _write_json(
        path / "validation_charter_proposal.json",
        {
            "proposal_status": (
                str(frame["proposal_status"].iloc[0])
                if not frame.empty and "proposal_status" in frame.columns
                else "not_produced"
            ),
            "active_for_phase1_readiness": active,
            "official_gate_unchanged": official_gate_unchanged,
            "evidence_gates_passed": bool(
                not gate_rows.empty and gate_rows["evidence_passed"].astype(bool).all()
            ),
            "draft_evidence_gates_passed": bool(
                not active
                and not gate_rows.empty
                and gate_rows["evidence_passed"].astype(bool).all()
            ),
            "rows": frame.to_dict(orient="records"),
        },
    )
