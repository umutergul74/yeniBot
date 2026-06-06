"""Threshold transfer, regime policy, and threshold diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
    _numeric_mean,
    _rank_ic_for_frame,
    _write_json,
)

from yenibot.experiment.folds import (
    _entry_threshold_policy_frame,
)

from yenibot.experiment.training import (
    _threshold_summary_metric,
)

__all__ = [
    '_threshold_forensics_frame',
    '_threshold_policy_review_frame',
    '_threshold_policy_review_markdown',
    '_write_threshold_policy_review',
    '_threshold_metrics_at_value',
    '_threshold_selection_stats',
    '_regime_columns',
    '_with_dominant_regime',
    '_select_validation_threshold',
    '_metrics_for_masked_predictions',
    '_regime_threshold_policy_frames',
    '_regime_threshold_policy_markdown',
    '_write_regime_threshold_policy',
    '_regime_stability_frames',
    '_regime_stability_markdown',
    '_write_regime_stability',
    '_ewma_last',
    '_threshold_transfer_review_frames',
    '_threshold_transfer_review_markdown',
    '_write_threshold_transfer_review',
]

def _threshold_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "selected_threshold",
        "test_f1_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "constrained_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "calibrated_constrained_threshold",
        "test_f1_at_calibrated_constrained_threshold",
        "test_pred_long_rate_at_calibrated_constrained_threshold",
        "official_threshold_source",
        "official_threshold_uses_calibration",
        "official_threshold",
        "test_f1_at_official_threshold",
        "test_precision_at_official_threshold",
        "test_recall_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "test_oracle_best_f1",
        "selected_f1_gap_vs_target",
        "constrained_f1_gap_vs_target",
        "official_f1_gap_vs_target",
        "selected_pred_rate_excess_vs_guardrail",
        "constrained_pred_rate_excess_vs_guardrail",
        "official_pred_rate_excess_vs_guardrail",
        "primary_issue",
        "recommended_action",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is None or threshold_metrics.empty:
            continue
        candidate = str(entry.get("profile", ""))
        for _, item in threshold_metrics.iterrows():
            row = item.to_dict()
            selected_f1 = _float(row, "test_f1_at_selected_threshold")
            constrained_f1 = _float(row, "test_f1_at_constrained_threshold")
            selected_rate = _float(row, "test_pred_long_rate_at_selected_threshold")
            constrained_rate = _float(row, "test_pred_long_rate_at_constrained_threshold")
            official_f1 = _float(row, "test_f1_at_official_threshold", constrained_f1)
            official_rate = _float(row, "test_pred_long_rate_at_official_threshold", constrained_rate)
            selected_gap = min_long_f1 - selected_f1
            constrained_gap = min_long_f1 - constrained_f1
            official_gap = min_long_f1 - official_f1
            selected_rate_excess = selected_rate - max_pred_long_rate
            constrained_rate_excess = constrained_rate - max_pred_long_rate
            official_rate_excess = official_rate - max_pred_long_rate
            official_source = str(row.get("official_threshold_source", ""))
            if np.isfinite(official_rate_excess) and official_rate_excess > 0:
                issue = "official_pred_long_rate"
                action = "raise_cv_threshold_or_reduce_score_compression"
            elif np.isfinite(official_gap) and official_gap > 0:
                issue = "official_f1"
                action = "improve_score_ranking_or_calibration_before_phase2"
            elif np.isfinite(selected_rate_excess) and selected_rate_excess > 0:
                issue = "selected_threshold_too_broad"
                action = "prefer_constrained_threshold_for_review"
            elif np.isfinite(selected_gap) and selected_gap > 0:
                issue = "selected_f1"
                action = "improve_validation_threshold_transfer"
            elif np.isfinite(constrained_rate_excess) and constrained_rate_excess > 0:
                issue = "constrained_pred_long_rate"
                action = "raise_cv_threshold_or_reduce_score_compression"
            elif np.isfinite(constrained_gap) and constrained_gap > 0:
                issue = "constrained_f1"
                action = "improve_score_ranking_or_calibration_before_phase2"
            else:
                issue = "ok"
                action = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": _diagnostic_candidate_type(fold_scope),
                    "fold_scope": fold_scope,
                    "fold": int(row.get("fold")),
                    "selected_threshold": _float(row, "selected_threshold"),
                    "test_f1_at_selected_threshold": selected_f1,
                    "test_pred_long_rate_at_selected_threshold": selected_rate,
                    "constrained_threshold": _float(row, "constrained_threshold"),
                    "test_f1_at_constrained_threshold": constrained_f1,
                    "test_pred_long_rate_at_constrained_threshold": constrained_rate,
                    "calibrated_constrained_threshold": _float(row, "calibrated_constrained_threshold"),
                    "test_f1_at_calibrated_constrained_threshold": _float(
                        row,
                        "calibrated_test_f1_at_constrained_threshold",
                    ),
                    "test_pred_long_rate_at_calibrated_constrained_threshold": _float(
                        row,
                        "calibrated_test_pred_long_rate_at_constrained_threshold",
                    ),
                    "official_threshold_source": official_source,
                    "official_threshold_uses_calibration": bool(row.get("official_threshold_uses_calibration", False)),
                    "official_threshold": _float(row, "official_threshold"),
                    "test_f1_at_official_threshold": official_f1,
                    "test_precision_at_official_threshold": _float(row, "test_precision_at_official_threshold"),
                    "test_recall_at_official_threshold": _float(row, "test_recall_at_official_threshold"),
                    "test_pred_long_rate_at_official_threshold": official_rate,
                    "test_oracle_best_f1": _float(row, "test_oracle_best_f1"),
                    "selected_f1_gap_vs_target": selected_gap,
                    "constrained_f1_gap_vs_target": constrained_gap,
                    "official_f1_gap_vs_target": official_gap,
                    "selected_pred_rate_excess_vs_guardrail": selected_rate_excess,
                    "constrained_pred_rate_excess_vs_guardrail": constrained_rate_excess,
                    "official_pred_rate_excess_vs_guardrail": official_rate_excess,
                    "primary_issue": issue,
                    "recommended_action": action,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "constrained_f1_gap_vs_target"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _threshold_policy_review_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "threshold_source",
        "threshold_cap",
        "threshold_mean",
        "source_selection_metric",
        "source_f1",
        "source_precision",
        "source_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "mean_lift_vs_base",
        "mean_forward_return",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "constraints_satisfied_fold_rate",
        "f1_passed",
        "precision_passed",
        "pred_long_rate_passed",
        "policy_passed_cv_test",
        "selection_guard",
        "recommended_action",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    rows: list[dict[str, Any]] = []

    def summary_metric(summary: pd.DataFrame | None, metric: str) -> float:
        return _threshold_summary_metric(summary, metric)

    def add_row(
        *,
        entry: dict[str, Any],
        policy_name: str,
        policy_type: str,
        threshold_source: str,
        threshold_cap: float = np.nan,
        threshold_mean: float = np.nan,
        source_selection_metric: str = "",
        source_f1: float = np.nan,
        source_precision: float = np.nan,
        source_pred_long_rate: float = np.nan,
        test_f1: float = np.nan,
        test_precision: float = np.nan,
        test_recall: float = np.nan,
        test_pred_long_rate: float = np.nan,
        mean_lift_vs_base: float = np.nan,
        mean_forward_return: float = np.nan,
        positive_lift_fold_rate: float = np.nan,
        positive_forward_return_fold_rate: float = np.nan,
        constraints_satisfied_fold_rate: float = np.nan,
    ) -> None:
        f1_passed = bool(np.isfinite(test_f1) and test_f1 > min_long_f1)
        precision_passed = bool(np.isfinite(test_precision) and test_precision >= min_precision)
        rate_passed = bool(np.isfinite(test_pred_long_rate) and test_pred_long_rate <= max_pred_long_rate)
        policy_passed = bool(f1_passed and precision_passed and rate_passed)
        if policy_passed:
            action = "monitor_on_future_oos_only"
        elif not f1_passed:
            action = "score_separation_gap_do_not_promote"
        elif not rate_passed:
            action = "threshold_too_broad_do_not_promote"
        else:
            action = "precision_gap_do_not_promote"
        rows.append(
            {
                "candidate": str(entry.get("profile", "")),
                "candidate_type": _diagnostic_candidate_type(str(entry.get("fold_scope", ""))),
                "fold_scope": str(entry.get("fold_scope", "")),
                "policy_name": policy_name,
                "policy_type": policy_type,
                "threshold_source": threshold_source,
                "threshold_cap": threshold_cap,
                "threshold_mean": threshold_mean,
                "source_selection_metric": source_selection_metric,
                "source_f1": source_f1,
                "source_precision": source_precision,
                "source_pred_long_rate": source_pred_long_rate,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_pred_long_rate": test_pred_long_rate,
                "mean_lift_vs_base": mean_lift_vs_base,
                "mean_forward_return": mean_forward_return,
                "positive_lift_fold_rate": positive_lift_fold_rate,
                "positive_forward_return_fold_rate": positive_forward_return_fold_rate,
                "constraints_satisfied_fold_rate": constraints_satisfied_fold_rate,
                "f1_passed": f1_passed,
                "precision_passed": precision_passed,
                "pred_long_rate_passed": rate_passed,
                "policy_passed_cv_test": policy_passed,
                "selection_guard": "source_threshold_selected_on_validation_not_test",
                "recommended_action": action,
            }
        )

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        summary = diagnostics.get("threshold_summary")
        if summary is not None and not summary.empty:
            add_row(
                entry=entry,
                policy_name="validation_selected_threshold",
                policy_type="threshold",
                threshold_source="validation_selected_threshold",
                threshold_mean=summary_metric(summary, "selected_threshold"),
                source_selection_metric="source_best_f1",
                source_f1=summary_metric(summary, "source_best_f1"),
                test_f1=summary_metric(summary, "test_f1_at_selected_threshold"),
                test_precision=summary_metric(summary, "test_precision_at_selected_threshold"),
                test_recall=summary_metric(summary, "test_recall_at_selected_threshold"),
                test_pred_long_rate=summary_metric(summary, "test_pred_long_rate_at_selected_threshold"),
            )
            add_row(
                entry=entry,
                policy_name="validation_constrained_threshold",
                policy_type="threshold",
                threshold_source="validation_constrained_threshold",
                threshold_mean=summary_metric(summary, "constrained_threshold"),
                source_selection_metric="source_constrained_f1",
                source_f1=summary_metric(summary, "source_constrained_f1"),
                source_precision=summary_metric(summary, "source_constrained_precision"),
                source_pred_long_rate=summary_metric(summary, "source_constrained_pred_long_rate"),
                test_f1=summary_metric(summary, "test_f1_at_constrained_threshold"),
                test_precision=summary_metric(summary, "test_precision_at_constrained_threshold"),
                test_recall=summary_metric(summary, "test_recall_at_constrained_threshold"),
                test_pred_long_rate=summary_metric(summary, "test_pred_long_rate_at_constrained_threshold"),
            )
        calibrated_summary = diagnostics.get("calibrated_threshold_summary")
        if calibrated_summary is not None and not calibrated_summary.empty:
            add_row(
                entry=entry,
                policy_name="calibrated_validation_constrained_threshold",
                policy_type="threshold",
                threshold_source="calibrated_validation_constrained_threshold",
                threshold_mean=summary_metric(calibrated_summary, "constrained_threshold"),
                source_selection_metric="source_constrained_f1",
                source_f1=summary_metric(calibrated_summary, "source_constrained_f1"),
                source_precision=summary_metric(calibrated_summary, "source_constrained_precision"),
                source_pred_long_rate=summary_metric(calibrated_summary, "source_constrained_pred_long_rate"),
                test_f1=summary_metric(calibrated_summary, "test_f1_at_constrained_threshold"),
                test_precision=summary_metric(calibrated_summary, "test_precision_at_constrained_threshold"),
                test_recall=summary_metric(calibrated_summary, "test_recall_at_constrained_threshold"),
                test_pred_long_rate=summary_metric(calibrated_summary, "test_pred_long_rate_at_constrained_threshold"),
            )
        grid_summary = diagnostics.get("threshold_grid_summary")
        if grid_summary is not None and not grid_summary.empty:
            for _, item in grid_summary.iterrows():
                cap = _float(item.to_dict(), "max_pred_long_rate")
                add_row(
                    entry=entry,
                    policy_name=f"validation_threshold_cap_{cap:.2f}",
                    policy_type="threshold_cap",
                    threshold_source="validation_threshold_cap_sweep",
                    threshold_cap=cap,
                    threshold_mean=_float(item.to_dict(), "threshold_mean"),
                    source_selection_metric="mean_source_f1",
                    source_f1=_float(item.to_dict(), "mean_source_f1"),
                    source_precision=_float(item.to_dict(), "mean_source_precision"),
                    source_pred_long_rate=_float(item.to_dict(), "mean_source_pred_long_rate"),
                    test_f1=_float(item.to_dict(), "mean_f1"),
                    test_precision=_float(item.to_dict(), "mean_precision"),
                    test_recall=_float(item.to_dict(), "mean_recall"),
                    test_pred_long_rate=_float(item.to_dict(), "mean_selection_rate"),
                    mean_lift_vs_base=_float(item.to_dict(), "mean_lift_vs_base"),
                    mean_forward_return=_float(item.to_dict(), "mean_forward_return"),
                    positive_lift_fold_rate=_float(item.to_dict(), "positive_lift_fold_rate"),
                    positive_forward_return_fold_rate=_float(item.to_dict(), "positive_forward_return_fold_rate"),
                    constraints_satisfied_fold_rate=_float(item.to_dict(), "constraints_satisfied_fold_rate"),
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "policy_passed_cv_test", "test_f1", "test_pred_long_rate"],
            ascending=[True, True, True, False, False, True],
        )
        .reset_index(drop=True)
    )

def _threshold_policy_review_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Threshold Policy Review", ""]
    if frame.empty:
        lines.append("No threshold policy rows were produced.")
        return "\n".join(lines)
    lines.append(
        "Policies are evaluated on CV test folds, but threshold values are selected from validation/source folds. "
        "Rows are diagnostics only and must not be promoted from the seen holdout."
    )
    lines.append("")
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "test_f1",
        "test_precision",
        "test_pred_long_rate",
        "source_f1",
        "mean_lift_vs_base",
        "mean_forward_return",
        "policy_passed_cv_test",
        "recommended_action",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_threshold_policy_review(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "threshold_policy_review.csv", index=False)
    (path / "threshold_policy_review.md").write_text(_threshold_policy_review_markdown(frame), encoding="utf-8")
    _write_json(path / "threshold_policy_review.json", {"rows": frame.to_dict(orient="records")})

def _threshold_metrics_at_value(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    labels_array = labels.astype(int).to_numpy()
    scores_array = scores.astype(float).to_numpy()
    predictions = scores_array >= float(threshold)
    tp = int(((labels_array == 1) & predictions).sum())
    fp = int(((labels_array == 0) & predictions).sum())
    fn = int(((labels_array == 1) & ~predictions).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(predictions.mean()) if len(predictions) else 0.0,
    }

def _threshold_selection_stats(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    if frame.empty or "prob_long" not in frame.columns or "label" not in frame.columns:
        return {
            "actual_long_rate": np.nan,
            "label_lift_vs_base": np.nan,
            "mean_forward_return": np.nan,
            "mean_tb_return": np.nan,
        }
    scores = pd.to_numeric(frame["prob_long"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    base_rate = float(labels.mean()) if labels.notna().any() else np.nan
    selected = frame.loc[scores >= float(threshold)].copy()
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

def _regime_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([column for column in frame.columns if str(column).startswith("regime_prob_")])

def _with_dominant_regime(frame: pd.DataFrame) -> pd.DataFrame:
    regime_columns = _regime_columns(frame)
    if not regime_columns:
        return pd.DataFrame()
    out = frame.copy()
    out["dominant_regime"] = out[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    return out

def _select_validation_threshold(
    frame: pd.DataFrame,
    *,
    max_pred_long_rate: float,
    min_precision: float,
) -> dict[str, float]:
    if frame.empty or not {"label", "prob_long"}.issubset(frame.columns):
        return {
            "threshold": np.nan,
            "source_f1": np.nan,
            "source_precision": np.nan,
            "source_recall": np.nan,
            "source_pred_long_rate": np.nan,
            "constraint_satisfied": False,
        }
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
    if clean.empty:
        return _select_validation_threshold(
            pd.DataFrame(),
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    labels = clean["label"].astype(int)
    scores = pd.to_numeric(clean["prob_long"], errors="coerce")
    candidates = set(float(value) for value in scores.dropna().unique())
    candidates.update(float(value) for value in scores.quantile([0.50, 0.60, 0.70, 0.80, 0.90]).dropna().to_list())
    candidates.add(0.5)
    if scores.notna().any():
        candidates.add(float(scores.max()) + 1e-9)
        candidates.add(float(scores.min()) - 1e-9)
    rows: list[dict[str, float]] = []
    for threshold in sorted(candidates):
        metrics = _threshold_metrics_at_value(labels, scores, threshold)
        constraint = (
            metrics["pred_long_rate"] <= max_pred_long_rate
            and metrics["precision"] >= min_precision
        )
        rows.append(
            {
                "threshold": float(threshold),
                "source_f1": metrics["f1"],
                "source_precision": metrics["precision"],
                "source_recall": metrics["recall"],
                "source_pred_long_rate": metrics["pred_long_rate"],
                "constraint_satisfied": bool(constraint),
            }
        )
    if not rows:
        return _select_validation_threshold(
            pd.DataFrame(),
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    frame_rows = pd.DataFrame(rows)
    constrained = frame_rows.loc[frame_rows["constraint_satisfied"].astype(bool)].copy()
    if constrained.empty:
        constrained = frame_rows.loc[frame_rows["source_pred_long_rate"] <= max_pred_long_rate].copy()
    if constrained.empty:
        constrained = frame_rows.copy()
    selected = constrained.sort_values(
        ["source_f1", "source_precision", "source_pred_long_rate", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "source_f1": float(selected["source_f1"]),
        "source_precision": float(selected["source_precision"]),
        "source_recall": float(selected["source_recall"]),
        "source_pred_long_rate": float(selected["source_pred_long_rate"]),
        "constraint_satisfied": bool(selected["constraint_satisfied"]),
    }

def _metrics_for_masked_predictions(frame: pd.DataFrame, predictions: pd.Series) -> dict[str, float]:
    labels = frame["label"].astype(int).to_numpy()
    pred = predictions.astype(bool).to_numpy()
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    selected = frame.loc[pred].copy()
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(pred.mean()) if len(pred) else 0.0,
        "selected_count": int(pred.sum()),
        "label_lift_vs_base": (
            float(selected["label"].mean() / frame["label"].mean())
            if not selected.empty and float(frame["label"].mean()) > 0
            else np.nan
        ),
        "mean_forward_return": (
            float(pd.to_numeric(selected["forward_return"], errors="coerce").mean())
            if not selected.empty and "forward_return" in selected.columns
            else np.nan
        ),
        "mean_tb_return": (
            float(pd.to_numeric(selected["tb_return"], errors="coerce").mean())
            if not selected.empty and "tb_return" in selected.columns
            else np.nan
        ),
    }

def _regime_threshold_policy_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "policy_name",
        "regime_count",
        "regime_threshold_count",
        "fallback_threshold",
        "regime_thresholds_json",
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "validation_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "official_f1",
        "official_precision",
        "official_recall",
        "official_pred_long_rate",
        "f1_delta_vs_official",
        "precision_delta_vs_official",
        "pred_long_rate_delta_vs_official",
        "policy_passed_fold",
        "selection_guard",
        "reject_reason",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "fold_count",
        "test_f1_mean",
        "test_precision_mean",
        "test_pred_long_rate_mean",
        "test_mean_forward_return_mean",
        "official_f1_mean",
        "official_pred_long_rate_mean",
        "f1_delta_vs_official_mean",
        "pred_long_rate_delta_vs_official_mean",
        "policy_passed_fold_rate",
        "positive_forward_return_fold_rate",
        "regime_threshold_count_mean",
        "reviewable",
        "reject_reason",
        "next_action",
    ]
    cfg = _cfg(config, ["validation", "regime_threshold_policy"], {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(columns=by_fold_columns), pd.DataFrame(columns=summary_columns)
    min_val_rows = int(cfg.get("min_regime_val_rows", 80))
    min_test_rows = int(cfg.get("min_regime_test_rows", 40))
    min_val_longs = int(cfg.get("min_regime_val_longs", 5))
    max_pred_long_rate = float(cfg.get("max_pred_long_rate", _cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70)))
    min_precision = float(cfg.get("min_precision", _cfg(config, ["validation", "threshold_checks", "min_precision"], 0.30)))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        frame = _with_dominant_regime(predictions)
        if frame.empty or not {"fold", "split", "label", "prob_long"}.issubset(frame.columns):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in frame.groupby("fold"):
            fold_id = int(fold)
            validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
            test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            if validation.empty or test.empty:
                continue
            threshold_row = threshold_by_fold.get(fold_id, {})
            fallback = _float(
                threshold_row,
                "official_threshold",
                _float(threshold_row, "constrained_threshold", _float(threshold_row, "selected_threshold", 0.5)),
            )
            if not np.isfinite(fallback):
                fallback = 0.5
            regime_thresholds: dict[int, dict[str, float]] = {}
            for regime, val_regime in validation.groupby("dominant_regime"):
                test_regime = test.loc[test["dominant_regime"].astype(int) == int(regime)]
                if len(val_regime) < min_val_rows or len(test_regime) < min_test_rows:
                    continue
                if int(pd.to_numeric(val_regime["label"], errors="coerce").sum()) < min_val_longs:
                    continue
                selected = _select_validation_threshold(
                    val_regime,
                    max_pred_long_rate=max_pred_long_rate,
                    min_precision=min_precision,
                )
                if np.isfinite(selected["threshold"]):
                    regime_thresholds[int(regime)] = selected

            def thresholds_for(part: pd.DataFrame) -> pd.Series:
                return part["dominant_regime"].astype(int).map(
                    {regime: values["threshold"] for regime, values in regime_thresholds.items()}
                ).fillna(fallback)

            validation_thresholds = thresholds_for(validation)
            test_thresholds = thresholds_for(test)
            validation_predictions = pd.to_numeric(validation["prob_long"], errors="coerce") >= validation_thresholds
            test_predictions = pd.to_numeric(test["prob_long"], errors="coerce") >= test_thresholds
            validation_metrics = _metrics_for_masked_predictions(validation, validation_predictions)
            test_metrics = _metrics_for_masked_predictions(test, test_predictions)
            official_f1 = _float(
                threshold_row,
                "test_f1_at_official_threshold",
                _float(threshold_row, "test_f1_at_constrained_threshold"),
            )
            official_precision = _float(
                threshold_row,
                "test_precision_at_official_threshold",
                _float(threshold_row, "test_precision_at_constrained_threshold"),
            )
            official_recall = _float(
                threshold_row,
                "test_recall_at_official_threshold",
                _float(threshold_row, "test_recall_at_constrained_threshold"),
            )
            official_rate = _float(
                threshold_row,
                "test_pred_long_rate_at_official_threshold",
                _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
            )
            reasons: list[str] = []
            if test_metrics["f1"] < min_long_f1:
                reasons.append("test_f1")
            if test_metrics["precision"] < min_precision:
                reasons.append("test_precision")
            if test_metrics["pred_long_rate"] > max_pred_long_rate:
                reasons.append("test_pred_long_rate")
            if np.isfinite(official_f1) and test_metrics["f1"] <= official_f1:
                reasons.append("f1_not_above_official")
            if np.isfinite(test_metrics["mean_forward_return"]) and test_metrics["mean_forward_return"] <= 0:
                reasons.append("selected_forward_return")
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "policy_name": "validation_regime_specific_threshold",
                    "regime_count": int(test["dominant_regime"].nunique()),
                    "regime_threshold_count": int(len(regime_thresholds)),
                    "fallback_threshold": float(fallback),
                    "regime_thresholds_json": json.dumps(regime_thresholds, sort_keys=True),
                    "validation_f1": validation_metrics["f1"],
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_pred_long_rate": validation_metrics["pred_long_rate"],
                    "test_f1": test_metrics["f1"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_pred_long_rate": test_metrics["pred_long_rate"],
                    "test_label_lift_vs_base": test_metrics["label_lift_vs_base"],
                    "test_mean_forward_return": test_metrics["mean_forward_return"],
                    "test_mean_tb_return": test_metrics["mean_tb_return"],
                    "official_f1": official_f1,
                    "official_precision": official_precision,
                    "official_recall": official_recall,
                    "official_pred_long_rate": official_rate,
                    "f1_delta_vs_official": (
                        test_metrics["f1"] - official_f1 if np.isfinite(official_f1) else np.nan
                    ),
                    "precision_delta_vs_official": (
                        test_metrics["precision"] - official_precision if np.isfinite(official_precision) else np.nan
                    ),
                    "pred_long_rate_delta_vs_official": (
                        test_metrics["pred_long_rate"] - official_rate if np.isfinite(official_rate) else np.nan
                    ),
                    "policy_passed_fold": not bool(reasons),
                    "selection_guard": "per_regime_thresholds_selected_on_validation_only",
                    "reject_reason": ";".join(dict.fromkeys(reasons)),
                }
            )
    by_fold = pd.DataFrame(rows, columns=by_fold_columns) if rows else pd.DataFrame(columns=by_fold_columns)
    if by_fold.empty:
        return by_fold, pd.DataFrame(columns=summary_columns)
    summary_rows: list[dict[str, Any]] = []
    min_delta = float(cfg.get("min_f1_delta_vs_official", 0.01))
    min_pass_rate = float(cfg.get("min_policy_pass_fold_rate", 0.55))
    min_positive_return_rate = float(cfg.get("min_positive_forward_return_fold_rate", 0.55))
    for (candidate, candidate_type, fold_scope, policy_name), part in by_fold.groupby(
        ["candidate", "candidate_type", "fold_scope", "policy_name"],
        dropna=False,
    ):
        f1_delta = _numeric_mean(part, "f1_delta_vs_official")
        pass_rate = float(part["policy_passed_fold"].astype(bool).mean())
        positive_return_rate = float((pd.to_numeric(part["test_mean_forward_return"], errors="coerce") > 0).mean())
        reasons: list[str] = []
        if _numeric_mean(part, "test_f1") < min_long_f1:
            reasons.append("test_f1")
        if _numeric_mean(part, "test_pred_long_rate") > max_pred_long_rate:
            reasons.append("test_pred_long_rate")
        if not np.isfinite(f1_delta) or f1_delta < min_delta:
            reasons.append("f1_delta_vs_official")
        if pass_rate < min_pass_rate:
            reasons.append("policy_pass_fold_rate")
        if positive_return_rate < min_positive_return_rate:
            reasons.append("positive_forward_return_fold_rate")
        reject_reason = ";".join(dict.fromkeys(reasons))
        summary_rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "fold_count": int(part["fold"].nunique()),
                "test_f1_mean": _numeric_mean(part, "test_f1"),
                "test_precision_mean": _numeric_mean(part, "test_precision"),
                "test_pred_long_rate_mean": _numeric_mean(part, "test_pred_long_rate"),
                "test_mean_forward_return_mean": _numeric_mean(part, "test_mean_forward_return"),
                "official_f1_mean": _numeric_mean(part, "official_f1"),
                "official_pred_long_rate_mean": _numeric_mean(part, "official_pred_long_rate"),
                "f1_delta_vs_official_mean": f1_delta,
                "pred_long_rate_delta_vs_official_mean": _numeric_mean(part, "pred_long_rate_delta_vs_official"),
                "policy_passed_fold_rate": pass_rate,
                "positive_forward_return_fold_rate": positive_return_rate,
                "regime_threshold_count_mean": _numeric_mean(part, "regime_threshold_count"),
                "reviewable": not bool(reject_reason),
                "reject_reason": reject_reason,
                "next_action": (
                    "pre_register_regime_threshold_policy_for_future_oos_review"
                    if not reject_reason
                    else "diagnostic_only_do_not_promote"
                ),
            }
        )
    summary = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        .sort_values(["reviewable", "f1_delta_vs_official_mean", "test_f1_mean"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return by_fold, summary

def _regime_threshold_policy_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Regime Threshold Policy Review", ""]
    lines.append(
        "Per-regime thresholds are selected on each fold's validation split and evaluated on that fold's test split. "
        "This is a CV-only diagnostic and must not be promoted from the current holdout."
    )
    if summary.empty:
        lines.extend(["", "No regime-threshold policy rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "test_f1_mean",
        "test_pred_long_rate_mean",
        "official_f1_mean",
        "f1_delta_vs_official_mean",
        "policy_passed_fold_rate",
        "positive_forward_return_fold_rate",
        "reviewable",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_regime_threshold_policy(path: Path, by_fold: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(path / "regime_threshold_policy_by_fold.csv", index=False)
    summary.to_csv(path / "regime_threshold_policy_summary.csv", index=False)
    (path / "regime_threshold_policy.md").write_text(_regime_threshold_policy_markdown(summary), encoding="utf-8")
    _write_json(
        path / "regime_threshold_policy.json",
        {
            "regime_threshold_policy_by_fold": by_fold.to_dict(orient="records"),
            "regime_threshold_policy_summary": summary.to_dict(orient="records"),
        },
    )

def _regime_stability_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forensics_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "regime",
        "count",
        "row_share",
        "fold_rank_ic",
        "fold_rank_ic_bucket",
        "regime_rank_ic",
        "regime_label_long_rate",
        "regime_prob_long_mean",
        "regime_score_gap",
        "regime_forward_return_mean",
        "official_f1_in_regime",
        "official_pred_long_rate_in_regime",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "regime",
        "fold_count",
        "row_share_mean",
        "row_share_negative_fold_mean",
        "row_share_positive_fold_mean",
        "row_share_gap_negative_minus_positive",
        "regime_rank_ic_mean",
        "regime_rank_ic_std",
        "regime_negative_ic_fraction",
        "fold_rank_ic_when_regime_present_mean",
        "official_f1_in_regime_mean",
        "official_pred_long_rate_in_regime_mean",
        "suspect_score",
        "likely_issue",
        "recommended_action",
    ]
    rows: list[dict[str, Any]] = []
    bad_ic = float(_cfg(config, ["validation", "bad_fold_ic_threshold"], -0.08))
    target_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        frame = _with_dominant_regime(predictions)
        if frame.empty or not {"fold", "split", "label", "prob_long", "forward_return"}.issubset(frame.columns):
            continue
        test_frame = frame.loc[frame["split"].astype(str) == "test"].copy()
        if test_frame.empty:
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_rank = (
            {int(row["fold"]): _float(row.to_dict(), "rank_ic") for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in test_frame.groupby("fold"):
            fold_id = int(fold)
            fold_total = int(len(fold_part))
            fold_ic = float(fold_rank.get(fold_id, _rank_ic_for_frame(fold_part)))
            if np.isfinite(fold_ic) and fold_ic <= bad_ic:
                bucket = "bad_fold"
            elif np.isfinite(fold_ic) and fold_ic < 0:
                bucket = "negative_fold"
            elif np.isfinite(fold_ic) and fold_ic < target_ic:
                bucket = "below_target_fold"
            else:
                bucket = "positive_fold"
            threshold_row = threshold_by_fold.get(fold_id, {})
            official_threshold = _float(
                threshold_row,
                "official_threshold",
                _float(threshold_row, "constrained_threshold", _float(threshold_row, "selected_threshold", 0.5)),
            )
            if not np.isfinite(official_threshold):
                official_threshold = 0.5
            for regime, part in fold_part.groupby("dominant_regime"):
                labels = part["label"].astype(int)
                scores = pd.to_numeric(part["prob_long"], errors="coerce")
                pred = scores >= official_threshold
                metrics = _metrics_for_masked_predictions(part, pred)
                pos_scores = scores.loc[labels == 1]
                neg_scores = scores.loc[labels == 0]
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "fold": fold_id,
                        "regime": int(regime),
                        "count": int(len(part)),
                        "row_share": float(len(part) / fold_total) if fold_total else np.nan,
                        "fold_rank_ic": fold_ic,
                        "fold_rank_ic_bucket": bucket,
                        "regime_rank_ic": _rank_ic_for_frame(part),
                        "regime_label_long_rate": float(labels.mean()) if len(labels) else np.nan,
                        "regime_prob_long_mean": float(scores.mean()) if scores.notna().any() else np.nan,
                        "regime_score_gap": (
                            float(pos_scores.mean() - neg_scores.mean())
                            if not pos_scores.empty and not neg_scores.empty
                            else np.nan
                        ),
                        "regime_forward_return_mean": _numeric_mean(part, "forward_return"),
                        "official_f1_in_regime": metrics["f1"],
                        "official_pred_long_rate_in_regime": metrics["pred_long_rate"],
                    }
                )
    forensics = pd.DataFrame(rows, columns=forensics_columns) if rows else pd.DataFrame(columns=forensics_columns)
    if forensics.empty:
        return forensics, pd.DataFrame(columns=summary_columns)
    summary_rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, regime), part in forensics.groupby(
        ["candidate", "candidate_type", "fold_scope", "regime"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["regime_rank_ic"], errors="coerce")
        negative_part = part.loc[pd.to_numeric(part["fold_rank_ic"], errors="coerce") < 0.0]
        positive_part = part.loc[pd.to_numeric(part["fold_rank_ic"], errors="coerce") >= 0.0]
        row_share_negative = _numeric_mean(negative_part, "row_share")
        row_share_positive = _numeric_mean(positive_part, "row_share")
        share_gap = (
            row_share_negative - row_share_positive
            if np.isfinite(row_share_negative) and np.isfinite(row_share_positive)
            else np.nan
        )
        negative_ic_fraction = float((rank < 0.0).mean()) if rank.notna().any() else np.nan
        suspect = 0.0
        if np.isfinite(share_gap) and share_gap > 0:
            suspect += min(share_gap * 2.0, 1.0)
        if np.isfinite(negative_ic_fraction):
            suspect += negative_ic_fraction
        regime_ic_mean = float(rank.mean()) if rank.notna().any() else np.nan
        if np.isfinite(regime_ic_mean) and regime_ic_mean < 0:
            suspect += 0.5
        if np.isfinite(share_gap) and share_gap > 0.08 and np.isfinite(regime_ic_mean) and regime_ic_mean < target_ic:
            issue = "regime_overrepresented_in_negative_folds"
            action = "inspect_regime_specific_score_distribution_before_new_features"
        elif np.isfinite(regime_ic_mean) and regime_ic_mean < 0:
            issue = "regime_signal_reversal"
            action = "inspect_regime_features_and_threshold_transfer"
        elif np.isfinite(negative_ic_fraction) and negative_ic_fraction >= 0.50:
            issue = "regime_unstable_rank_ic"
            action = "monitor_regime_before_policy_changes"
        else:
            issue = "monitor"
            action = "no_regime_specific_change"
        summary_rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "regime": int(regime),
                "fold_count": int(part["fold"].nunique()),
                "row_share_mean": _numeric_mean(part, "row_share"),
                "row_share_negative_fold_mean": row_share_negative,
                "row_share_positive_fold_mean": row_share_positive,
                "row_share_gap_negative_minus_positive": share_gap,
                "regime_rank_ic_mean": regime_ic_mean,
                "regime_rank_ic_std": float(rank.std(ddof=1)) if rank.notna().sum() > 1 else np.nan,
                "regime_negative_ic_fraction": negative_ic_fraction,
                "fold_rank_ic_when_regime_present_mean": _numeric_mean(part, "fold_rank_ic"),
                "official_f1_in_regime_mean": _numeric_mean(part, "official_f1_in_regime"),
                "official_pred_long_rate_in_regime_mean": _numeric_mean(part, "official_pred_long_rate_in_regime"),
                "suspect_score": float(suspect),
                "likely_issue": issue,
                "recommended_action": action,
            }
        )
    summary = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        .sort_values(["candidate_type", "candidate", "suspect_score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return forensics, summary

def _regime_stability_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Regime Stability Forensics", ""]
    lines.append(
        "This report checks whether HMM regimes are overrepresented in negative or unstable folds. "
        "It is diagnostic only; do not use it to change holdout policy without future-OOS confirmation."
    )
    if summary.empty:
        lines.extend(["", "No regime-stability rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "regime",
        "row_share_gap_negative_minus_positive",
        "regime_rank_ic_mean",
        "regime_negative_ic_fraction",
        "official_f1_in_regime_mean",
        "suspect_score",
        "likely_issue",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_regime_stability(path: Path, forensics: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    forensics.to_csv(path / "regime_stability_forensics.csv", index=False)
    summary.to_csv(path / "regime_stability_summary.csv", index=False)
    (path / "regime_stability.md").write_text(_regime_stability_markdown(summary), encoding="utf-8")
    _write_json(
        path / "regime_stability.json",
        {
            "regime_stability_forensics": forensics.to_dict(orient="records"),
            "regime_stability_summary": summary.to_dict(orient="records"),
        },
    )

def _ewma_last(values: list[float], *, alpha: float = 0.35) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    state = finite[0]
    for value in finite[1:]:
        state = alpha * value + (1.0 - alpha) * state
    return float(state)

def _threshold_transfer_review_frames(
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
        "threshold",
        "threshold_source",
        "threshold_history_count",
        "selection_guard",
        "validation_f1_at_policy_threshold",
        "validation_precision_at_policy_threshold",
        "validation_pred_long_rate_at_policy_threshold",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_actual_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold_count",
        "threshold_mean",
        "threshold_std",
        "threshold_history_mean",
        "validation_f1",
        "validation_precision",
        "validation_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "f1_delta_vs_official",
        "pred_long_rate_delta_vs_official",
        "precision_delta_vs_official",
        "f1_passed",
        "precision_passed",
        "pred_long_rate_passed",
        "policy_passed_cv_test",
        "selection_guard",
        "recommended_action",
    ]
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
        if not {"fold", "label", "prob_long"}.issubset(predictions.columns):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is None or threshold_metrics.empty or "fold" not in threshold_metrics.columns:
            continue
        metric_by_fold = {
            int(row["fold"]): row.to_dict()
            for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()
        }
        sorted_folds = sorted(metric_by_fold)
        prior_constrained: list[float] = []
        prior_selected: list[float] = []
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)

        for fold in sorted_folds:
            fold_part = predictions.loc[predictions["fold"].astype(int) == int(fold)].copy()
            if fold_part.empty:
                continue
            if "split" in fold_part.columns:
                validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
                test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            else:
                validation = fold_part.copy()
                test = fold_part.copy()
            if test.empty:
                continue
            metrics_row = metric_by_fold[fold]
            current_constrained = _float(metrics_row, "constrained_threshold")
            current_selected = _float(metrics_row, "selected_threshold")
            current_official = _float(metrics_row, "official_threshold", current_constrained)
            policy_thresholds: list[dict[str, Any]] = [
                {
                    "policy_name": "official_threshold_policy",
                    "policy_type": "current_fold_validation",
                    "threshold": current_official,
                    "threshold_source": str(metrics_row.get("official_threshold_source", "validation_constrained_threshold")),
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
                {
                    "policy_name": "validation_constrained_threshold",
                    "policy_type": "current_fold_validation",
                    "threshold": current_constrained,
                    "threshold_source": "validation_constrained_threshold",
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
                {
                    "policy_name": "validation_selected_threshold",
                    "policy_type": "current_fold_validation",
                    "threshold": current_selected,
                    "threshold_source": "validation_selected_threshold",
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
            ]
            if prior_constrained:
                policy_thresholds.extend(
                    [
                        {
                            "policy_name": "past_median_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(np.median(prior_constrained)),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "past_mean_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(np.mean(prior_constrained)),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "past_ewma_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": _ewma_last(prior_constrained),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "previous_fold_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(prior_constrained[-1]),
                            "threshold_source": "previous_validation_constrained_threshold",
                            "threshold_history_count": 1,
                            "selection_guard": "previous_fold_validation_threshold_only",
                        },
                    ]
                )
            if prior_selected:
                policy_thresholds.append(
                    {
                        "policy_name": "past_median_selected_threshold",
                        "policy_type": "causal_threshold_transfer",
                        "threshold": float(np.median(prior_selected)),
                        "threshold_source": "historical_validation_selected_thresholds",
                        "threshold_history_count": len(prior_selected),
                        "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                    }
                )

            for policy in policy_thresholds:
                threshold = float(policy["threshold"])
                if not np.isfinite(threshold):
                    continue
                threshold = float(np.clip(threshold, 0.0, 1.0))
                test_metrics = _threshold_metrics_at_value(test["label"], test["prob_long"], threshold)
                selection_stats = _threshold_selection_stats(test, threshold)
                if validation.empty:
                    validation_metrics = {"f1": np.nan, "precision": np.nan, "pred_long_rate": np.nan}
                else:
                    validation_metrics = _threshold_metrics_at_value(
                        validation["label"],
                        validation["prob_long"],
                        threshold,
                    )
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "policy_name": policy["policy_name"],
                        "policy_type": policy["policy_type"],
                        "fold": int(fold),
                        "threshold": threshold,
                        "threshold_source": policy["threshold_source"],
                        "threshold_history_count": int(policy["threshold_history_count"]),
                        "selection_guard": policy["selection_guard"],
                        "validation_f1_at_policy_threshold": validation_metrics["f1"],
                        "validation_precision_at_policy_threshold": validation_metrics["precision"],
                        "validation_pred_long_rate_at_policy_threshold": validation_metrics["pred_long_rate"],
                        "test_f1": test_metrics["f1"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_pred_long_rate": test_metrics["pred_long_rate"],
                        "test_actual_long_rate": selection_stats["actual_long_rate"],
                        "test_label_lift_vs_base": selection_stats["label_lift_vs_base"],
                        "test_mean_forward_return": selection_stats["mean_forward_return"],
                        "test_mean_tb_return": selection_stats["mean_tb_return"],
                    }
                )
            if np.isfinite(current_constrained):
                prior_constrained.append(float(current_constrained))
            if np.isfinite(current_selected):
                prior_selected.append(float(current_selected))

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
        official = by_fold.loc[
            (by_fold["candidate"].astype(str) == str(candidate))
            & (by_fold["fold_scope"].astype(str) == str(fold_scope))
            & (by_fold["policy_name"].astype(str) == "official_threshold_policy")
        ]
        official_f1 = float(pd.to_numeric(official["test_f1"], errors="coerce").mean()) if not official.empty else np.nan
        official_rate = (
            float(pd.to_numeric(official["test_pred_long_rate"], errors="coerce").mean())
            if not official.empty
            else np.nan
        )
        official_precision = (
            float(pd.to_numeric(official["test_precision"], errors="coerce").mean())
            if not official.empty
            else np.nan
        )
        test_f1 = float(pd.to_numeric(part["test_f1"], errors="coerce").mean())
        test_precision = float(pd.to_numeric(part["test_precision"], errors="coerce").mean())
        test_pred_rate = float(pd.to_numeric(part["test_pred_long_rate"], errors="coerce").mean())
        f1_passed = bool(np.isfinite(test_f1) and test_f1 > min_long_f1)
        precision_passed = bool(np.isfinite(test_precision) and test_precision >= min_precision)
        rate_passed = bool(np.isfinite(test_pred_rate) and test_pred_rate <= max_pred_long_rate)
        policy_passed = bool(f1_passed and precision_passed and rate_passed)
        f1_delta = test_f1 - official_f1 if np.isfinite(official_f1) else np.nan
        rate_delta = test_pred_rate - official_rate if np.isfinite(official_rate) else np.nan
        precision_delta = test_precision - official_precision if np.isfinite(official_precision) else np.nan
        if policy_passed and np.isfinite(f1_delta) and f1_delta >= 0.005:
            action = "pre_register_for_future_oos_threshold_policy"
        elif not rate_passed:
            action = "reject_threshold_too_broad"
        elif not f1_passed:
            action = "score_separation_gap_not_threshold_only"
        elif np.isfinite(f1_delta) and f1_delta < -0.005:
            action = "reject_weaker_than_official"
        else:
            action = "monitor_no_clear_advantage"
        summaries.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "fold_count": int(part["fold"].nunique()),
                "threshold_mean": float(pd.to_numeric(part["threshold"], errors="coerce").mean()),
                "threshold_std": float(pd.to_numeric(part["threshold"], errors="coerce").std(ddof=0)),
                "threshold_history_mean": float(pd.to_numeric(part["threshold_history_count"], errors="coerce").mean()),
                "validation_f1": float(pd.to_numeric(part["validation_f1_at_policy_threshold"], errors="coerce").mean()),
                "validation_precision": float(pd.to_numeric(part["validation_precision_at_policy_threshold"], errors="coerce").mean()),
                "validation_pred_long_rate": float(
                    pd.to_numeric(part["validation_pred_long_rate_at_policy_threshold"], errors="coerce").mean()
                ),
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": float(pd.to_numeric(part["test_recall"], errors="coerce").mean()),
                "test_pred_long_rate": test_pred_rate,
                "test_label_lift_vs_base": float(pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce").mean()),
                "test_mean_forward_return": float(pd.to_numeric(part["test_mean_forward_return"], errors="coerce").mean()),
                "test_mean_tb_return": float(pd.to_numeric(part["test_mean_tb_return"], errors="coerce").mean()),
                "positive_lift_fold_rate": float((pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce") > 1.0).mean()),
                "positive_forward_return_fold_rate": float(
                    (pd.to_numeric(part["test_mean_forward_return"], errors="coerce") > 0.0).mean()
                ),
                "f1_delta_vs_official": f1_delta,
                "pred_long_rate_delta_vs_official": rate_delta,
                "precision_delta_vs_official": precision_delta,
                "f1_passed": f1_passed,
                "precision_passed": precision_passed,
                "pred_long_rate_passed": rate_passed,
                "policy_passed_cv_test": policy_passed,
                "selection_guard": ";".join(sorted({str(value) for value in part["selection_guard"].dropna()})),
                "recommended_action": action,
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "policy_passed_cv_test", "test_f1", "test_pred_long_rate"],
            ascending=[True, True, True, False, False, True],
        )
        .reset_index(drop=True)
    )
    return summary, by_fold

def _threshold_transfer_review_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Threshold Transfer Review", ""]
    if summary.empty:
        lines.append("No threshold transfer rows were produced.")
        return "\n".join(lines)
    lines.append(
        "This report compares current validation thresholds with causal threshold-transfer policies "
        "built only from prior validation folds. It is diagnostic only and must not override Phase 1 gates."
    )
    lines.append("")
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "fold_count",
        "test_f1",
        "test_precision",
        "test_pred_long_rate",
        "f1_delta_vs_official",
        "test_label_lift_vs_base",
        "positive_lift_fold_rate",
        "policy_passed_cv_test",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not by_fold.empty:
        lines.append("")
        lines.append(f"By-fold rows: {len(by_fold)}. See `threshold_transfer_by_fold.csv` for fold-level evidence.")
    return "\n".join(lines)

def _write_threshold_transfer_review(path: Path, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "threshold_transfer_review.csv", index=False)
    by_fold.to_csv(path / "threshold_transfer_by_fold.csv", index=False)
    (path / "threshold_transfer_review.md").write_text(
        _threshold_transfer_review_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "threshold_transfer_review.json",
        {
            "summary": summary.to_dict(orient="records"),
            "by_fold": by_fold.to_dict(orient="records"),
        },
    )
