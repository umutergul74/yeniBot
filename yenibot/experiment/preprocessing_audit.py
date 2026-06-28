from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_preprocessing_audit(
    entries: list[dict[str, Any]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Collect fold preprocess decisions into one experiment-level report."""

    frames: list[pd.DataFrame] = []
    for entry in entries:
        scope_dir = Path(entry.get("scope_dir") or entry.get("output_dir") or "")
        path = scope_dir / "preprocessing_audit.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "fold_scope", str(entry.get("fold_scope", "")))
        frame.insert(0, "profile", str(entry.get("profile", "")))
        frames.append(frame)

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "profile",
                "fold_scope",
                "fold",
                "feature",
                "clip_enabled",
                "clip_lower",
                "clip_upper",
                "train_clip_fraction",
                "stability_checked",
                "block_count",
                "prior_rank_ic",
                "recent_rank_ic",
                "prior_label_gap",
                "recent_label_gap",
                "rank_ic_sign_agreement",
                "rank_ic_reversal",
                "label_gap_reversal",
                "masked",
                "mask_reason",
                "block_rank_ics",
                "block_label_gaps",
            ]
        )
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "preprocessing_audit.csv", index=False)
    if combined.empty:
        summary = pd.DataFrame(
            columns=[
                "profile",
                "fold_scope",
                "fold_count",
                "audited_feature_count",
                "masked_decision_count",
                "masked_fold_count",
                "mean_train_clip_fraction",
            ]
        )
    else:
        working = combined.copy()
        if "train_clip_fraction" not in working:
            working["train_clip_fraction"] = np.nan
        working["masked"] = working["masked"].fillna(False).map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
        working["train_clip_fraction"] = pd.to_numeric(
            working["train_clip_fraction"],
            errors="coerce",
        )
        working["masked_fold"] = working["fold"].where(working["masked"])
        summary = (
            working.groupby(["profile", "fold_scope"], observed=True)
            .agg(
                fold_count=("fold", "nunique"),
                audited_feature_count=("feature", "nunique"),
                masked_decision_count=("masked", "sum"),
                masked_fold_count=("masked_fold", "nunique"),
                mean_train_clip_fraction=("train_clip_fraction", "mean"),
            )
            .reset_index()
        )
    summary.to_csv(target / "preprocessing_audit_summary.csv", index=False)
    summary.to_json(
        target / "preprocessing_audit_summary.json",
        orient="records",
        indent=2,
    )
    markdown = "# Train-Fold Preprocessing Audit\n\n"
    markdown += (
        "No train-fold clipping or reliability-mask decisions were recorded.\n"
        if summary.empty
        else summary.to_markdown(index=False) + "\n"
    )
    (target / "preprocessing_audit_summary.md").write_text(markdown, encoding="utf-8")
    return combined


def write_sample_weight_audit(
    entries: list[dict[str, Any]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Collect train-fold sample weighting decisions into one report."""

    columns = [
        "profile",
        "fold_scope",
        "fold",
        "component",
        "enabled",
        "row_count",
        "mean_weight",
        "std_weight",
        "min_weight",
        "p10_weight",
        "p50_weight",
        "p90_weight",
        "p99_weight",
        "max_weight",
        "p90_p10_spread",
        "dominant_weight_fraction",
        "active_fraction",
        "effective_sample_size",
        "effective_sample_fraction",
        "selected_column_count",
        "selected_columns",
        "horizon_bars",
        "event_quantile",
        "event_strength",
        "event_aggregation",
        "notes",
    ]
    frames: list[pd.DataFrame] = []
    for entry in entries:
        scope_dir = Path(entry.get("scope_dir") or entry.get("output_dir") or "")
        path = scope_dir / "sample_weight_audit.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "fold_scope", str(entry.get("fold_scope", "")))
        frame.insert(0, "profile", str(entry.get("profile", "")))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined[columns]

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "sample_weight_audit.csv", index=False)
    if combined.empty:
        summary = pd.DataFrame(
            columns=[
                "profile",
                "fold_scope",
                "component",
                "fold_count",
                "mean_effective_sample_fraction",
                "mean_weight",
                "p90_weight",
                "max_weight",
                "mean_p90_p10_spread",
                "max_dominant_weight_fraction",
                "mean_active_fraction",
                "selected_column_count_max",
            ]
        )
    else:
        working = combined.copy()
        for column in (
            "mean_weight",
            "p90_weight",
            "max_weight",
            "p90_p10_spread",
            "dominant_weight_fraction",
            "active_fraction",
            "effective_sample_fraction",
            "selected_column_count",
        ):
            working[column] = pd.to_numeric(working[column], errors="coerce")
        summary = (
            working.groupby(["profile", "fold_scope", "component"], observed=True)
            .agg(
                fold_count=("fold", "nunique"),
                mean_effective_sample_fraction=("effective_sample_fraction", "mean"),
                mean_weight=("mean_weight", "mean"),
                p90_weight=("p90_weight", "mean"),
                max_weight=("max_weight", "max"),
                mean_p90_p10_spread=("p90_p10_spread", "mean"),
                max_dominant_weight_fraction=("dominant_weight_fraction", "max"),
                mean_active_fraction=("active_fraction", "mean"),
                selected_column_count_max=("selected_column_count", "max"),
            )
            .reset_index()
        )
    summary.to_csv(target / "sample_weight_audit_summary.csv", index=False)
    summary.to_json(
        target / "sample_weight_audit_summary.json",
        orient="records",
        indent=2,
    )
    markdown = "# Train-Fold Sample Weight Audit\n\n"
    markdown += (
        "No sample weighting decisions were recorded.\n"
        if summary.empty
        else summary.to_markdown(index=False) + "\n"
    )
    (target / "sample_weight_audit_summary.md").write_text(markdown, encoding="utf-8")
    return combined


def write_auxiliary_task_audit(
    entries: list[dict[str, Any]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Collect fold-level auxiliary-head evidence without changing primary metrics."""

    metric_columns = [
        "fold",
        "enabled",
        "weight",
        "target_scale",
        "target_clip",
        "huber_beta",
        "val_auxiliary_return_rank_ic",
        "test_auxiliary_return_rank_ic",
        "val_auxiliary_return_mae",
        "test_auxiliary_return_mae",
        "test_auxiliary_probability_rank_correlation",
    ]
    columns = ["profile", "fold_scope", *metric_columns]
    frames: list[pd.DataFrame] = []
    for entry in entries:
        scope_dir = Path(entry.get("scope_dir") or entry.get("output_dir") or "")
        path = scope_dir / "auxiliary_task_audit.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "fold_scope", str(entry.get("fold_scope", "")))
        frame.insert(0, "profile", str(entry.get("profile", "")))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined[columns]

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "auxiliary_task_audit.csv", index=False)
    if combined.empty:
        summary = pd.DataFrame(
            columns=[
                "profile",
                "fold_scope",
                "enabled",
                "fold_count",
                "mean_test_auxiliary_return_rank_ic",
                "positive_test_auxiliary_return_rank_ic_fraction",
                "mean_test_auxiliary_return_mae",
                "mean_test_auxiliary_probability_rank_correlation",
            ]
        )
    else:
        working = combined.copy()
        working["enabled"] = working["enabled"].fillna(False).map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
        numeric = [
            "test_auxiliary_return_rank_ic",
            "test_auxiliary_return_mae",
            "test_auxiliary_probability_rank_correlation",
        ]
        for column in numeric:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working["positive_aux_ic"] = (
            working["test_auxiliary_return_rank_ic"] > 0
        ).where(working["test_auxiliary_return_rank_ic"].notna())
        summary = (
            working.groupby(
                ["profile", "fold_scope", "enabled"],
                observed=True,
                dropna=False,
            )
            .agg(
                fold_count=("fold", "nunique"),
                mean_test_auxiliary_return_rank_ic=(
                    "test_auxiliary_return_rank_ic",
                    "mean",
                ),
                positive_test_auxiliary_return_rank_ic_fraction=(
                    "positive_aux_ic",
                    "mean",
                ),
                mean_test_auxiliary_return_mae=(
                    "test_auxiliary_return_mae",
                    "mean",
                ),
                mean_test_auxiliary_probability_rank_correlation=(
                    "test_auxiliary_probability_rank_correlation",
                    "mean",
                ),
            )
            .reset_index()
        )
    summary.to_csv(target / "auxiliary_task_audit_summary.csv", index=False)
    summary.to_json(
        target / "auxiliary_task_audit_summary.json",
        orient="records",
        indent=2,
    )
    markdown = "# Auxiliary Task Audit\n\n"
    markdown += (
        "No auxiliary-task decisions were recorded.\n"
        if summary.empty
        else summary.to_markdown(index=False) + "\n"
    )
    (target / "auxiliary_task_audit_summary.md").write_text(
        markdown,
        encoding="utf-8",
    )
    return combined


def write_training_input_audits(
    entries: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write all train-fold input audits used by diagnostics bundles."""

    preprocessing = write_preprocessing_audit(entries, output_dir)
    sample_weights = write_sample_weight_audit(entries, output_dir)
    auxiliary_tasks = write_auxiliary_task_audit(entries, output_dir)
    return preprocessing, sample_weights, auxiliary_tasks
