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
