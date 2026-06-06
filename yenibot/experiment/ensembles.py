"""Seed audits, seed ensembles, profile blends, and profile deltas."""

from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _float,
    _hash_payload,
    _metric_or,
    _set_cfg,
    _slug,
    _write_json,
)

from yenibot.experiment.training import (
    summarize_profile_predictions,
)

__all__ = [
    '_fold_delta_frame',
    '_profile_delta_vs_control',
    '_write_profile_delta',
    '_seed_audit_scope',
    '_seed_audit_entries_to_frames',
    '_available_fold_ids_from_entries',
    '_seed_audit_coverage_frame',
    '_seed_stability_frame',
    '_seed_audit_markdown',
    '_seed_audit_coverage_markdown',
    '_write_seed_audit_files',
    '_seed_from_scope',
    '_seed_ensemble_predictions',
    '_seed_ensemble_entries',
    '_seed_ensemble_frame',
    '_seed_ensemble_markdown',
    '_write_seed_ensemble_files',
    '_prediction_key_columns',
    '_rank_score_by_fold',
    '_profile_blend_predictions',
    '_blend_entry_config',
    '_profile_blend_entries',
    '_profile_blend_frame',
    '_profile_blend_review_frame',
    '_profile_blend_gate_reasons',
    '_select_profile_blend_leader',
    '_profile_blend_leaders',
    '_mark_profile_blend_leaders',
    '_best_profile_blend',
    '_profile_blend_markdown',
    '_write_profile_blend_files',
    '_write_profile_diagnostic_summaries',
]

def _fold_delta_frame(entry: dict[str, Any]) -> pd.DataFrame:
    diagnostics = entry["diagnostics"]
    fold_metrics = diagnostics["fold_metrics"].copy()
    columns = ["fold", "rank_ic", "long_f1", "prauc"]
    optional = [column for column in ("start", "end") if column in fold_metrics.columns]
    frame = fold_metrics[["fold", *optional, *columns[1:]]].copy()
    score_lift = diagnostics["score_lift_by_fold"]
    if score_lift is not None and not score_lift.empty:
        lift_columns = [column for column in ("top_lift_vs_base", "top_minus_bottom_forward_return") if column in score_lift.columns]
        if lift_columns:
            frame = frame.merge(score_lift[["fold", *lift_columns]], on="fold", how="left")
    thresholds = diagnostics["threshold_metrics"]
    if thresholds is not None and not thresholds.empty and "test_f1_at_selected_threshold" in thresholds.columns:
        merge_columns = [
            column
            for column in (
                "fold",
                "test_f1_at_selected_threshold",
                "test_f1_at_constrained_threshold",
                "test_pred_long_rate_at_constrained_threshold",
            )
            if column in thresholds.columns
        ]
        frame = frame.merge(thresholds[merge_columns], on="fold", how="left")
    return frame

def _profile_delta_vs_control(entries: list[dict[str, Any]], control_profile: str) -> pd.DataFrame:
    columns = [
        "profile",
        "fold_scope",
        "fold",
        "start",
        "end",
        "control_rank_ic",
        "candidate_rank_ic",
        "rank_ic_delta",
        "control_top_10_lift",
        "candidate_top_10_lift",
        "top_10_lift_delta",
        "control_threshold_f1",
        "candidate_threshold_f1",
        "threshold_f1_delta",
        "control_prauc",
        "candidate_prauc",
        "prauc_delta",
        "control_long_f1",
        "candidate_long_f1",
        "long_f1_delta",
    ]
    controls = {
        str(entry["fold_scope"]): _fold_delta_frame(entry)
        for entry in entries
        if str(entry["profile"]) == control_profile
    }
    rows = []
    for entry in entries:
        profile = str(entry["profile"])
        fold_scope = str(entry["fold_scope"])
        if profile == control_profile or fold_scope not in controls:
            continue
        control = controls[fold_scope].copy()
        candidate = _fold_delta_frame(entry).copy()
        merged = control.merge(candidate, on="fold", how="inner", suffixes=("_control", "_candidate"))
        for _, row in merged.iterrows():
            start = row.get("start_candidate", row.get("start_control", ""))
            end = row.get("end_candidate", row.get("end_control", ""))
            control_top = _float(row.to_dict(), "top_lift_vs_base_control")
            candidate_top = _float(row.to_dict(), "top_lift_vs_base_candidate")
            control_threshold_f1 = _float(row.to_dict(), "test_f1_at_selected_threshold_control")
            candidate_threshold_f1 = _float(row.to_dict(), "test_f1_at_selected_threshold_candidate")
            control_prauc = _float(row.to_dict(), "prauc_control")
            candidate_prauc = _float(row.to_dict(), "prauc_candidate")
            control_long_f1 = _float(row.to_dict(), "long_f1_control")
            candidate_long_f1 = _float(row.to_dict(), "long_f1_candidate")
            control_rank_ic = _float(row.to_dict(), "rank_ic_control")
            candidate_rank_ic = _float(row.to_dict(), "rank_ic_candidate")
            rows.append(
                {
                    "profile": profile,
                    "fold_scope": fold_scope,
                    "fold": int(row["fold"]),
                    "start": start,
                    "end": end,
                    "control_rank_ic": control_rank_ic,
                    "candidate_rank_ic": candidate_rank_ic,
                    "rank_ic_delta": candidate_rank_ic - control_rank_ic,
                    "control_top_10_lift": control_top,
                    "candidate_top_10_lift": candidate_top,
                    "top_10_lift_delta": candidate_top - control_top,
                    "control_threshold_f1": control_threshold_f1,
                    "candidate_threshold_f1": candidate_threshold_f1,
                    "threshold_f1_delta": candidate_threshold_f1 - control_threshold_f1,
                    "control_prauc": control_prauc,
                    "candidate_prauc": candidate_prauc,
                    "prauc_delta": candidate_prauc - control_prauc,
                    "control_long_f1": control_long_f1,
                    "candidate_long_f1": candidate_long_f1,
                    "long_f1_delta": candidate_long_f1 - control_long_f1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["fold_scope", "profile", "fold"]).reset_index(drop=True)

def _write_profile_delta(path: Path, profile_delta: pd.DataFrame | None) -> None:
    if profile_delta is None:
        return
    profile_delta.to_csv(path / "profile_delta_vs_control.csv", index=False)

def _seed_audit_scope(seed: int) -> str:
    return f"seed_audit_seed_{int(seed):03d}"

def _seed_audit_entries_to_frames(entries: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not fold_scope.startswith("seed_audit_seed_"):
            continue
        seed_text = fold_scope.rsplit("_", 1)[-1]
        try:
            seed = int(seed_text)
        except ValueError:
            seed = np.nan
        row = dict(entry["diagnostics"]["row"])
        row["seed"] = seed
        row["audit_scope"] = fold_scope
        rows.append(row)
    seed_audit = pd.DataFrame(rows).sort_values(["profile", "seed"]).reset_index(drop=True) if rows else pd.DataFrame()
    return seed_audit, _seed_stability_frame(seed_audit)

def _available_fold_ids_from_entries(
    entries: list[dict[str, Any]],
    control_profile: str,
) -> list[int]:
    preferred = [
        entry
        for entry in entries
        if str(entry.get("profile", "")) == str(control_profile)
        and str(entry.get("fold_scope", "")) == "full"
    ]
    candidates = preferred or [
        entry
        for entry in entries
        if str(entry.get("fold_scope", "")) == "full"
    ]
    fold_ids: set[int] = set()
    for entry in candidates:
        predictions = entry.get("predictions")
        if isinstance(predictions, pd.DataFrame) and not predictions.empty and "fold" in predictions.columns:
            fold_ids.update(
                int(value)
                for value in pd.to_numeric(predictions["fold"], errors="coerce").dropna().unique()
            )
    return sorted(fold_ids)

def _seed_audit_coverage_frame(
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    available_fold_ids: list[int] | None = None,
) -> pd.DataFrame:
    columns = [
        "enabled",
        "profile",
        "seed",
        "available_fold_count",
        "configured_fold_count",
        "configured_fold_ids",
        "valid_configured_fold_ids",
        "invalid_configured_fold_ids",
        "observed_fold_count",
        "observed_fold_ids",
        "missing_valid_fold_ids",
        "temporal_span_fraction",
        "minimum_temporal_span_fraction",
        "first_available_fold_covered",
        "last_available_fold_covered",
        "coverage_passed",
        "status",
    ]
    seed_cfg = settings.get("seed_audit", {}) or {}
    enabled = bool(seed_cfg.get("enabled", False))
    if not enabled:
        return pd.DataFrame(
            [
                {
                    "enabled": False,
                    "coverage_passed": True,
                    "status": "disabled",
                }
            ],
            columns=columns,
        )

    available = sorted(
        set(
            int(fold_id)
            for fold_id in (
                available_fold_ids
                if available_fold_ids is not None
                else _available_fold_ids_from_entries(entries, str(settings.get("control_profile", "")))
            )
        )
    )
    if not available:
        observed_universe: set[int] = set()
        for entry in entries:
            predictions = entry.get("predictions")
            if isinstance(predictions, pd.DataFrame) and not predictions.empty and "fold" in predictions.columns:
                observed_universe.update(
                    int(value)
                    for value in pd.to_numeric(predictions["fold"], errors="coerce").dropna().unique()
                )
        available = sorted(observed_universe)
    configured = list(
        dict.fromkeys(int(fold_id) for fold_id in seed_cfg.get("fold_ids", []) or available)
    )
    available_set = set(available)
    valid = [fold_id for fold_id in configured if fold_id in available_set]
    invalid = [fold_id for fold_id in configured if fold_id not in available_set]
    min_span = float(seed_cfg.get("min_temporal_span_fraction", 0.80))
    if valid and available and len(available) == 1:
        span = 1.0
    elif valid and available:
        span = float((max(valid) - min(valid)) / (max(available) - min(available)))
    else:
        span = 0.0
    entry_lookup = {
        (str(entry.get("profile", "")), _seed_from_scope(str(entry.get("fold_scope", "")))): entry
        for entry in entries
        if _seed_from_scope(str(entry.get("fold_scope", ""))) is not None
    }
    rows: list[dict[str, Any]] = []
    profiles = [str(profile) for profile in seed_cfg.get("profiles", []) or [settings.get("control_profile", "")]]
    seeds = [int(seed) for seed in seed_cfg.get("seeds", []) or []]
    for profile in profiles:
        for seed in seeds:
            entry = entry_lookup.get((profile, seed))
            observed: list[int] = []
            if entry is not None:
                predictions = entry.get("predictions")
                if isinstance(predictions, pd.DataFrame) and not predictions.empty and "fold" in predictions.columns:
                    observed = sorted(
                        set(
                            int(value)
                            for value in pd.to_numeric(predictions["fold"], errors="coerce").dropna().unique()
                        )
                    )
            missing = [fold_id for fold_id in valid if fold_id not in set(observed)]
            first_covered = bool(available and min(available) in valid)
            last_covered = bool(available and max(available) in valid)
            passed = bool(
                available
                and not invalid
                and not missing
                and span >= min_span
                and first_covered
                and last_covered
            )
            if invalid:
                status = "invalid_configured_fold_ids"
            elif missing:
                status = "incomplete_seed_fold_outputs"
            elif span < min_span or not first_covered or not last_covered:
                status = "insufficient_temporal_coverage"
            elif not available:
                status = "available_fold_universe_missing"
            else:
                status = "passed"
            rows.append(
                {
                    "enabled": True,
                    "profile": profile,
                    "seed": seed,
                    "available_fold_count": len(available),
                    "configured_fold_count": len(configured),
                    "configured_fold_ids": ",".join(str(value) for value in configured),
                    "valid_configured_fold_ids": ",".join(str(value) for value in valid),
                    "invalid_configured_fold_ids": ",".join(str(value) for value in invalid),
                    "observed_fold_count": len(observed),
                    "observed_fold_ids": ",".join(str(value) for value in observed),
                    "missing_valid_fold_ids": ",".join(str(value) for value in missing),
                    "temporal_span_fraction": span,
                    "minimum_temporal_span_fraction": min_span,
                    "first_available_fold_covered": first_covered,
                    "last_available_fold_covered": last_covered,
                    "coverage_passed": passed,
                    "status": status,
                }
            )
    return pd.DataFrame(rows, columns=columns)

def _seed_stability_frame(seed_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "profile",
        "seed_count",
        "feature_count",
        "fold_count",
        "mean_rank_ic_seed_mean",
        "mean_rank_ic_seed_std",
        "positive_ic_fraction_seed_mean",
        "positive_ic_fraction_seed_std",
        "std_rank_ic_seed_mean",
        "top_10_lift_global_seed_mean",
        "top_10_lift_global_seed_std",
        "test_f1_at_selected_threshold_seed_mean",
        "test_f1_at_selected_threshold_seed_std",
        "test_f1_at_constrained_threshold_seed_mean",
        "test_f1_at_constrained_threshold_seed_std",
        "test_pred_long_rate_at_constrained_threshold_seed_mean",
        "test_pred_long_rate_at_constrained_threshold_seed_std",
        "worst_5_rank_ic_mean_seed_mean",
        "worst_5_rank_ic_mean_seed_std",
    ]
    if seed_audit.empty:
        return pd.DataFrame(columns=columns)

    metric_columns = [
        "mean_rank_ic",
        "positive_ic_fraction",
        "std_rank_ic",
        "top_10_lift_global",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "worst_5_rank_ic_mean",
    ]
    frame = seed_audit.copy()
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    grouped = frame.groupby("profile", as_index=False).agg(
        seed_count=("seed", "nunique"),
        feature_count=("feature_count", "first"),
        fold_count=("fold_count", "first"),
        mean_rank_ic_seed_mean=("mean_rank_ic", "mean"),
        mean_rank_ic_seed_std=("mean_rank_ic", "std"),
        positive_ic_fraction_seed_mean=("positive_ic_fraction", "mean"),
        positive_ic_fraction_seed_std=("positive_ic_fraction", "std"),
        std_rank_ic_seed_mean=("std_rank_ic", "mean"),
        top_10_lift_global_seed_mean=("top_10_lift_global", "mean"),
        top_10_lift_global_seed_std=("top_10_lift_global", "std"),
        test_f1_at_selected_threshold_seed_mean=("test_f1_at_selected_threshold", "mean"),
        test_f1_at_selected_threshold_seed_std=("test_f1_at_selected_threshold", "std"),
        test_f1_at_constrained_threshold_seed_mean=("test_f1_at_constrained_threshold", "mean"),
        test_f1_at_constrained_threshold_seed_std=("test_f1_at_constrained_threshold", "std"),
        test_pred_long_rate_at_constrained_threshold_seed_mean=("test_pred_long_rate_at_constrained_threshold", "mean"),
        test_pred_long_rate_at_constrained_threshold_seed_std=("test_pred_long_rate_at_constrained_threshold", "std"),
        worst_5_rank_ic_mean_seed_mean=("worst_5_rank_ic_mean", "mean"),
        worst_5_rank_ic_mean_seed_std=("worst_5_rank_ic_mean", "std"),
    )
    return grouped[columns].reset_index(drop=True)

def _seed_audit_markdown(seed_audit: pd.DataFrame, seed_stability: pd.DataFrame) -> str:
    lines = ["# Seed Audit", ""]
    if seed_audit.empty:
        lines.append("Seed audit was disabled or produced no completed runs.")
    else:
        display_cols = [
            "profile",
            "seed",
            "fold_count",
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "top_10_lift_global",
            "test_f1_at_selected_threshold",
            "test_f1_at_constrained_threshold",
            "test_pred_long_rate_at_constrained_threshold",
            "worst_5_rank_ic_mean",
        ]
        lines.extend(["## Per Seed", ""])
        visible = seed_audit[[column for column in display_cols if column in seed_audit.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")

    lines.extend(["", "## Stability", ""])
    if seed_stability.empty:
        lines.append("No stability summary available.")
    else:
        lines.append("| " + " | ".join(seed_stability.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(seed_stability.columns)) + " |")
        for _, row in seed_stability.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in seed_stability.columns) + " |")
    return "\n".join(lines)

def _seed_audit_coverage_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Seed Audit Coverage", ""]
    lines.append(
        "Configured seed folds are checked against the available purged walk-forward fold universe. "
        "Unavailable or missing folds are never treated as completed coverage."
    )
    if frame.empty:
        lines.extend(["", "No seed coverage rows were produced."])
        return "\n".join(lines)
    lines.extend(["", "| " + " | ".join(frame.columns) + " |"])
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)

def _write_seed_audit_files(
    path: Path,
    seed_audit: pd.DataFrame,
    seed_stability: pd.DataFrame,
    seed_coverage: pd.DataFrame | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    coverage = seed_coverage if seed_coverage is not None else pd.DataFrame()
    seed_audit.to_csv(path / "seed_audit.csv", index=False)
    seed_stability.to_csv(path / "seed_stability.csv", index=False)
    coverage.to_csv(path / "seed_audit_coverage.csv", index=False)
    (path / "seed_audit.md").write_text(_seed_audit_markdown(seed_audit, seed_stability), encoding="utf-8")
    (path / "seed_audit_coverage.md").write_text(
        _seed_audit_coverage_markdown(coverage),
        encoding="utf-8",
    )
    _write_json(path / "seed_audit.json", {"rows": seed_audit.to_dict(orient="records")})
    _write_json(path / "seed_stability.json", {"rows": seed_stability.to_dict(orient="records")})
    _write_json(
        path / "seed_audit_coverage.json",
        {
            "coverage_passed": bool(
                not coverage.empty and coverage["coverage_passed"].astype(bool).all()
            )
            if "coverage_passed" in coverage.columns
            else False,
            "rows": coverage.to_dict(orient="records"),
        },
    )

def _seed_from_scope(fold_scope: str) -> int | None:
    if not fold_scope.startswith("seed_audit_seed_"):
        return None
    seed_text = fold_scope.rsplit("_", 1)[-1]
    try:
        return int(seed_text)
    except ValueError:
        return None

def _seed_ensemble_predictions(seed_entries: list[dict[str, Any]]) -> pd.DataFrame:
    if len(seed_entries) < 2:
        return pd.DataFrame()

    frames = []
    seeds = []
    for entry in seed_entries:
        seed = _seed_from_scope(str(entry.get("fold_scope", "")))
        if seed is None:
            continue
        prediction = entry["predictions"].copy()
        prediction["_ensemble_seed"] = seed
        frames.append(prediction)
        seeds.append(seed)
    if len(frames) < 2:
        return pd.DataFrame()

    stacked = pd.concat(frames, ignore_index=True)
    required_keys = ["fold", "timestamp"]
    if "split" in stacked.columns:
        required_keys.insert(0, "split")
    if "source_row_position" in stacked.columns:
        required_keys.append("source_row_position")
    key_columns = [column for column in required_keys if column in stacked.columns]
    if not {"fold", "timestamp"}.issubset(key_columns):
        return pd.DataFrame()

    seed_count = len(set(seeds))
    grouped = stacked.groupby(key_columns, dropna=False)
    stats = grouped["prob_long"].agg(
        prob_long_ensemble="mean",
        prob_long_seed_std="std",
        prob_long_seed_min="min",
        prob_long_seed_max="max",
        ensemble_seed_count="count",
    ).reset_index()
    stats = stats.loc[stats["ensemble_seed_count"] == seed_count].copy()
    if stats.empty:
        return pd.DataFrame()

    base = grouped.first().reset_index()
    base = base.merge(stats, on=key_columns, how="inner")
    base["prob_long"] = base["prob_long_ensemble"]
    regime_columns = [column for column in stacked.columns if column.startswith("regime_prob_")]
    if regime_columns:
        regime_avg = grouped[regime_columns].mean().reset_index()
        base = base.drop(columns=[column for column in regime_columns if column in base.columns]).merge(
            regime_avg,
            on=key_columns,
            how="left",
        )
    base = base.drop(columns=["_ensemble_seed", "prob_long_ensemble"], errors="ignore")
    base["ensemble_seeds"] = ",".join(str(seed) for seed in sorted(set(seeds)))
    return base.sort_values(key_columns).reset_index(drop=True)

def _seed_ensemble_entries(entries: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if _seed_from_scope(fold_scope) is None:
            continue
        grouped.setdefault(str(entry["profile"]), []).append(entry)

    ensemble_entries = []
    for profile, seed_entries in grouped.items():
        predictions = _seed_ensemble_predictions(seed_entries)
        if predictions.empty:
            continue
        feature_columns = list(seed_entries[0]["feature_columns"])
        diagnostics = summarize_profile_predictions(
            predictions,
            config,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope="seed_ensemble",
        )
        ensemble_entries.append(
            {
                "profile": profile,
                "fold_scope": "seed_ensemble",
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": diagnostics["row"],
            }
        )
    return ensemble_entries

def _seed_ensemble_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if str(entry.get("fold_scope", "")) != "seed_ensemble":
            continue
        row = dict(entry["diagnostics"]["row"])
        predictions = entry.get("predictions", pd.DataFrame())
        if isinstance(predictions, pd.DataFrame) and "ensemble_seed_count" in predictions.columns and not predictions.empty:
            row["seed_count"] = int(predictions["ensemble_seed_count"].max())
            row["prob_long_seed_std_mean"] = float(pd.to_numeric(predictions["prob_long_seed_std"], errors="coerce").mean())
            row["prob_long_seed_std_p90"] = float(pd.to_numeric(predictions["prob_long_seed_std"], errors="coerce").quantile(0.90))
            row["ensemble_seeds"] = str(predictions["ensemble_seeds"].iloc[0]) if "ensemble_seeds" in predictions.columns else ""
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

def _seed_ensemble_markdown(seed_ensemble: pd.DataFrame) -> str:
    lines = ["# Seed Ensemble", ""]
    if seed_ensemble.empty:
        lines.append("No seed ensemble was produced.")
        return "\n".join(lines)
    display_cols = [
        "profile",
        "fold_scope",
        "seed_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "top_10_lift_global",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "prob_long_seed_std_mean",
        "prob_long_seed_std_p90",
    ]
    visible = seed_ensemble[[column for column in display_cols if column in seed_ensemble.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_seed_ensemble_files(path: Path, seed_ensemble: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    seed_ensemble.to_csv(path / "seed_ensemble.csv", index=False)
    (path / "seed_ensemble.md").write_text(_seed_ensemble_markdown(seed_ensemble), encoding="utf-8")
    _write_json(path / "seed_ensemble.json", {"rows": seed_ensemble.to_dict(orient="records")})

def _prediction_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = []
    if "split" in frame.columns:
        keys.append("split")
    for column in ("fold", "timestamp", "source_row_position"):
        if column in frame.columns:
            keys.append(column)
    return keys

def _rank_score_by_fold(frame: pd.DataFrame) -> pd.Series:
    group_keys = [column for column in ("split", "fold") if column in frame.columns]
    if not group_keys:
        return frame["prob_long"].rank(method="average", pct=True)
    return frame.groupby(group_keys, dropna=False)["prob_long"].rank(method="average", pct=True)

def _profile_blend_predictions(
    entries: list[dict[str, Any]],
    *,
    method: str,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    if len(entries) < 2:
        return pd.DataFrame()

    normalized_weights: list[float] | None = None
    if weights is not None:
        if len(weights) != len(entries):
            raise ValueError("Blend weights must match the number of profile entries")
        raw_weights = np.asarray(weights, dtype=float)
        if not np.isfinite(raw_weights).all() or (raw_weights < 0).any() or raw_weights.sum() <= 0:
            raise ValueError("Blend weights must be finite non-negative values with a positive sum")
        normalized_weights = (raw_weights / raw_weights.sum()).tolist()

    frames = []
    profiles = []
    for idx, entry in enumerate(entries):
        prediction = entry["predictions"].copy()
        profile = str(entry["profile"])
        prediction["_blend_profile"] = profile
        prediction["_blend_weight"] = 1.0 if normalized_weights is None else float(normalized_weights[idx])
        if method in {"rank_mean", "rank_weighted"}:
            prediction["_blend_score"] = _rank_score_by_fold(prediction)
        elif method in {"prob_mean", "prob_weighted"}:
            prediction["_blend_score"] = prediction["prob_long"].astype(float)
        else:
            raise ValueError(f"Unknown profile blend method: {method}")
        prediction["_blend_weighted_score"] = prediction["_blend_score"] * prediction["_blend_weight"]
        frames.append(prediction)
        profiles.append(profile)
    if len(frames) < 2:
        return pd.DataFrame()

    stacked = pd.concat(frames, ignore_index=True)
    key_columns = _prediction_key_columns(stacked)
    if not {"fold", "timestamp"}.issubset(key_columns):
        return pd.DataFrame()

    profile_count = len(set(profiles))
    grouped = stacked.groupby(key_columns, dropna=False)
    if normalized_weights is None:
        stats = grouped["_blend_score"].agg(
            prob_long_blend="mean",
            prob_long_profile_std="std",
            prob_long_profile_min="min",
            prob_long_profile_max="max",
            blend_profile_count="count",
        ).reset_index()
    else:
        stats = grouped.agg(
            prob_long_blend=("_blend_weighted_score", "sum"),
            prob_long_profile_std=("_blend_score", "std"),
            prob_long_profile_min=("_blend_score", "min"),
            prob_long_profile_max=("_blend_score", "max"),
            blend_profile_count=("_blend_score", "count"),
        ).reset_index()
    stats = stats.loc[stats["blend_profile_count"] == profile_count].copy()
    if stats.empty:
        return pd.DataFrame()

    base = grouped.first().reset_index()
    drop_columns = ["_blend_profile", "_blend_score", "_blend_weight", "_blend_weighted_score", "prob_long_blend"]
    base = base.drop(columns=[column for column in drop_columns if column in base.columns], errors="ignore")
    base = base.merge(stats, on=key_columns, how="inner")
    base["prob_long"] = base["prob_long_blend"]
    regime_columns = [column for column in stacked.columns if column.startswith("regime_prob_")]
    if regime_columns:
        regime_avg = grouped[regime_columns].mean().reset_index()
        base = base.drop(columns=[column for column in regime_columns if column in base.columns]).merge(
            regime_avg,
            on=key_columns,
            how="left",
        )
    base = base.drop(columns=["prob_long_blend"], errors="ignore")
    base["blend_method"] = method
    base["blend_profiles"] = ",".join(profiles)
    if normalized_weights is not None:
        base["blend_weights"] = ",".join(f"{weight:.6g}" for weight in normalized_weights)
    return base.sort_values(key_columns).reset_index(drop=True)

def _blend_entry_config(config: dict[str, Any], profile: str, feature_columns: list[str], *, description: str) -> dict[str, Any]:
    blend_cfg = copy.deepcopy(config)
    profiles = copy.deepcopy(_cfg(blend_cfg, ["features", "profiles"], {}) or {})
    profiles[profile] = {
        "description": description,
        "include_patterns": list(feature_columns),
        "exclude_patterns": [],
    }
    _set_cfg(blend_cfg, ["features", "profiles"], profiles)
    return blend_cfg

def _profile_blend_entries(entries: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    full_entries = [
        entry
        for entry in entries
        if str(entry.get("fold_scope", "")) == "full"
        and not str(entry.get("profile", "")).startswith("blend_")
    ]
    if len(full_entries) < 2:
        return []

    blend_settings = _cfg(config, ["experiments", "profile_blends"], {}) or {}
    include_auto_equal = bool(blend_settings.get("include_auto_equal_weight", True))
    include_auto_rank = bool(blend_settings.get("include_auto_rank_mean", True))
    blend_entries = []
    entry_by_profile = {str(entry["profile"]): entry for entry in full_entries}

    def append_blend(
        combo: list[dict[str, Any]],
        *,
        method: str,
        weights: list[float] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        profiles = [str(entry["profile"]) for entry in combo]
        feature_columns = sorted({column for entry in combo for column in entry.get("feature_columns", [])})
        combo_hash = _hash_payload({"profiles": profiles, "method": method, "weights": weights})[:10]
        predictions = _profile_blend_predictions(combo, method=method, weights=weights)
        if predictions.empty:
            return
        profile = _slug(name) if name else f"blend_{method}_{combo_hash}"
        if not profile.startswith("blend_"):
            profile = f"blend_{profile}"
        blend_cfg = _blend_entry_config(
            config,
            profile,
            feature_columns,
            description=description or f"Diagnostic {method} blend of: {', '.join(profiles)}",
        )
        diagnostics = summarize_profile_predictions(
            predictions,
            blend_cfg,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope=f"blend_{method}",
        )
        diagnostics["row"]["blend_profiles"] = ",".join(profiles)
        diagnostics["row"]["blend_method"] = method
        diagnostics["row"]["profile_count"] = len(profiles)
        if weights is not None:
            raw_weights = np.asarray(weights, dtype=float)
            normalized = raw_weights / raw_weights.sum()
            diagnostics["row"]["blend_weights"] = ",".join(f"{weight:.6g}" for weight in normalized)
        diagnostics["row"]["prob_long_profile_std_mean"] = float(
            pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").mean()
        )
        diagnostics["row"]["prob_long_profile_std_p90"] = float(
            pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").quantile(0.90)
        )
        blend_entries.append(
            {
                "profile": profile,
                "fold_scope": f"blend_{method}",
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": diagnostics["row"],
                "config": blend_cfg,
            }
        )

    if include_auto_equal:
        combos = list(combinations(full_entries, 2))
        if len(full_entries) > 2:
            combos.append(tuple(full_entries))
        for combo in combos:
            methods = ["prob_mean"]
            if include_auto_rank:
                methods.append("rank_mean")
            for method in methods:
                append_blend(list(combo), method=method)

    for spec in blend_settings.get("weighted", []) or []:
        if not isinstance(spec, dict) or not bool(spec.get("enabled", True)):
            continue
        profiles = [str(profile) for profile in spec.get("profiles", []) or []]
        if len(profiles) < 2:
            continue
        if any(profile not in entry_by_profile for profile in profiles):
            continue
        method = str(spec.get("method", "prob_weighted"))
        weights = [float(weight) for weight in spec.get("weights", []) or []]
        append_blend(
            [entry_by_profile[profile] for profile in profiles],
            method=method,
            weights=weights,
            name=str(spec.get("name", "")) or None,
            description=str(spec.get("description", "")) or None,
        )
    return blend_entries

def _profile_blend_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if not str(entry.get("fold_scope", "")).startswith("blend_"):
            continue
        row = dict(entry["diagnostics"]["row"])
        predictions = entry.get("predictions", pd.DataFrame())
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            row["blend_profiles"] = str(predictions["blend_profiles"].iloc[0]) if "blend_profiles" in predictions.columns else row.get("blend_profiles", "")
            row["blend_method"] = str(predictions["blend_method"].iloc[0]) if "blend_method" in predictions.columns else row.get("blend_method", "")
            row["blend_weights"] = str(predictions["blend_weights"].iloc[0]) if "blend_weights" in predictions.columns else row.get("blend_weights", "")
            row["profile_count"] = int(predictions["blend_profile_count"].max()) if "blend_profile_count" in predictions.columns else row.get("profile_count", 0)
            row["prob_long_profile_std_mean"] = float(pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").mean())
            row["prob_long_profile_std_p90"] = float(pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

def _profile_blend_review_frame(
    profile_blend: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    control_profile: str,
) -> pd.DataFrame:
    if profile_blend.empty:
        return profile_blend
    reviewed = profile_blend.copy()
    control_rows = comparison[
        (comparison["profile"] == control_profile)
        & (comparison["fold_scope"] == "full")
    ]
    if control_rows.empty:
        reviewed["control_profile"] = control_profile
        reviewed["reviewable"] = False
        reviewed["review_reason"] = "missing_full_control"
        return reviewed

    control = control_rows.iloc[0].to_dict()
    gates = _cfg(config, ["experiments", "profile_blend_review_gates"], {}) or {}
    min_mean_delta = float(gates.get("min_mean_rank_ic_delta", 0.005))
    max_std_delta = float(gates.get("max_std_rank_ic_delta", 0.0))
    min_positive = float(gates.get("min_positive_ic_fraction", 0.70))
    min_top_delta = float(gates.get("min_top_10_lift_global_delta", 0.02))
    leader_gates = _cfg(config, ["experiments", "profile_blend_leader_gates"], {}) or {}
    tail_lift_gates = leader_gates.get("tail_lift", gates) or gates
    stability_gates = leader_gates.get("stability", {}) or {}
    balanced_gates = leader_gates.get("balanced")

    rows = []
    for _, item in reviewed.iterrows():
        row = item.to_dict()
        row["control_profile"] = control_profile
        row["mean_rank_ic_delta_vs_control"] = _float(row, "mean_rank_ic") - _float(control, "mean_rank_ic")
        row["std_rank_ic_delta_vs_control"] = _float(row, "std_rank_ic") - _float(control, "std_rank_ic")
        row["positive_ic_fraction_delta_vs_control"] = _float(row, "positive_ic_fraction") - _float(control, "positive_ic_fraction")
        row["top_10_lift_global_delta_vs_control"] = _float(row, "top_10_lift_global") - _float(control, "top_10_lift_global")
        row["selected_threshold_f1_delta_vs_control"] = _metric_or(
            row,
            "test_f1_at_official_threshold",
            _metric_or(
                row,
                "test_f1_at_constrained_threshold",
                _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
            ),
        ) - _metric_or(
            control,
            "test_f1_at_official_threshold",
            _metric_or(
                control,
                "test_f1_at_constrained_threshold",
                _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
            ),
        )
        row["worst_5_rank_ic_delta_vs_control"] = _float(row, "worst_5_rank_ic_mean") - _float(control, "worst_5_rank_ic_mean")
        reasons = []
        if row["mean_rank_ic_delta_vs_control"] < min_mean_delta:
            reasons.append("mean_rank_ic_delta")
        if row["std_rank_ic_delta_vs_control"] > max_std_delta:
            reasons.append("std_rank_ic_delta")
        if _float(row, "positive_ic_fraction") < min_positive:
            reasons.append("positive_ic_fraction")
        if row["top_10_lift_global_delta_vs_control"] < min_top_delta:
            reasons.append("top_10_lift_global_delta")
        if not bool(row.get("mtf_leakage_passed", False)):
            reasons.append("mtf_leakage")
        if not bool(row.get("stationarity_policy_passed", False)):
            reasons.append("stationarity_policy")
        row["reviewable"] = not reasons
        row["review_reason"] = ";".join(reasons)
        tail_lift_reasons = _profile_blend_gate_reasons(row, tail_lift_gates)
        stability_reasons = _profile_blend_gate_reasons(row, stability_gates)
        balanced_reasons = _profile_blend_gate_reasons(row, balanced_gates or {})
        row["tail_lift_eligible"] = not tail_lift_reasons
        row["tail_lift_reason"] = ";".join(tail_lift_reasons)
        row["stability_eligible"] = not stability_reasons
        row["stability_reason"] = ";".join(stability_reasons)
        row["balanced_eligible"] = bool(balanced_gates) and not balanced_reasons
        row["balanced_reason"] = ";".join(balanced_reasons if balanced_gates else ["not_configured"])
        rows.append(row)

    if not rows:
        return reviewed
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            ["reviewable", "mean_rank_ic", "top_10_lift_global", "worst_5_rank_ic_mean"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    return _mark_profile_blend_leaders(frame)

def _profile_blend_gate_reasons(row: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    reasons = []
    if not gates:
        return reasons
    checks = [
        ("min_mean_rank_ic_delta", "mean_rank_ic_delta_vs_control", "mean_rank_ic_delta", "min"),
        ("max_std_rank_ic_delta", "std_rank_ic_delta_vs_control", "std_rank_ic_delta", "max"),
        ("min_positive_ic_fraction", "positive_ic_fraction", "positive_ic_fraction", "min"),
        ("min_top_10_lift_global", "top_10_lift_global", "top_10_lift_global", "min"),
        ("min_top_10_lift_global_delta", "top_10_lift_global_delta_vs_control", "top_10_lift_global_delta", "min"),
        ("min_worst_5_rank_ic_delta", "worst_5_rank_ic_delta_vs_control", "worst_5_rank_ic_delta", "min"),
    ]
    for gate_key, metric_key, reason, direction in checks:
        if gate_key not in gates:
            continue
        value = _float(row, metric_key)
        gate = float(gates[gate_key])
        if direction == "min" and value < gate:
            reasons.append(reason)
        if direction == "max" and value > gate:
            reasons.append(reason)
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return reasons

def _select_profile_blend_leader(profile_blend: pd.DataFrame, role: str) -> dict[str, Any]:
    if profile_blend.empty:
        return {}
    if role == "tail_lift":
        eligible_column = "tail_lift_eligible"
        sort_columns = ["top_10_lift_global", "mean_rank_ic", "worst_5_rank_ic_mean"]
        ascending = [False, False, False]
    elif role == "stability":
        eligible_column = "stability_eligible"
        sort_columns = ["mean_rank_ic", "worst_5_rank_ic_mean", "std_rank_ic", "positive_ic_fraction"]
        ascending = [False, False, True, False]
    elif role == "balanced":
        eligible_column = "balanced_eligible"
        sort_columns = [
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "top_10_lift_global",
            "worst_5_rank_ic_mean",
        ]
        ascending = [False, True, False, False, False]
    else:
        raise ValueError(f"Unknown profile blend leader role: {role}")
    if eligible_column not in profile_blend.columns:
        return {}
    candidates = profile_blend[profile_blend[eligible_column].astype(bool)].copy()
    if candidates.empty:
        return {}
    candidates = candidates.sort_values(sort_columns, ascending=ascending)
    return candidates.iloc[0].to_dict()

def _profile_blend_leaders(profile_blend: pd.DataFrame) -> dict[str, Any]:
    leaders = {
        "balanced_leader": _select_profile_blend_leader(profile_blend, "balanced"),
        "tail_lift_leader": _select_profile_blend_leader(profile_blend, "tail_lift"),
        "stability_leader": _select_profile_blend_leader(profile_blend, "stability"),
    }
    return {key: value for key, value in leaders.items() if value}

def _mark_profile_blend_leaders(profile_blend: pd.DataFrame) -> pd.DataFrame:
    if profile_blend.empty:
        return profile_blend
    marked = profile_blend.copy()
    marked["balanced_leader"] = False
    marked["tail_lift_leader"] = False
    marked["stability_leader"] = False
    leaders = _profile_blend_leaders(marked)
    for role, leader in leaders.items():
        profile = str(leader.get("profile", ""))
        if not profile:
            continue
        marked.loc[marked["profile"] == profile, role] = True
    roles = []
    for _, row in marked.iterrows():
        item_roles = []
        if bool(row.get("balanced_leader", False)):
            item_roles.append("balanced")
        if bool(row.get("tail_lift_leader", False)):
            item_roles.append("tail_lift")
        if bool(row.get("stability_leader", False)):
            item_roles.append("stability")
        roles.append(",".join(item_roles))
    marked["leader_roles"] = roles
    return marked

def _best_profile_blend(profile_blend: pd.DataFrame) -> dict[str, Any]:
    leaders = _profile_blend_leaders(profile_blend)
    return leaders.get("balanced_leader") or leaders.get("tail_lift_leader") or leaders.get("stability_leader") or {}

def _profile_blend_markdown(profile_blend: pd.DataFrame) -> str:
    lines = ["# Profile Blend Diagnostics", ""]
    if profile_blend.empty:
        lines.append("No full-profile blends were produced.")
        return "\n".join(lines)
    display_cols = [
        "profile",
        "blend_method",
        "blend_weights",
        "profile_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "mean_rank_ic_delta_vs_control",
        "std_rank_ic_delta_vs_control",
        "positive_ic_fraction_delta_vs_control",
        "top_10_lift_global",
        "top_10_lift_global_delta_vs_control",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "reviewable",
        "review_reason",
        "balanced_eligible",
        "balanced_reason",
        "tail_lift_eligible",
        "tail_lift_reason",
        "stability_eligible",
        "stability_reason",
        "leader_roles",
        "prob_long_profile_std_mean",
        "prob_long_profile_std_p90",
        "blend_profiles",
    ]
    visible = profile_blend[[column for column in display_cols if column in profile_blend.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_profile_blend_files(path: Path, profile_blend: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    profile_blend.to_csv(path / "profile_blend.csv", index=False)
    (path / "profile_blend.md").write_text(_profile_blend_markdown(profile_blend), encoding="utf-8")
    _write_json(path / "profile_blend.json", {"rows": profile_blend.to_dict(orient="records")})

def _write_profile_diagnostic_summaries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)

    def tagged_frame(entry: dict[str, Any], key: str) -> pd.DataFrame:
        frame = entry["diagnostics"].get(key)
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        out.insert(0, "fold_scope", str(entry["fold_scope"]))
        out.insert(0, "profile", str(entry["profile"]))
        return out

    for key, filename in [
        ("fold_metrics", "profile_fold_metrics.csv"),
        ("threshold_summary", "profile_threshold_summary.csv"),
        ("calibrated_threshold_summary", "profile_calibrated_threshold_summary.csv"),
        ("threshold_grid_summary", "profile_threshold_grid_summary.csv"),
        ("score_band_summary", "profile_score_band_summary.csv"),
        ("score_policy_grid", "profile_score_policy_grid.csv"),
        ("score_policy_selection", "profile_score_policy_selection.csv"),
        ("feature_groups", "profile_feature_groups.csv"),
    ]:
        frames = [tagged_frame(entry, key) for entry in entries]
        frames = [frame for frame in frames if not frame.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(path / filename, index=False)
