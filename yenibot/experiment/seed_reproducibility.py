"""Seed-audit reproducibility and persisted coverage reconciliation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.experiment.common import _hash_payload, _json_ready, _read_json, _write_json
from yenibot.experiment.ensembles import _seed_from_scope

__all__ = [
    "_seed_reproducibility_reports",
    "_reconcile_seed_extension_summary",
    "_seed_reproducibility_audit_frame",
    "_seed_reproducibility_manifest_diff_frame",
    "_write_seed_reproducibility_files",
]


def _seed_reproducibility_reports(
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
    config: dict[str, Any],
    extension_summary: dict[str, Any] | None,
    coverage: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    return (
        _reconcile_seed_extension_summary(extension_summary, coverage),
        _seed_reproducibility_audit_frame(entries, settings, config),
    )


def _manifest(entry: dict[str, Any]) -> dict[str, Any]:
    scope_dir = entry.get("scope_dir")
    path = Path(scope_dir) / "training_manifest.json" if scope_dir else None
    if path is None or not path.exists():
        return {}
    return _read_json(path)


def _test_predictions(entry: dict[str, Any]) -> pd.DataFrame:
    predictions = entry.get("predictions")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _prediction_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    required = ["fold", "timestamp"]
    if not set(required).issubset(left.columns) or not set(required).issubset(right.columns):
        return []
    keys = ["fold", "timestamp"]
    for column in ("source_row_position",):
        if column in left.columns and column in right.columns:
            keys.append(column)
    return keys


def _return_column(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    for column in ("fwd_return_10h", "forward_return", "tb_return"):
        if column in left.columns and column in right.columns:
            return column
    return None


def _match_fraction(left: pd.Series, right: pd.Series, *, numeric: bool = False) -> float:
    if left.empty:
        return np.nan
    if numeric:
        left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
        return float(np.isclose(left_values, right_values, equal_nan=True).mean())
    left_values = left.astype(object).where(left.notna(), "__nan__")
    right_values = right.astype(object).where(right.notna(), "__nan__")
    return float((left_values == right_values).mean())


def _fold_rank_ic(frame: pd.DataFrame, score_column: str, return_column: str) -> pd.Series:
    def rank_ic(group: pd.DataFrame) -> float:
        clean = group[[score_column, return_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3 or clean[score_column].nunique() < 2 or clean[return_column].nunique() < 2:
            return np.nan
        return float(clean[score_column].corr(clean[return_column], method="spearman"))

    return frame.groupby("fold", sort=True).apply(rank_ic, include_groups=False)


def _input_fingerprint(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    return_column: str | None,
    suffix: str,
) -> str:
    columns = [column for column in keys if column in frame.columns]
    for base in ("label", return_column):
        if not base:
            continue
        column = f"{base}_{suffix}"
        if column in frame.columns and column not in columns:
            columns.append(column)
    if not columns:
        return ""
    selected = frame[columns].copy()
    sort_columns = [column for column in keys if column in selected.columns]
    if sort_columns:
        selected = selected.sort_values(sort_columns).reset_index(drop=True)
    if "timestamp" in selected.columns:
        selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True).astype(str)
    return _hash_payload(selected.to_dict(orient="list"))


def _comparison_status(
    *,
    same_seed: bool,
    frame_match: bool,
    feature_match: bool,
    config_match: bool,
    overlap_input_match: bool,
    aligned_rows: int,
    score_spearman: float,
    score_allclose: float,
    mean_rank_ic_delta: float,
) -> tuple[str, str, str]:
    if not feature_match or (not frame_match and not overlap_input_match):
        return (
            "invalid_data_or_feature_signature_mismatch",
            "source_and_audit_training_inputs_differ",
            "Do not interpret the result as seed sensitivity; first restore identical feature signatures and overlapping input rows.",
        )
    if not aligned_rows:
        return (
            "invalid_no_aligned_predictions",
            "prediction_keys_do_not_align",
            "Verify fold ids, timestamps, and source-row positions before comparing runs.",
        )
    if not same_seed:
        return (
            "independent_seed_reference",
            "independent_initialization",
            "Use this row only for seed-dispersion evidence, not exact reproducibility.",
        )
    if not config_match:
        return (
            "invalid_same_seed_training_config_mismatch",
            "same_seed_but_training_configuration_differs",
            "Compare the training config payloads before attributing the difference to runtime nondeterminism.",
        )
    if score_allclose >= 0.999 and abs(mean_rank_ic_delta) <= 1e-6:
        return (
            "same_seed_reproduced",
            "functionally_identical_predictions",
            "The same-seed scope is reproducible at the configured numeric tolerance.",
        )
    if score_spearman >= 0.999 and abs(mean_rank_ic_delta) <= 0.002:
        return (
            "same_seed_ranking_reproduced_with_numeric_drift",
            "minor_numeric_runtime_drift",
            "Record runtime versions, but ranking behavior is functionally reproduced.",
        )
    return (
        "same_seed_not_reproduced_environment_audit_required",
        "runtime_or_kernel_nondeterminism_or_unrecorded_code_drift",
        "Overlapping source and audit inputs align, but predictions do not reproduce. Record torch/CUDA/cuDNN versions and deterministic settings, then rerun only the same-seed audit if needed.",
    )


def _seed_reproducibility_audit_frame(
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "profile",
        "source_fold_scope",
        "audit_fold_scope",
        "audit_seed",
        "expected_source_seed",
        "comparison_role",
        "manifest_available",
        "frame_fingerprint_match",
        "overlap_input_fingerprint_match",
        "feature_columns_hash_match",
        "training_config_hash_match",
        "manifest_compatible",
        "source_frame_fingerprint",
        "audit_frame_fingerprint",
        "source_data_start",
        "source_data_end",
        "audit_data_start",
        "audit_data_end",
        "source_signature_hash",
        "audit_signature_hash",
        "overlap_fold_count",
        "overlap_fold_ids",
        "aligned_prediction_rows",
        "source_only_rows",
        "audit_only_rows",
        "label_match_fraction",
        "return_match_fraction",
        "probability_pearson",
        "probability_spearman",
        "probability_mae",
        "probability_max_abs_diff",
        "probability_allclose_fraction",
        "source_overlap_mean_rank_ic",
        "audit_overlap_mean_rank_ic",
        "mean_rank_ic_delta",
        "fold_rank_ic_delta_std",
        "fold_rank_ic_sign_agreement",
        "reproducibility_status",
        "likely_cause",
        "recommended_action",
    ]
    expected_seed = int(config.get("project", {}).get("random_seed", 42))
    source_entries = {
        str(entry.get("profile", "")): entry
        for entry in entries
        if str(entry.get("fold_scope", "")) == "full"
    }
    rows: list[dict[str, Any]] = []
    for audit in entries:
        audit_scope = str(audit.get("fold_scope", ""))
        audit_seed = _seed_from_scope(audit_scope)
        if audit_seed is None:
            continue
        profile = str(audit.get("profile", ""))
        source = source_entries.get(profile)
        if source is None:
            continue
        source_manifest = _manifest(source)
        audit_manifest = _manifest(audit)
        source_predictions = _test_predictions(source)
        audit_predictions = _test_predictions(audit)
        keys = _prediction_keys(source_predictions, audit_predictions)
        overlap_folds = sorted(
            set(pd.to_numeric(source_predictions.get("fold"), errors="coerce").dropna().astype(int))
            & set(pd.to_numeric(audit_predictions.get("fold"), errors="coerce").dropna().astype(int))
        )
        source_overlap = source_predictions.loc[
            pd.to_numeric(source_predictions.get("fold"), errors="coerce").isin(overlap_folds)
        ].copy()
        audit_overlap = audit_predictions.loc[
            pd.to_numeric(audit_predictions.get("fold"), errors="coerce").isin(overlap_folds)
        ].copy()
        return_column = _return_column(source_overlap, audit_overlap)
        selected = [*keys, "prob_long"]
        for column in ("label", return_column):
            if column and column not in selected:
                selected.append(column)
        if keys:
            merged = source_overlap[selected].merge(
                audit_overlap[selected],
                on=keys,
                how="outer",
                suffixes=("_source", "_audit"),
                indicator=True,
            )
        else:
            merged = pd.DataFrame()
        aligned = merged.loc[merged.get("_merge", pd.Series(dtype=str)).eq("both")].copy()
        source_input_fingerprint = _input_fingerprint(
            aligned,
            keys=keys,
            return_column=return_column,
            suffix="source",
        )
        audit_input_fingerprint = _input_fingerprint(
            aligned,
            keys=keys,
            return_column=return_column,
            suffix="audit",
        )
        overlap_input_match = bool(
            source_input_fingerprint
            and source_input_fingerprint == audit_input_fingerprint
        )
        source_scores = pd.to_numeric(aligned.get("prob_long_source"), errors="coerce")
        audit_scores = pd.to_numeric(aligned.get("prob_long_audit"), errors="coerce")
        valid_scores = source_scores.notna() & audit_scores.notna()
        source_scores = source_scores.loc[valid_scores]
        audit_scores = audit_scores.loc[valid_scores]
        probability_pearson = (
            float(source_scores.corr(audit_scores, method="pearson"))
            if len(source_scores) >= 2 and source_scores.nunique() > 1 and audit_scores.nunique() > 1
            else np.nan
        )
        probability_spearman = (
            float(source_scores.corr(audit_scores, method="spearman"))
            if len(source_scores) >= 2 and source_scores.nunique() > 1 and audit_scores.nunique() > 1
            else np.nan
        )
        differences = (audit_scores - source_scores).abs()
        if return_column and not aligned.empty:
            rank_frame = aligned[
                ["fold", "prob_long_source", "prob_long_audit", f"{return_column}_source"]
            ].rename(columns={f"{return_column}_source": "_return"})
            source_ic = _fold_rank_ic(rank_frame, "prob_long_source", "_return")
            audit_ic = _fold_rank_ic(rank_frame, "prob_long_audit", "_return")
            fold_ic = pd.concat([source_ic.rename("source"), audit_ic.rename("audit")], axis=1).dropna()
        else:
            fold_ic = pd.DataFrame(columns=["source", "audit"])
        fold_delta = fold_ic["audit"] - fold_ic["source"] if not fold_ic.empty else pd.Series(dtype=float)
        source_mean_ic = float(fold_ic["source"].mean()) if not fold_ic.empty else np.nan
        audit_mean_ic = float(fold_ic["audit"].mean()) if not fold_ic.empty else np.nan
        mean_delta = audit_mean_ic - source_mean_ic if np.isfinite(source_mean_ic) and np.isfinite(audit_mean_ic) else np.nan
        frame_match = bool(
            source_manifest
            and audit_manifest
            and source_manifest.get("frame_fingerprint") == audit_manifest.get("frame_fingerprint")
        )
        feature_match = bool(
            source_manifest
            and audit_manifest
            and source_manifest.get("feature_columns_hash") == audit_manifest.get("feature_columns_hash")
        )
        config_match = bool(
            source_manifest
            and audit_manifest
            and source_manifest.get("training_config_hash") == audit_manifest.get("training_config_hash")
        )
        same_seed = audit_seed == expected_seed
        status, cause, action = _comparison_status(
            same_seed=same_seed,
            frame_match=frame_match,
            feature_match=feature_match,
            config_match=config_match,
            overlap_input_match=overlap_input_match,
            aligned_rows=len(aligned),
            score_spearman=probability_spearman if np.isfinite(probability_spearman) else -1.0,
            score_allclose=float(np.isclose(source_scores, audit_scores, atol=1e-6, rtol=1e-5).mean())
            if len(source_scores)
            else 0.0,
            mean_rank_ic_delta=mean_delta if np.isfinite(mean_delta) else np.inf,
        )
        rows.append(
            {
                "profile": profile,
                "source_fold_scope": str(source.get("fold_scope", "")),
                "audit_fold_scope": audit_scope,
                "audit_seed": audit_seed,
                "expected_source_seed": expected_seed,
                "comparison_role": "same_seed_reproduction" if same_seed else "independent_seed",
                "manifest_available": bool(source_manifest and audit_manifest),
                "frame_fingerprint_match": frame_match,
                "overlap_input_fingerprint_match": overlap_input_match,
                "feature_columns_hash_match": feature_match,
                "training_config_hash_match": config_match,
                "manifest_compatible": bool(
                    feature_match
                    and (frame_match or overlap_input_match)
                    and (config_match or not same_seed)
                ),
                "source_frame_fingerprint": str(source_manifest.get("frame_fingerprint", "")),
                "audit_frame_fingerprint": str(audit_manifest.get("frame_fingerprint", "")),
                "source_data_start": str(source_manifest.get("data_start", "")),
                "source_data_end": str(source_manifest.get("data_end", "")),
                "audit_data_start": str(audit_manifest.get("data_start", "")),
                "audit_data_end": str(audit_manifest.get("data_end", "")),
                "source_signature_hash": str(source_manifest.get("signature_hash", "")),
                "audit_signature_hash": str(audit_manifest.get("signature_hash", "")),
                "overlap_fold_count": len(overlap_folds),
                "overlap_fold_ids": ",".join(str(value) for value in overlap_folds),
                "aligned_prediction_rows": len(aligned),
                "source_only_rows": int(merged["_merge"].eq("left_only").sum()) if not merged.empty else 0,
                "audit_only_rows": int(merged["_merge"].eq("right_only").sum()) if not merged.empty else 0,
                "label_match_fraction": _match_fraction(
                    aligned.get("label_source", pd.Series(dtype=float)),
                    aligned.get("label_audit", pd.Series(dtype=float)),
                ),
                "return_match_fraction": _match_fraction(
                    aligned.get(f"{return_column}_source", pd.Series(dtype=float)),
                    aligned.get(f"{return_column}_audit", pd.Series(dtype=float)),
                    numeric=True,
                )
                if return_column
                else np.nan,
                "probability_pearson": probability_pearson,
                "probability_spearman": probability_spearman,
                "probability_mae": float(differences.mean()) if len(differences) else np.nan,
                "probability_max_abs_diff": float(differences.max()) if len(differences) else np.nan,
                "probability_allclose_fraction": float(
                    np.isclose(source_scores, audit_scores, atol=1e-6, rtol=1e-5).mean()
                )
                if len(source_scores)
                else np.nan,
                "source_overlap_mean_rank_ic": source_mean_ic,
                "audit_overlap_mean_rank_ic": audit_mean_ic,
                "mean_rank_ic_delta": mean_delta,
                "fold_rank_ic_delta_std": float(fold_delta.std(ddof=1)) if len(fold_delta) > 1 else np.nan,
                "fold_rank_ic_sign_agreement": float(
                    (np.sign(fold_ic["source"]) == np.sign(fold_ic["audit"])).mean()
                )
                if not fold_ic.empty
                else np.nan,
                "reproducibility_status": status,
                "likely_cause": cause,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _seed_reproducibility_manifest_diff_frame(audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "profile",
        "audit_fold_scope",
        "audit_seed",
        "comparison_role",
        "frame_fingerprint_match",
        "overlap_input_fingerprint_match",
        "feature_columns_hash_match",
        "training_config_hash_match",
        "manifest_compatible",
        "global_frame_mismatch_but_overlap_inputs_match",
        "same_seed_reproducibility_interpretable",
        "source_data_start",
        "source_data_end",
        "audit_data_start",
        "audit_data_end",
        "source_frame_fingerprint",
        "audit_frame_fingerprint",
        "source_signature_hash",
        "audit_signature_hash",
        "likely_cause",
        "recommended_action",
    ]
    if audit.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in audit.iterrows():
        global_mismatch_overlap_match = bool(
            not bool(row.get("frame_fingerprint_match", False))
            and bool(row.get("overlap_input_fingerprint_match", False))
        )
        same_seed = str(row.get("comparison_role", "")) == "same_seed_reproduction"
        interpretable = bool(
            same_seed
            and bool(row.get("feature_columns_hash_match", False))
            and bool(row.get("training_config_hash_match", False))
            and (
                bool(row.get("frame_fingerprint_match", False))
                or bool(row.get("overlap_input_fingerprint_match", False))
            )
        )
        rows.append(
            {
                "profile": row.get("profile", ""),
                "audit_fold_scope": row.get("audit_fold_scope", ""),
                "audit_seed": row.get("audit_seed", ""),
                "comparison_role": row.get("comparison_role", ""),
                "frame_fingerprint_match": row.get("frame_fingerprint_match", False),
                "overlap_input_fingerprint_match": row.get("overlap_input_fingerprint_match", False),
                "feature_columns_hash_match": row.get("feature_columns_hash_match", False),
                "training_config_hash_match": row.get("training_config_hash_match", False),
                "manifest_compatible": row.get("manifest_compatible", False),
                "global_frame_mismatch_but_overlap_inputs_match": global_mismatch_overlap_match,
                "same_seed_reproducibility_interpretable": interpretable,
                "source_data_start": row.get("source_data_start", ""),
                "source_data_end": row.get("source_data_end", ""),
                "audit_data_start": row.get("audit_data_start", ""),
                "audit_data_end": row.get("audit_data_end", ""),
                "source_frame_fingerprint": row.get("source_frame_fingerprint", ""),
                "audit_frame_fingerprint": row.get("audit_frame_fingerprint", ""),
                "source_signature_hash": row.get("source_signature_hash", ""),
                "audit_signature_hash": row.get("audit_signature_hash", ""),
                "likely_cause": row.get("likely_cause", ""),
                "recommended_action": row.get("recommended_action", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _reconcile_seed_extension_summary(
    summary: dict[str, Any] | None,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    if not summary:
        return {}
    reconciled = dict(summary)
    effective = bool(
        not coverage.empty
        and "coverage_passed" in coverage.columns
        and coverage["coverage_passed"].astype(bool).all()
    )
    original = bool(summary.get("coverage_passed", False))
    reconciled["coverage_passed_at_extension"] = original
    reconciled["coverage_passed"] = effective
    reconciled["coverage_reconciled_at_diagnostics"] = original != effective
    reconciled["coverage_rows_at_diagnostics"] = int(len(coverage))
    reconciled["coverage_statuses_at_diagnostics"] = (
        sorted(set(coverage["status"].astype(str))) if "status" in coverage.columns else []
    )
    return reconciled


def _seed_reproducibility_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Seed Reproducibility Audit",
        "",
        "The same-seed row tests exact experiment reproducibility. Other seed rows measure initialization sensitivity.",
        "",
    ]
    if frame.empty:
        lines.append("No full-control and seed-audit scopes were available for comparison.")
        return "\n".join(lines)
    visible_columns = [
        "profile",
        "audit_seed",
        "comparison_role",
        "manifest_compatible",
        "overlap_input_fingerprint_match",
        "overlap_fold_count",
        "probability_spearman",
        "mean_rank_ic_delta",
        "fold_rank_ic_sign_agreement",
        "reproducibility_status",
    ]
    visible = frame[[column for column in visible_columns if column in frame.columns]]
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _seed_reproducibility_manifest_diff_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Seed Reproducibility Manifest Diff",
        "",
        "This report separates global frame hash drift from the actual overlapping fold inputs used for seed comparison.",
        "",
    ]
    if frame.empty:
        lines.append("No manifest-diff rows were produced.")
        return "\n".join(lines)
    visible_columns = [
        "profile",
        "audit_seed",
        "comparison_role",
        "frame_fingerprint_match",
        "overlap_input_fingerprint_match",
        "feature_columns_hash_match",
        "training_config_hash_match",
        "same_seed_reproducibility_interpretable",
        "global_frame_mismatch_but_overlap_inputs_match",
        "likely_cause",
    ]
    visible = frame[[column for column in visible_columns if column in frame.columns]]
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_seed_reproducibility_files(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest_diff = _seed_reproducibility_manifest_diff_frame(frame)
    same_seed_rows = (
        frame.loc[frame["comparison_role"].eq("same_seed_reproduction")]
        if not frame.empty and "comparison_role" in frame.columns
        else pd.DataFrame(columns=frame.columns)
    )
    frame.to_csv(path / "seed_reproducibility_audit.csv", index=False)
    manifest_diff.to_csv(path / "seed_reproducibility_manifest_diff.csv", index=False)
    (path / "seed_reproducibility_audit.md").write_text(
        _seed_reproducibility_markdown(frame),
        encoding="utf-8",
    )
    (path / "seed_reproducibility_manifest_diff.md").write_text(
        _seed_reproducibility_manifest_diff_markdown(manifest_diff),
        encoding="utf-8",
    )
    _write_json(
        path / "seed_reproducibility_audit.json",
        {
            "same_seed_reproduced": bool(
                not same_seed_rows.empty
                and same_seed_rows["reproducibility_status"].isin(
                    {
                        "same_seed_reproduced",
                        "same_seed_ranking_reproduced_with_numeric_drift",
                    }
                ).all()
            ),
            "rows": frame.to_dict(orient="records"),
        },
    )
    _write_json(
        path / "seed_reproducibility_manifest_diff.json",
        {"rows": _json_ready(manifest_diff.to_dict(orient="records"))},
    )
