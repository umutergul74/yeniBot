"""Content-addressed frozen candidate manifests for future-OOS evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.experiment.common import _cfg, _hash_payload, _table_markdown, _write_json

__all__ = ["freeze_candidate_manifests", "verify_frozen_manifest_artifacts"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _entry_by_profile(entries: list[dict[str, Any]], *, fold_scope: str) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("profile")): entry
        for entry in entries
        if str(entry.get("fold_scope")) == fold_scope
    }


def _artifact_records(scope_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    paths = [
        scope_dir / "training_manifest.json",
        *sorted(scope_dir.glob("model_fold_*.pt")),
        *sorted(scope_dir.glob("scaler_fold_*.pkl")),
        *sorted(scope_dir.glob("hmm_fold_*.pkl")),
    ]
    records = []
    for path in paths:
        if not path.exists():
            continue
        records.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return records


def _threshold_payload(entry: dict[str, Any]) -> dict[str, Any]:
    row = (entry.get("diagnostics", {}) or {}).get("row", {}) or {}
    for value_key, source_key in (
        ("official_threshold_mean", "official_threshold_source"),
        ("guarded_threshold_mean", "guarded_threshold_source"),
        ("constrained_threshold_mean", None),
        ("selected_threshold_mean", None),
    ):
        value = _finite_float(row.get(value_key))
        if value is not None:
            return {
                "value": value,
                "source": str(row.get(source_key, value_key) if source_key else value_key),
                "selected_from": "pre_anchor_walk_forward_validation",
            }
    return {
        "value": 0.5,
        "source": "fallback_0.50_missing_cv_threshold",
        "selected_from": "fallback_not_tuned_on_future_oos",
    }


def _profile_component(
    *,
    profile: str,
    fold_scope: str,
    entries: dict[str, dict[str, Any]],
    run_dir: Path,
    anchor: pd.Timestamp,
) -> tuple[dict[str, Any], str]:
    entry = entries.get(profile)
    if entry is None:
        return {}, f"missing_profile_scope:{profile}:{fold_scope}"
    scope_dir = Path(entry["scope_dir"])
    manifest_path = scope_dir / "training_manifest.json"
    if not manifest_path.exists():
        return {}, f"missing_training_manifest:{profile}:{fold_scope}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit_end = pd.to_datetime(manifest.get("data_end"), utc=True, errors="coerce")
    if pd.isna(fit_end):
        return {}, f"missing_fit_data_end:{profile}:{fold_scope}"
    if fit_end > anchor:
        return {}, f"fit_data_end_after_anchor:{profile}:{fit_end.isoformat()}"
    artifacts = _artifact_records(scope_dir, run_dir)
    model_count = sum(item["relative_path"].endswith(".pt") for item in artifacts)
    scaler_count = sum("scaler_fold_" in item["relative_path"] for item in artifacts)
    hmm_count = sum("hmm_fold_" in item["relative_path"] for item in artifacts)
    if not model_count or model_count != scaler_count or model_count != hmm_count:
        return {}, f"incomplete_artifact_triplets:{profile}:{model_count}:{scaler_count}:{hmm_count}"
    return (
        {
            "profile": profile,
            "fold_scope": fold_scope,
            "scope_relative_path": scope_dir.relative_to(run_dir).as_posix(),
            "fit_data_start": str(manifest.get("data_start", "")),
            "fit_data_end": fit_end.isoformat(),
            "feature_columns": list(manifest.get("feature_columns", [])),
            "feature_columns_hash": str(manifest.get("feature_columns_hash", "")),
            "training_signature_hash": str(manifest.get("signature_hash", "")),
            "model_count": model_count,
            "artifacts": artifacts,
        },
        "",
    )


def freeze_candidate_manifests(
    *,
    run_dir: str | Path,
    report_dir: str | Path,
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Freeze pre-anchor profile/blend definitions and artifact hashes."""

    run_path = Path(run_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    frozen_cfg = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    enabled = bool(frozen_cfg.get("enabled", False))
    anchor_value = frozen_cfg.get("anchor_data_end")
    if not enabled or not anchor_value:
        index = pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_type",
                "candidate_status",
                "available",
                "profile_count",
                "model_count",
                "threshold",
                "threshold_source",
                "anchor_data_end",
                "manifest_hash",
                "unavailable_reasons",
            ]
        )
        index.to_csv(report_path / "frozen_candidate_index.csv", index=False)
        (report_path / "frozen_candidate_index.md").write_text(
            _table_markdown("Frozen Candidate Index", index),
            encoding="utf-8",
        )
        disabled = {
            "available": False,
            "enabled": enabled,
            "unavailable_reasons": ["frozen_candidate_protocol_disabled_or_missing_anchor"],
        }
        _write_json(report_path / "frozen_candidate_manifests.json", {"candidates": []})
        _write_json(report_path / "frozen_candidate_manifest.json", disabled)
        return [], index
    anchor = pd.to_datetime(anchor_value, utc=True, errors="raise")
    profile_entries = _entry_by_profile(entries, fold_scope="full")
    all_entries = {str(entry.get("profile")): entry for entry in entries}
    manifests: list[dict[str, Any]] = []

    for spec in frozen_cfg.get("candidates", []) or []:
        if not isinstance(spec, dict):
            continue
        candidate_id = str(spec.get("candidate_id", "")).strip()
        candidate_type = str(spec.get("candidate_type", "profile"))
        fold_scope = str(spec.get("fold_scope", "full"))
        components: list[dict[str, Any]] = []
        errors: list[str] = []
        profiles = (
            [str(spec.get("profile", ""))]
            if candidate_type == "profile"
            else [str(item) for item in spec.get("profiles", []) or []]
        )
        for profile in profiles:
            component, error = _profile_component(
                profile=profile,
                fold_scope=fold_scope,
                entries=profile_entries,
                run_dir=run_path,
                anchor=anchor,
            )
            if error:
                errors.append(error)
            else:
                components.append(component)

        diagnostic_entry = (
            profile_entries.get(str(spec.get("profile")))
            if candidate_type == "profile"
            else all_entries.get(
                f"blend_{str(spec.get('candidate_id', '')).removesuffix('_v1')}"
            )
        )
        if diagnostic_entry is None and candidate_type == "blend":
            wanted_profiles = profiles
            wanted_weights = [float(item) for item in spec.get("weights", []) or []]
            for entry in entries:
                row = (entry.get("diagnostics", {}) or {}).get("row", {}) or {}
                entry_profiles = [item for item in str(row.get("blend_profiles", "")).split(",") if item]
                entry_weights = [
                    float(item) for item in str(row.get("blend_weights", "")).split(",") if item
                ]
                if entry_profiles == wanted_profiles and entry_weights == wanted_weights:
                    diagnostic_entry = entry
                    break

        content = {
            "protocol_version": str(frozen_cfg.get("protocol_version", "v1")),
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "candidate_status": str(spec.get("status", "preregistered")),
            "source_run_id": run_path.name,
            "anchor_run_id": str(frozen_cfg.get("anchor_run_id", "")),
            "anchor_data_end": anchor.isoformat(),
            "artifact_policy": str(frozen_cfg.get("artifact_policy", "")),
            "profiles": profiles,
            "blend_method": str(spec.get("blend_method", "")),
            "weights": [float(item) for item in spec.get("weights", []) or []],
            "threshold": _threshold_payload(diagnostic_entry or {}),
            "components": components,
            "available": enabled and not errors and len(components) == len(profiles),
            "unavailable_reasons": errors,
            "future_oos_fit_allowed": False,
        }
        manifest_hash = _hash_payload(content)
        manifest = {
            **content,
            "manifest_hash": manifest_hash,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }
        manifests.append(manifest)
        immutable_path = (
            run_path
            / "frozen_candidates"
            / candidate_id
            / f"manifest_{manifest_hash}.json"
        )
        if not immutable_path.exists():
            _write_json(immutable_path, manifest)

    index_rows = [
        {
            "candidate_id": item["candidate_id"],
            "candidate_type": item["candidate_type"],
            "candidate_status": item["candidate_status"],
            "available": item["available"],
            "profile_count": len(item["profiles"]),
            "model_count": sum(int(component.get("model_count", 0)) for component in item["components"]),
            "threshold": item["threshold"]["value"],
            "threshold_source": item["threshold"]["source"],
            "anchor_data_end": item["anchor_data_end"],
            "manifest_hash": item["manifest_hash"],
            "unavailable_reasons": ";".join(item["unavailable_reasons"]),
        }
        for item in manifests
    ]
    index = pd.DataFrame(index_rows)
    index.to_csv(report_path / "frozen_candidate_index.csv", index=False)
    (report_path / "frozen_candidate_index.md").write_text(
        _table_markdown("Frozen Candidate Index", index),
        encoding="utf-8",
    )
    _write_json(report_path / "frozen_candidate_manifests.json", {"candidates": manifests})
    primary_id = str(frozen_cfg.get("primary_candidate_id", ""))
    primary = next((item for item in manifests if item["candidate_id"] == primary_id), {})
    _write_json(report_path / "frozen_candidate_manifest.json", primary)
    return manifests, index


def verify_frozen_manifest_artifacts(
    manifest: dict[str, Any],
    *,
    run_dir: str | Path,
) -> list[str]:
    """Return artifact-integrity errors; never silently accept tampering."""

    root = Path(run_dir)
    errors: list[str] = []
    for component in manifest.get("components", []) or []:
        for artifact in component.get("artifacts", []) or []:
            path = root / str(artifact["relative_path"])
            if not path.exists():
                errors.append(f"missing:{artifact['relative_path']}")
                continue
            if int(path.stat().st_size) != int(artifact["size_bytes"]):
                errors.append(f"size_mismatch:{artifact['relative_path']}")
                continue
            if _sha256_file(path) != str(artifact["sha256"]):
                errors.append(f"sha256_mismatch:{artifact['relative_path']}")
    return errors
