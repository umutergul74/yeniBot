"""Content-addressed frozen candidate manifests for future-OOS evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.experiment.common import _cfg, _hash_payload, _slug, _table_markdown, _write_json

__all__ = [
    "freeze_candidate_manifests",
    "frozen_manifest_content_hash",
    "frozen_manifest_source_run_dir",
    "verify_frozen_manifest_artifacts",
]


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


def frozen_manifest_content_hash(manifest: dict[str, Any]) -> str:
    """Recompute the immutable content hash of a frozen candidate manifest."""

    excluded = {
        "manifest_hash",
        "expected_manifest_hash",
        "manifest_hash_verified",
        "frozen_at",
    }
    content = {key: value for key, value in manifest.items() if key not in excluded}
    return _hash_payload(content)


def _entry_by_profile(entries: list[dict[str, Any]], *, fold_scope: str) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("profile")): entry
        for entry in entries
        if str(entry.get("fold_scope")) == fold_scope
    }

def frozen_manifest_source_run_dir(
    manifest: dict[str, Any],
    *,
    run_dir: str | Path,
) -> Path:
    """Resolve the immutable artifact root pinned by a frozen manifest."""

    current = Path(run_dir)
    source_run_id = str(manifest.get("source_run_id", "") or current.name)
    if Path(source_run_id).name != source_run_id or source_run_id in {"", ".", ".."}:
        raise ValueError(f"Invalid frozen source_run_id: {source_run_id!r}")
    return current if current.name == source_run_id else current.parent / source_run_id


def _artifact_records(scope_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    paths = [
        scope_dir / "training_manifest.json",
        scope_dir / "replacement_candidate_fit.json",
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

def _configured_threshold_payload(spec: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    configured = spec.get("threshold")
    if isinstance(configured, dict):
        value = _finite_float(configured.get("value"))
        if value is not None:
            return {
                "value": value,
                "source": str(configured.get("source", "configured_frozen_threshold")),
                "selected_from": str(
                    configured.get("selected_from", "pre_anchor_walk_forward_validation")
                ),
            }
    return _threshold_payload(entry)


def _profile_component(
    *,
    profile: str,
    fold_scope: str,
    run_dir: Path,
    anchor: pd.Timestamp,
) -> tuple[dict[str, Any], str]:
    scope_dir = run_dir / _slug(profile) / fold_scope
    if not scope_dir.exists():
        return {}, f"missing_profile_scope:{profile}:{fold_scope}:{run_dir.name}"
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
    primary_id = str(frozen_cfg.get("primary_candidate_id", ""))
    if not enabled or not anchor_value:
        index = pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_type",
                "candidate_status",
                "evaluation_role",
                "required_for_evaluation",
                "available",
                "profile_count",
                "model_count",
                "threshold",
                "threshold_source",
                "source_run_id",
                "anchor_data_end",
                "manifest_hash",
                "expected_manifest_hash",
                "manifest_hash_verified",
                "unavailable_reasons",
            ]
        )
        index.to_csv(report_path / "frozen_candidate_index.csv", index=False)
        (report_path / "frozen_candidate_index.md").write_text(
            _table_markdown("Frozen Candidate Index", index),
            encoding="utf-8",
        )
        awaiting_replacement = enabled and not anchor_value
        disabled = {
            "available": False,
            "enabled": enabled,
            "candidate_status": (
                "awaiting_replacement_preregistration"
                if awaiting_replacement
                else "frozen_candidate_protocol_disabled"
            ),
            "unavailable_reasons": [
                "replacement_candidate_not_preregistered"
                if awaiting_replacement
                else "frozen_candidate_protocol_disabled"
            ],
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
        source_run_id = str(spec.get("source_run_id", "") or run_path.name)
        if Path(source_run_id).name != source_run_id or source_run_id in {"", ".", ".."}:
            source_run_path = run_path
            source_run_error = f"invalid_source_run_id:{source_run_id}"
        else:
            source_run_path = (
                run_path if run_path.name == source_run_id else run_path.parent / source_run_id
            )
            source_run_error = (
                "" if source_run_path.exists() else f"missing_source_run:{source_run_id}"
            )
        required_for_evaluation = bool(
            spec.get("required_for_evaluation", candidate_id == primary_id)
        )
        components: list[dict[str, Any]] = []
        errors: list[str] = [source_run_error] if source_run_error else []
        profiles = (
            [str(spec.get("profile", ""))]
            if candidate_type in {"profile", "recency_profile"}
            else [str(item) for item in spec.get("profiles", []) or []]
        )
        candidate_anchor = pd.to_datetime(
            spec.get("anchor_data_end", anchor),
            utc=True,
            errors="raise",
        )
        for profile in profiles:
            component, error = _profile_component(
                profile=profile,
                fold_scope=fold_scope,
                run_dir=source_run_path,
                anchor=candidate_anchor,
            )
            if error:
                errors.append(error)
            else:
                components.append(component)

        diagnostic_entry = (
            profile_entries.get(str(spec.get("profile")))
            if candidate_type in {"profile", "recency_profile"}
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

        replacement_metadata: dict[str, Any] = {}
        if candidate_type == "recency_profile":
            replacement_path = (
                source_run_path
                / _slug(str(spec.get("profile", "")))
                / fold_scope
                / "replacement_candidate_fit.json"
            )
            if not replacement_path.exists():
                errors.append(
                    f"missing_replacement_candidate_fit:{candidate_id}:{fold_scope}"
                )
            else:
                replacement_metadata = json.loads(
                    replacement_path.read_text(encoding="utf-8")
                )
                if replacement_metadata.get("candidate_id") != candidate_id:
                    errors.append(
                        "replacement_candidate_id_mismatch:"
                        f"{replacement_metadata.get('candidate_id')}:{candidate_id}"
                    )
                if pd.to_datetime(
                    replacement_metadata.get("anchor_data_end"),
                    utc=True,
                    errors="coerce",
                ) != candidate_anchor:
                    errors.append("replacement_anchor_mismatch")
        configured_threshold = spec.get("threshold")
        if isinstance(configured_threshold, dict):
            threshold_payload = _configured_threshold_payload(
                spec,
                diagnostic_entry or {},
            )
        elif replacement_metadata.get("threshold"):
            threshold_payload = dict(replacement_metadata["threshold"])
        else:
            threshold_payload = _threshold_payload(diagnostic_entry or {})
        content = {
            "protocol_version": str(frozen_cfg.get("protocol_version", "v1")),
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "candidate_status": str(spec.get("status", "preregistered")),
            "evaluation_role": str(
                spec.get(
                    "evaluation_role",
                    "primary" if candidate_id == primary_id else "optional_benchmark",
                )
            ),
            "required_for_evaluation": required_for_evaluation,
            "source_run_id": source_run_id,
            "anchor_run_id": str(frozen_cfg.get("anchor_run_id", "")),
            "anchor_data_end": candidate_anchor.isoformat(),
            "artifact_policy": str(frozen_cfg.get("artifact_policy", "")),
            "profiles": profiles,
            "blend_method": str(spec.get("blend_method", "")),
            "weights": [float(item) for item in spec.get("weights", []) or []],
            "threshold": threshold_payload,
            "recency_policy": str(spec.get("recency_policy", "")),
            "recent_k": int(spec.get("recent_k", 0) or 0),
            "selected_model_folds": list(
                replacement_metadata.get("selected_model_folds", []) or []
            ),
            "selection_evidence_hash": str(
                replacement_metadata.get("selection_evidence_hash", "")
            ),
            "components": components,
            "available": enabled and not errors and len(components) == len(profiles),
            "unavailable_reasons": errors,
            "future_oos_fit_allowed": False,
        }
        manifest_hash = _hash_payload(content)
        expected_manifest_hash = str(spec.get("expected_manifest_hash", "") or "")
        if expected_manifest_hash and manifest_hash != expected_manifest_hash:
            errors.append(
                "expected_manifest_hash_mismatch:"
                f"{expected_manifest_hash}:{manifest_hash}"
            )
            content["available"] = False
            content["unavailable_reasons"] = errors
        manifest = {
            **content,
            "manifest_hash": manifest_hash,
            "expected_manifest_hash": expected_manifest_hash,
            "manifest_hash_verified": bool(
                not expected_manifest_hash or manifest_hash == expected_manifest_hash
            ),
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }
        manifests.append(manifest)
        immutable_path = (
            source_run_path
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
            "evaluation_role": item["evaluation_role"],
            "required_for_evaluation": item["required_for_evaluation"],
            "available": item["available"],
            "profile_count": len(item["profiles"]),
            "model_count": sum(int(component.get("model_count", 0)) for component in item["components"]),
            "threshold": item["threshold"]["value"],
            "threshold_source": item["threshold"]["source"],
            "source_run_id": item["source_run_id"],
            "anchor_data_end": item["anchor_data_end"],
            "manifest_hash": item["manifest_hash"],
            "expected_manifest_hash": item.get("expected_manifest_hash", ""),
            "manifest_hash_verified": item.get("manifest_hash_verified", False),
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
    primary = next((item for item in manifests if item["candidate_id"] == primary_id), {})
    _write_json(report_path / "frozen_candidate_manifest.json", primary)
    return manifests, index


def verify_frozen_manifest_artifacts(
    manifest: dict[str, Any],
    *,
    run_dir: str | Path,
) -> list[str]:
    """Return artifact-integrity errors; never silently accept tampering."""

    try:
        root = frozen_manifest_source_run_dir(manifest, run_dir=run_dir)
    except ValueError as exc:
        return [str(exc)]
    if not root.exists():
        return [f"missing_source_run:{root.name}"]
    errors: list[str] = []
    recorded_manifest_hash = str(manifest.get("manifest_hash", "") or "")
    actual_manifest_hash = frozen_manifest_content_hash(manifest)
    if not recorded_manifest_hash:
        errors.append("missing_manifest_hash")
    elif actual_manifest_hash != recorded_manifest_hash:
        errors.append(
            f"manifest_content_hash_mismatch:{recorded_manifest_hash}:{actual_manifest_hash}"
        )
    expected_manifest_hash = str(manifest.get("expected_manifest_hash", "") or "")
    if expected_manifest_hash and recorded_manifest_hash != expected_manifest_hash:
        errors.append(
            "manifest_hash_mismatch:"
            f"{expected_manifest_hash}:{recorded_manifest_hash}"
        )
    for component in manifest.get("components", []) or []:
        feature_columns = list(component.get("feature_columns", []) or [])
        feature_columns_hash = str(component.get("feature_columns_hash", "") or "")
        if not feature_columns:
            errors.append(f"missing_feature_columns:{component.get('profile', '')}")
        elif not feature_columns_hash:
            errors.append(f"missing_feature_columns_hash:{component.get('profile', '')}")
        elif _hash_payload(feature_columns) != feature_columns_hash:
            errors.append(f"feature_columns_hash_mismatch:{component.get('profile', '')}")
        if not str(component.get("training_signature_hash", "") or ""):
            errors.append(f"missing_training_signature_hash:{component.get('profile', '')}")
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
