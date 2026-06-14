"""Read-only readiness and integrity checks for frozen future-OOS evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.experiment.common import _cfg
from yenibot.experiment.configuration import experiment_root
from yenibot.experiment.frozen import verify_frozen_manifest_artifacts

__all__ = [
    "future_oos_preflight",
    "future_oos_preflight_markdown",
]


def _primary_spec(config: dict[str, Any]) -> dict[str, Any]:
    frozen = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    primary_id = str(frozen.get("primary_candidate_id", ""))
    for item in frozen.get("candidates", []) or []:
        if isinstance(item, dict) and str(item.get("candidate_id", "")) == primary_id:
            return item
    return {}


def _immutable_manifest_path(
    checkpoint_dir: str | Path,
    *,
    source_run_id: str,
    candidate_id: str,
    expected_hash: str,
) -> Path:
    base = (
        experiment_root(checkpoint_dir)
        / source_run_id
        / "frozen_candidates"
        / candidate_id
    )
    if expected_hash:
        return base / f"manifest_{expected_hash}.json"
    candidates = sorted(base.glob("manifest_*.json"))
    return candidates[0] if len(candidates) == 1 else base / "manifest_missing.json"


def _load_primary_manifest(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    manifests: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], Path]:
    frozen = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    primary_id = str(frozen.get("primary_candidate_id", ""))
    if manifests is not None:
        manifest = next(
            (
                item
                for item in manifests
                if str(item.get("candidate_id", "")) == primary_id
            ),
            {},
        )
        spec = _primary_spec(config)
        source_run_id = str(
            manifest.get("source_run_id", "")
            or spec.get("source_run_id", "")
        )
        expected_hash = str(spec.get("expected_manifest_hash", "") or "")
        return manifest, _immutable_manifest_path(
            checkpoint_dir,
            source_run_id=source_run_id,
            candidate_id=primary_id,
            expected_hash=expected_hash,
        )

    spec = _primary_spec(config)
    source_run_id = str(spec.get("source_run_id", ""))
    expected_hash = str(spec.get("expected_manifest_hash", "") or "")
    path = _immutable_manifest_path(
        checkpoint_dir,
        source_run_id=source_run_id,
        candidate_id=primary_id,
        expected_hash=expected_hash,
    )
    if not path.exists():
        return {}, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _check_rows(frame: pd.DataFrame, anchor: pd.Timestamp) -> dict[str, Any]:
    if "timestamp" not in frame.columns:
        timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    else:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    valid = timestamps.dropna()
    future = valid.loc[valid > anchor]
    diffs = valid.sort_values().diff().dropna()
    return {
        "rows": int(len(frame)),
        "data_start": valid.min().isoformat() if not valid.empty else None,
        "data_end": valid.max().isoformat() if not valid.empty else None,
        "fresh_labeled_rows": int(len(future)),
        "future_start": future.min().isoformat() if not future.empty else None,
        "future_end": future.max().isoformat() if not future.empty else None,
        "timestamps_valid": bool(len(valid) == len(frame)),
        "timestamps_unique": bool(valid.is_unique),
        "timestamps_monotonic": bool(valid.is_monotonic_increasing),
        "max_gap_hours": (
            float(diffs.max() / pd.Timedelta(hours=1)) if not diffs.empty else None
        ),
    }


def future_oos_preflight(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    manifests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect future-OOS readiness without writing files or fitting objects."""

    frozen = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    future = _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
    spec = _primary_spec(config)
    primary_id = str(frozen.get("primary_candidate_id", ""))
    source_run_id = str(spec.get("source_run_id", ""))
    expected_hash = str(spec.get("expected_manifest_hash", "") or "")
    expected_threshold = (spec.get("threshold", {}) or {}).get("value")
    anchor_value = frozen.get("anchor_data_end")
    anchor = pd.to_datetime(anchor_value, utc=True, errors="coerce")
    min_rows = int(future.get("min_rows", 720))
    preferred_rows = int(future.get("preferred_rows", 2160))
    if not primary_id or pd.isna(anchor):
        return {
            "preflight_version": "future_oos_preflight_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "side_effect_free": True,
            "fit_operations_performed": 0,
            "state": "awaiting_replacement_preregistration",
            "invariants_passed": False,
            "ready_for_evaluation": False,
            "next_action": "select_and_preregister_replacement_before_new_oos_anchor",
            "required_notebook_sequence_when_refreshing": [],
            "forbidden_before_frozen_evaluation": [
                "future_oos_scoring_without_preregistered_candidate",
                "threshold_selection_from_failed_future_oos",
                "manifest_regeneration_from_retired_candidate",
            ],
            "primary_candidate": {
                "candidate_id": primary_id,
                "source_run_id": source_run_id,
                "anchor_data_end": None,
                "expected_manifest_hash": expected_hash,
                "manifest_path": "",
                "manifest_hash": "",
                "threshold_expected": expected_threshold,
                "threshold_manifest": None,
                "threshold_source": "",
                "profile": str(spec.get("profile", "")),
                "component_count": 0,
                "model_count": 0,
            },
            "data": {
                "fresh_labeled_rows": 0,
                "min_rows": min_rows,
                "preferred_rows": preferred_rows,
                "min_rows_remaining": None,
                "preferred_rows_remaining": None,
            },
            "checks": {
                "protocol_enabled": bool(frozen.get("enabled", False)),
                "active_primary_candidate_present": False,
                "new_anchor_present": False,
            },
            "failed_checks": [
                "active_primary_candidate_present",
                "new_anchor_present",
            ],
            "artifact_integrity_errors": [],
            "missing_frozen_feature_columns": [],
            "warnings": [
                "The prior frozen candidate is retired. This is an expected research state, not an artifact-integrity failure."
            ],
        }
    manifest, manifest_path = _load_primary_manifest(
        checkpoint_dir=checkpoint_dir,
        config=config,
        manifests=manifests,
    )

    data_dir = Path(str(_cfg(config, ["paths", "data_dir"], "data")))
    labeled_path = data_dir / "processed" / "labeled_1h.parquet"
    raw_path = data_dir / "raw" / "btc_1h.parquet"
    labeled = pd.read_parquet(labeled_path) if labeled_path.exists() else pd.DataFrame()
    data = (
        _check_rows(labeled, anchor)
        if not labeled.empty and not pd.isna(anchor)
        else {
            "rows": 0,
            "data_start": None,
            "data_end": None,
            "fresh_labeled_rows": 0,
            "future_start": None,
            "future_end": None,
            "timestamps_valid": False,
            "timestamps_unique": False,
            "timestamps_monotonic": False,
            "max_gap_hours": None,
        }
    )
    data["labeled_path"] = str(labeled_path)
    data["labeled_exists"] = labeled_path.exists()
    data["min_rows"] = min_rows
    data["preferred_rows"] = preferred_rows
    data["min_rows_remaining"] = max(0, min_rows - int(data["fresh_labeled_rows"]))
    data["preferred_rows_remaining"] = max(
        0, preferred_rows - int(data["fresh_labeled_rows"])
    )

    label_horizon = int(_cfg(config, ["labeling", "max_holding_bars"], 10))
    warnings: list[str] = []
    if raw_path.exists() and not labeled.empty:
        raw = pd.read_parquet(raw_path, columns=["timestamp"])
        raw_ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dropna()
        labeled_end = pd.to_datetime(data["data_end"], utc=True, errors="coerce")
        raw_end = raw_ts.max() if not raw_ts.empty else pd.NaT
        maturity_hours = (
            float((raw_end - labeled_end) / pd.Timedelta(hours=1))
            if not pd.isna(raw_end) and not pd.isna(labeled_end)
            else None
        )
        data["raw_data_end"] = raw_end.isoformat() if not pd.isna(raw_end) else None
        data["label_maturity_lag_hours"] = maturity_hours
        data["label_horizon_bars"] = label_horizon
        data["label_tail_matured"] = bool(
            maturity_hours is not None and maturity_hours >= label_horizon
        )
    else:
        data["raw_data_end"] = None
        data["label_maturity_lag_hours"] = None
        data["label_horizon_bars"] = label_horizon
        data["label_tail_matured"] = None
        warnings.append("raw_1h_unavailable_label_maturity_not_cross_checked")

    manifest_threshold = (manifest.get("threshold", {}) or {}).get("value")
    component_features = sorted(
        {
            str(column)
            for component in manifest.get("components", []) or []
            for column in component.get("feature_columns", []) or []
        }
    )
    missing_features = (
        sorted(set(component_features).difference(labeled.columns))
        if not labeled.empty
        else component_features
    )
    fit_cutoffs = [
        pd.to_datetime(component.get("fit_data_end"), utc=True, errors="coerce")
        for component in manifest.get("components", []) or []
    ]
    fit_cutoffs_valid = bool(
        fit_cutoffs
        and not pd.isna(anchor)
        and all(not pd.isna(value) and value <= anchor for value in fit_cutoffs)
    )
    run_dir = experiment_root(checkpoint_dir) / source_run_id
    artifact_errors = (
        verify_frozen_manifest_artifacts(manifest, run_dir=run_dir)
        if manifest
        else ["missing_primary_manifest"]
    )

    checks = {
        "protocol_enabled": bool(frozen.get("enabled", False)),
        "future_oos_enabled": bool(future.get("enabled", False)),
        "anchor_valid": not pd.isna(anchor),
        "primary_spec_present": bool(spec),
        "primary_manifest_present": bool(manifest),
        "candidate_id_matches": str(manifest.get("candidate_id", "")) == primary_id,
        "source_run_matches": str(manifest.get("source_run_id", "")) == source_run_id,
        "expected_manifest_hash_pinned": bool(expected_hash),
        "manifest_hash_matches_config": (
            bool(manifest)
            and str(manifest.get("manifest_hash", "")) == expected_hash
        ),
        "manifest_declared_available": bool(manifest.get("available", False)),
        "future_fit_forbidden": manifest.get("future_oos_fit_allowed") is False,
        "threshold_matches_config": (
            expected_threshold is not None
            and manifest_threshold is not None
            and bool(
                np.isclose(
                    float(manifest_threshold),
                    float(expected_threshold),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        ),
        "fit_cutoff_not_after_anchor": fit_cutoffs_valid,
        "artifact_integrity": not artifact_errors,
        "labeled_data_present": labeled_path.exists() and not labeled.empty,
        "timestamps_valid": bool(data["timestamps_valid"]),
        "timestamps_unique": bool(data["timestamps_unique"]),
        "timestamps_monotonic": bool(data["timestamps_monotonic"]),
        "required_columns_present": (
            not labeled.empty
            and {"timestamp", "label", "fwd_return_10h"}.issubset(labeled.columns)
        ),
        "frozen_feature_columns_present": not missing_features,
        "sequence_context_available": (
            not labeled.empty
            and not pd.isna(anchor)
            and int(
                (
                    pd.to_datetime(labeled["timestamp"], utc=True, errors="coerce")
                    <= anchor
                ).sum()
            )
            >= max(0, int(_cfg(config, ["model", "seq_len"], 64)) - 1)
        ),
    }
    invariants_passed = all(checks.values())
    enough_rows = int(data["fresh_labeled_rows"]) >= min_rows
    if not invariants_passed:
        state = "blocked_integrity_or_data_contract"
        next_action = "repair_preflight_blockers_do_not_evaluate"
    elif enough_rows:
        state = "ready_prediction_only"
        next_action = "run_notebook_05_prediction_only"
    else:
        state = "waiting_for_mature_labeled_rows"
        next_action = "refresh_01_02_03_then_recheck_without_running_04"

    return {
        "preflight_version": "future_oos_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "side_effect_free": True,
        "fit_operations_performed": 0,
        "state": state,
        "invariants_passed": invariants_passed,
        "ready_for_evaluation": bool(invariants_passed and enough_rows),
        "next_action": next_action,
        "required_notebook_sequence_when_refreshing": [
            "01_data_preparation.ipynb",
            "02_feature_engineering.ipynb",
            "03_labeling.ipynb",
            "05_diagnostics_validation.ipynb",
        ],
        "forbidden_before_frozen_evaluation": [
            "04_training_walk_forward.ipynb",
            "profile_changes",
            "threshold_changes",
            "manifest_regeneration_from_new_training",
        ],
        "primary_candidate": {
            "candidate_id": primary_id,
            "source_run_id": source_run_id,
            "anchor_data_end": None if pd.isna(anchor) else anchor.isoformat(),
            "expected_manifest_hash": expected_hash,
            "manifest_path": str(manifest_path),
            "manifest_hash": str(manifest.get("manifest_hash", "")),
            "threshold_expected": expected_threshold,
            "threshold_manifest": manifest_threshold,
            "threshold_source": str((manifest.get("threshold", {}) or {}).get("source", "")),
            "profile": str(spec.get("profile", "")),
            "component_count": len(manifest.get("components", []) or []),
            "model_count": sum(
                int(component.get("model_count", 0))
                for component in manifest.get("components", []) or []
            ),
        },
        "data": data,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "artifact_integrity_errors": artifact_errors,
        "missing_frozen_feature_columns": missing_features,
        "warnings": warnings,
    }


def future_oos_preflight_markdown(preflight: dict[str, Any]) -> str:
    """Render a compact operator-facing preflight summary."""

    candidate = preflight.get("primary_candidate", {}) or {}
    data = preflight.get("data", {}) or {}
    failed = preflight.get("failed_checks", []) or []
    integrity = preflight.get("artifact_integrity_errors", []) or []
    return "\n".join(
        [
            "# Future OOS Preflight",
            "",
            f"- State: `{preflight.get('state')}`",
            f"- Invariants passed: `{preflight.get('invariants_passed')}`",
            f"- Ready for evaluation: `{preflight.get('ready_for_evaluation')}`",
            f"- Side-effect free: `{preflight.get('side_effect_free')}`",
            f"- Fit operations performed: `{preflight.get('fit_operations_performed')}`",
            f"- Primary candidate: `{candidate.get('candidate_id')}`",
            f"- Source run: `{candidate.get('source_run_id')}`",
            f"- Manifest hash: `{candidate.get('manifest_hash')}`",
            f"- Frozen threshold: `{candidate.get('threshold_manifest')}`",
            f"- Fresh labeled rows: `{data.get('fresh_labeled_rows')}` / `{data.get('min_rows')}`",
            f"- Rows remaining: `{data.get('min_rows_remaining')}`",
            f"- Latest labeled timestamp: `{data.get('data_end')}`",
            f"- Failed checks: `{failed}`",
            f"- Artifact errors: `{integrity}`",
            f"- Next action: `{preflight.get('next_action')}`",
            "",
        ]
    )
