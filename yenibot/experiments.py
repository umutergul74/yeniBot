from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from yenibot.diagnostics import (
    attach_threshold_summary_to_phase1_report,
    calibration_table,
    calibrate_test_probabilities_from_val,
    bad_fold_regime_diagnostics,
    experiment_ledger_diagnostics,
    feature_group_diagnostics,
    feature_profile_diagnostics,
    fold_diagnostics,
    mtf_leakage_diagnostics,
    phase1_report,
    recent_fold_diagnostics,
    regime_by_fold_diagnostics,
    regime_diagnostics,
    score_band_by_fold_diagnostics,
    score_band_diagnostics,
    score_band_summary_diagnostics,
    score_policy_grid_diagnostics,
    select_score_policy,
    score_lift_by_fold_diagnostics,
    score_lift_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_grid_diagnostics,
    threshold_grid_summary_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)
from yenibot.features import filter_feature_columns, resolve_feature_profile, select_feature_columns
from yenibot.training import run_walk_forward_training
from yenibot.training.trainer import _add_regime_probs, _build_model, _device, _make_dataset, _predict_dataset


def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _set_cfg(config: dict[str, Any], path: list[str], value: Any) -> None:
    current: dict[str, Any] = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _table_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No rows were produced.")
        return "\n".join(lines)
    lines.append("| " + " | ".join(frame.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return an in-memory config copy with the requested active feature profile."""

    updated = copy.deepcopy(config)
    _set_cfg(updated, ["features", "active_profile"], profile)
    resolve_feature_profile(updated)
    return updated


def _profile_rejection_reason(profile: str, experiments: dict[str, Any]) -> str:
    memory = experiments.get("experiment_memory", {}) or {}
    if not bool(memory.get("enabled", False)) or not bool(memory.get("reject_retests", True)):
        return ""
    if profile in {str(item) for item in memory.get("allow_retest_profiles", []) or []}:
        return ""

    rejected_profiles = memory.get("rejected_profiles", {}) or {}
    if isinstance(rejected_profiles, dict) and profile in rejected_profiles:
        value = rejected_profiles[profile]
        if isinstance(value, dict):
            return str(value.get("reason") or "historically_rejected_profile")
        return str(value or "historically_rejected_profile")

    for item in memory.get("rejected_profile_patterns", []) or []:
        if isinstance(item, str):
            pattern = item
            reason = "historically_rejected_profile_pattern"
        elif isinstance(item, dict):
            pattern = str(item.get("pattern", ""))
            reason = str(item.get("reason") or "historically_rejected_profile_pattern")
        else:
            continue
        if pattern and fnmatch(profile, pattern):
            return reason
    return ""


def _filter_memory_rejected_profiles(
    profiles: list[str],
    experiments: dict[str, Any],
    *,
    role: str,
    protected_profiles: set[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    protected = set() if protected_profiles is None else {str(profile) for profile in protected_profiles}
    selected: list[str] = []
    skipped: list[dict[str, str]] = []
    for profile in profiles:
        profile = str(profile)
        if profile in selected:
            continue
        reason = "" if profile in protected else _profile_rejection_reason(profile, experiments)
        if reason:
            skipped.append({"profile": profile, "role": role, "skip_reason": reason})
            continue
        selected.append(profile)
    return selected, skipped


def experiment_settings(config: dict[str, Any]) -> dict[str, Any]:
    experiments = copy.deepcopy(_cfg(config, ["experiments"], {}) or {})
    control = str(experiments.get("control_profile") or _cfg(config, ["features", "active_profile"]))
    raw_candidates = [str(profile) for profile in experiments.get("candidate_profiles", [])]
    candidates, skipped_candidates = _filter_memory_rejected_profiles(
        raw_candidates,
        experiments,
        role="candidate_profile",
        protected_profiles={control},
    )
    profiles = []
    for profile in [control, *candidates]:
        if profile not in profiles:
            profiles.append(profile)
    experiments.setdefault("mode", "staged")
    experiments["control_profile"] = control
    experiments["candidate_profiles"] = [profile for profile in profiles if profile != control]
    experiments["profiles"] = profiles
    experiments.setdefault("triage_fold_ids", [])
    experiments.setdefault("full_cv_profiles", "auto")
    experiments.setdefault("always_full_profiles", [control])
    always_full, skipped_always_full = _filter_memory_rejected_profiles(
        [str(profile) for profile in experiments.get("always_full_profiles", []) or []],
        experiments,
        role="always_full_profile",
        protected_profiles={control},
    )
    if control not in always_full:
        always_full.insert(0, control)
    experiments["always_full_profiles"] = always_full
    experiments.setdefault("max_auto_full_candidates", None)
    experiments.setdefault("resume_existing", True)
    experiments.setdefault("force_retrain", False)
    seed_audit = copy.deepcopy(experiments.get("seed_audit", {}) or {})
    seed_audit.setdefault("enabled", False)
    seed_audit.setdefault("profiles", [control])
    seed_profiles, skipped_seed_profiles = _filter_memory_rejected_profiles(
        [str(profile) for profile in seed_audit.get("profiles", []) or [control]],
        experiments,
        role="seed_audit_profile",
        protected_profiles={control},
    )
    seed_audit["profiles"] = seed_profiles or [control]
    seed_audit.setdefault("seeds", [])
    seed_audit.setdefault("fold_ids", experiments.get("triage_fold_ids", []))
    experiments["seed_audit"] = seed_audit
    experiments["skipped_profiles"] = [*skipped_candidates, *skipped_always_full, *skipped_seed_profiles]
    return experiments


def _profile_requires_intrahour_features(config: dict[str, Any], profile: str) -> bool:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    return any("ih15" in str(pattern) for pattern in resolved.get("include_patterns", []) or [])


def _missing_intrahour_include_patterns(config: dict[str, Any], profile: str, feature_columns: tuple[str, ...]) -> list[str]:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    patterns = [str(pattern) for pattern in resolved.get("include_patterns", []) or [] if "ih15_" in str(pattern)]
    return [
        pattern
        for pattern in patterns
        if not any(fnmatch(column, pattern) for column in feature_columns)
    ]


def _preflight_experiment_profiles(
    settings: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Skip candidate profiles that cannot change the current feature matrix."""

    updated = copy.deepcopy(settings)
    base_columns = select_feature_columns(frame)
    control = str(updated["control_profile"])
    selected_profiles: list[str] = []
    skipped_profiles = list(updated.get("skipped_profiles", []) or [])
    seen_signatures: dict[tuple[str, ...], str] = {}

    for profile in [str(item) for item in updated.get("profiles", [])]:
        cfg = profile_config(config, profile)
        feature_columns = tuple(filter_feature_columns(base_columns, cfg))
        has_intrahour = any(column.startswith("ih15_") for column in feature_columns)
        missing_intrahour_patterns = _missing_intrahour_include_patterns(config, profile, feature_columns)
        if profile != control and _profile_requires_intrahour_features(config, profile) and (
            not has_intrahour or missing_intrahour_patterns
        ):
            reason = "missing_intrahour_features_rerun_01_02_03"
            if missing_intrahour_patterns:
                reason = f"{reason}:{','.join(missing_intrahour_patterns[:6])}"
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": reason,
                }
            )
            continue
        duplicate_of = seen_signatures.get(feature_columns)
        if profile != control and duplicate_of:
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": f"duplicate_feature_signature:{duplicate_of}",
                }
            )
            continue
        selected_profiles.append(profile)
        seen_signatures[feature_columns] = profile

    if control not in selected_profiles:
        raise ValueError(f"Control profile was removed during experiment preflight: {control}")

    selected_set = set(selected_profiles)
    updated["profiles"] = selected_profiles
    updated["candidate_profiles"] = [profile for profile in selected_profiles if profile != control]
    updated["always_full_profiles"] = [
        str(profile)
        for profile in updated.get("always_full_profiles", []) or []
        if str(profile) == control or str(profile) in selected_set or not _profile_requires_intrahour_features(config, str(profile))
    ]
    seed_audit = copy.deepcopy(updated.get("seed_audit", {}) or {})
    if seed_audit:
        seed_audit["profiles"] = [
            str(profile)
            for profile in seed_audit.get("profiles", []) or []
            if str(profile) == control or str(profile) in selected_set or not _profile_requires_intrahour_features(config, str(profile))
        ]
        updated["seed_audit"] = seed_audit
    updated["skipped_profiles"] = skipped_profiles
    return updated


def experiment_root(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "experiments"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _experiment_signature(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    comparable_settings = copy.deepcopy(settings)
    comparable_settings.pop("run_id", None)
    return {
        "settings": comparable_settings,
        "feature_profiles": _cfg(config, ["features", "profiles"], {}),
        "model": _cfg(config, ["model"], {}),
        "training": _cfg(config, ["training"], {}),
        "walk_forward": _cfg(config, ["walk_forward"], {}),
        "validation": _cfg(config, ["validation"], {}),
    }


def _matching_latest_run(checkpoint_dir: str | Path, signature_hash: str) -> Path | None:
    root = experiment_root(checkpoint_dir)
    if not root.exists():
        return None
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name, reverse=True)
    for run in runs:
        manifest_path = run / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
        except json.JSONDecodeError:
            continue
        if manifest.get("signature_hash") == signature_hash:
            return run
    return None


def resolve_experiment_run_id(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    settings: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[str, str]:
    settings = experiment_settings(config) if settings is None else settings
    if run_id:
        return str(run_id), "explicit_argument"
    if settings.get("run_id"):
        return str(settings["run_id"]), "config"
    signature_hash = _hash_payload(_experiment_signature(config, settings))
    if bool(settings.get("resume_existing", True)) and not bool(settings.get("force_retrain", False)):
        existing = _matching_latest_run(checkpoint_dir, signature_hash)
        if existing is not None:
            return existing.name, "matching_existing"
    return new_run_id(), "new"


def latest_experiment_run(checkpoint_dir: str | Path) -> Path:
    root = experiment_root(checkpoint_dir)
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name)
    if not runs:
        raise FileNotFoundError(f"No experiment runs found under {root}")
    return runs[-1]


def profile_run_dir(checkpoint_dir: str | Path, run_id: str, profile: str) -> Path:
    return experiment_root(checkpoint_dir) / run_id / _slug(profile)


def _frame_window(frame: pd.DataFrame) -> dict[str, str]:
    if "timestamp" not in frame.columns or frame.empty:
        return {"data_start": "", "data_end": ""}
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return {"data_start": str(timestamps.min()), "data_end": str(timestamps.max())}


def _training_signature(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    profile: str,
    feature_columns: list[str],
    fold_ids: list[int] | None,
    fold_scope: str,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "fold_scope": fold_scope,
        "fold_ids": fold_ids,
        "feature_columns": feature_columns,
        "feature_columns_hash": _hash_payload(feature_columns),
        "config_hash": _hash_payload(config),
        "frame_rows": int(len(frame)),
        **_frame_window(frame),
    }


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "training_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _is_complete(output_dir: Path, expected_signature_hash: str) -> bool:
    manifest_path = _manifest_path(output_dir)
    predictions_path = output_dir / "predictions_all.parquet"
    if not manifest_path.exists() or not predictions_path.exists():
        return False
    manifest = _read_json(manifest_path)
    return bool(manifest.get("completed")) and manifest.get("signature_hash") == expected_signature_hash


def _test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "split" in predictions.columns:
        return predictions[predictions["split"] == "test"].copy()
    return predictions.copy()


def summarize_profile_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    *,
    profile: str,
    feature_columns: list[str],
    fold_scope: str,
    promotable: bool | None = None,
    reject_reason: str = "",
) -> dict[str, Any]:
    profile_cfg = profile_config(config, profile)
    test_predictions = _test_predictions(predictions)
    report = phase1_report(test_predictions, profile_cfg)
    calibration = calibration_table(
        test_predictions["label"],
        test_predictions["prob_long"],
        bins=int(_cfg(profile_cfg, ["validation", "calibration_bins"], 10)),
    )
    fold_metrics = fold_diagnostics(test_predictions)
    regime_metrics = regime_diagnostics(test_predictions)
    regime_by_fold = regime_by_fold_diagnostics(
        test_predictions,
        fold_metrics,
        bad_ic=float(_cfg(profile_cfg, ["validation", "bad_fold_ic_threshold"], -0.08)),
    )
    bad_fold_regime = bad_fold_regime_diagnostics(regime_by_fold)
    threshold_cfg = _cfg(profile_cfg, ["validation", "threshold_checks"], {}) or {}
    threshold_metrics = threshold_diagnostics(
        predictions,
        max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    threshold_summary = threshold_summary_diagnostics(threshold_metrics)
    report = attach_threshold_summary_to_phase1_report(report, threshold_summary, profile_cfg)
    calibrated_report = None
    calibrated_calibration = pd.DataFrame()
    calibrated_predictions = pd.DataFrame()
    calibration_cfg = _cfg(profile_cfg, ["validation", "calibration"], {}) or {}
    if bool(calibration_cfg.get("enabled", False)):
        try:
            calibrated_predictions, calibrated_report, calibrated_calibration = calibrate_test_probabilities_from_val(
                predictions,
                profile_cfg,
                method=str(calibration_cfg.get("method", "isotonic")),
            )
        except ValueError:
            calibrated_report = None
            calibrated_calibration = pd.DataFrame()
            calibrated_predictions = pd.DataFrame()
    score_bins = int(_cfg(profile_cfg, ["validation", "score_lift_bins"], _cfg(profile_cfg, ["validation", "calibration_bins"], 10)))
    score_bands = _cfg(profile_cfg, ["validation", "score_bands"], None)
    policy_cfg = _cfg(profile_cfg, ["validation", "policy_selection"], {}) or {}
    threshold_caps = [float(value) for value in policy_cfg.get("threshold_caps", [0.30, 0.40, 0.50, 0.60, 0.70])]
    score_lift = score_lift_diagnostics(test_predictions, bins=score_bins)
    score_lift_by_fold = score_lift_by_fold_diagnostics(test_predictions, bins=score_bins)
    score_band_lift = score_band_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_by_fold = score_band_by_fold_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_summary = score_band_summary_diagnostics(score_band_by_fold)
    threshold_grid = threshold_grid_diagnostics(
        predictions,
        max_pred_long_rates=threshold_caps,
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    threshold_grid_summary = threshold_grid_summary_diagnostics(threshold_grid)
    score_policy_grid = score_policy_grid_diagnostics(
        predictions,
        bins=score_bins,
        bands=score_bands,
        threshold_caps=threshold_caps,
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    score_policy_selection = select_score_policy(score_policy_grid, profile_cfg)
    recent = recent_fold_diagnostics(
        fold_metrics,
        recent_folds=int(_cfg(profile_cfg, ["validation", "recent_folds"], 5)),
    )
    mtf = mtf_leakage_diagnostics(test_predictions)
    stationarity = stationarity_policy_diagnostics(feature_columns, profile_cfg)
    data_window = _frame_window(test_predictions)
    ledger = experiment_ledger_diagnostics(
        report=report,
        config=profile_cfg,
        feature_columns=feature_columns,
        fold_metrics=fold_metrics,
        recent_fold_summary=recent,
        threshold_summary=threshold_summary,
        score_band_lift=score_band_lift,
        score_lift_by_fold=score_lift_by_fold,
        score_band_summary=score_band_summary,
        fold_scope=fold_scope,
        data_start=data_window["data_start"],
        data_end=data_window["data_end"],
        promotable=promotable,
        reject_reason=reject_reason,
    )
    row = ledger.iloc[0].to_dict()
    row["mtf_leakage_passed"] = bool(mtf.empty or mtf["passed"].all())
    row["stationarity_policy_passed"] = bool(stationarity.empty or stationarity["passed"].all())
    row["fold_count"] = int(fold_metrics["fold"].nunique()) if not fold_metrics.empty else 0
    return {
        "report": report,
        "calibration": calibration,
        "calibrated_report": calibrated_report,
        "calibrated_calibration": calibrated_calibration,
        "calibrated_predictions": calibrated_predictions,
        "fold_metrics": fold_metrics,
        "regime_metrics": regime_metrics,
        "regime_by_fold": regime_by_fold,
        "bad_fold_regime": bad_fold_regime,
        "threshold_metrics": threshold_metrics,
        "threshold_summary": threshold_summary,
        "threshold_grid": threshold_grid,
        "threshold_grid_summary": threshold_grid_summary,
        "score_lift": score_lift,
        "score_lift_by_fold": score_lift_by_fold,
        "score_band_lift": score_band_lift,
        "score_band_by_fold": score_band_by_fold,
        "score_band_summary": score_band_summary,
        "score_policy_grid": score_policy_grid,
        "score_policy_selection": score_policy_selection,
        "recent_fold_summary": recent,
        "mtf_leakage": mtf,
        "stationarity_policy": stationarity,
        "feature_groups": feature_group_diagnostics(feature_columns),
        "feature_profile": feature_profile_diagnostics(feature_columns, profile_cfg),
        "ledger": ledger,
        "row": row,
    }


def run_profile_experiment(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    profile: str,
    checkpoint_dir: str | Path,
    run_id: str,
    fold_scope: str,
    fold_ids: list[int] | None = None,
    resume_existing: bool = True,
    force_retrain: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    cfg = profile_config(config, profile)
    feature_columns = filter_feature_columns(select_feature_columns(frame), cfg)
    output_dir = profile_run_dir(checkpoint_dir, run_id, profile) / fold_scope
    signature = _training_signature(
        frame=frame,
        config=cfg,
        profile=profile,
        feature_columns=feature_columns,
        fold_ids=fold_ids,
        fold_scope=fold_scope,
    )
    signature_hash = _hash_payload(signature)
    skipped = False
    if resume_existing and not force_retrain and _is_complete(output_dir, signature_hash):
        predictions = pd.read_parquet(output_dir / "predictions_all.parquet")
        skipped = True
    else:
        result = run_walk_forward_training(
            frame,
            cfg,
            feature_columns=feature_columns,
            checkpoint_dir=output_dir,
            fold_ids=fold_ids,
            device=device,
        )
        predictions = result["predictions"]
        manifest = {
            **signature,
            "signature_hash": signature_hash,
            "completed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prediction_rows": int(len(predictions)),
        }
        _write_json(_manifest_path(output_dir), manifest)

    diagnostics = summarize_profile_predictions(
        predictions,
        config,
        profile=profile,
        feature_columns=feature_columns,
        fold_scope=fold_scope,
    )
    return {
        "profile": profile,
        "fold_scope": fold_scope,
        "output_dir": output_dir,
        "skipped": skipped,
        "feature_columns": feature_columns,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "summary": diagnostics["row"],
    }


def _float(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_gate_float(gates: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = gates.get(key, default)
    if value is None:
        return None
    return float(value)


def _metric_or(row: dict[str, Any], key: str, fallback: float) -> float:
    value = _float(row, key, np.nan)
    if np.isnan(value):
        return fallback
    return value


def _passes_triage(row: dict[str, Any], control: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    gates = _cfg(config, ["experiments", "promotion_gates", "triage"], {}) or {}
    reasons = []
    if _float(row, "mean_rank_ic") < _float(control, "mean_rank_ic") + float(gates.get("min_mean_rank_ic_delta", 0.005)):
        reasons.append("mean_rank_ic_delta")
    if _float(row, "std_rank_ic") > _float(control, "std_rank_ic") + float(gates.get("max_std_rank_ic_delta", 0.005)):
        reasons.append("std_rank_ic")
    if _float(row, "positive_ic_fraction") < _float(control, "positive_ic_fraction"):
        reasons.append("positive_ic_fraction")
    if _float(row, "top_10_lift_global") < float(gates.get("min_top_10_lift_global", 1.05)):
        reasons.append("top_10_lift_global")
    if _float(row, "top_10_positive_lift_fold_rate") < float(gates.get("min_top_10_positive_lift_fold_rate", 0.55)):
        reasons.append("top_10_positive_lift_fold_rate")
    worst_5_delta = _optional_gate_float(gates, "min_worst_5_rank_ic_delta", None)
    if worst_5_delta is not None and _float(row, "worst_5_rank_ic_mean") < _float(control, "worst_5_rank_ic_mean") + worst_5_delta:
        reasons.append("worst_5_rank_ic_delta")
    negative_delta = _optional_gate_float(gates, "max_negative_ic_fraction_delta", None)
    if negative_delta is not None and _float(row, "negative_ic_fraction") > _float(control, "negative_ic_fraction") + negative_delta:
        reasons.append("negative_ic_fraction")
    bad_fold_lift_floor = _optional_gate_float(gates, "min_top_10_bad_fold_lift_mean", None)
    if bad_fold_lift_floor is not None and _float(row, "top_10_bad_fold_lift_mean") < bad_fold_lift_floor:
        reasons.append("top_10_bad_fold_lift_mean")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return not reasons, ";".join(reasons)


def _passes_full(row: dict[str, Any], control: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    if bool(row.get("passed_phase1", False)):
        return True, ""
    gates = _cfg(config, ["experiments", "promotion_gates", "full"], {}) or {}
    reasons = []
    if _float(row, "mean_rank_ic") < _float(control, "mean_rank_ic") + float(gates.get("min_mean_rank_ic_delta", 0.005)):
        reasons.append("mean_rank_ic_delta")
    min_positive = max(_float(control, "positive_ic_fraction"), float(gates.get("min_positive_ic_fraction_floor", 0.75)))
    if _float(row, "positive_ic_fraction") < min_positive:
        reasons.append("positive_ic_fraction")
    if _float(row, "std_rank_ic") > _float(control, "std_rank_ic") + float(gates.get("max_std_rank_ic_delta", 0.0)):
        reasons.append("std_rank_ic")
    selected_f1 = _metric_or(
        row,
        "test_f1_at_constrained_threshold",
        _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
    )
    control_selected_f1 = _metric_or(
        control,
        "test_f1_at_constrained_threshold",
        _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
    )
    selected_f1_floor = _optional_gate_float(gates, "min_selected_threshold_f1", None)
    if selected_f1_floor is not None and selected_f1 < selected_f1_floor:
        reasons.append("constrained_threshold_f1")
    selected_f1_delta = _optional_gate_float(gates, "min_selected_threshold_f1_delta", None)
    if selected_f1_delta is not None and selected_f1 < control_selected_f1 + selected_f1_delta:
        reasons.append("constrained_threshold_f1_delta")
    threshold_checks = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_checks.get("max_pred_long_rate", 0.70))
    constrained_pred_rate = _float(row, "test_pred_long_rate_at_constrained_threshold", np.nan)
    if np.isfinite(constrained_pred_rate) and constrained_pred_rate > max_pred_long_rate:
        reasons.append("constrained_pred_long_rate")
    mean_long_f1_delta = _optional_gate_float(gates, "min_long_f1_delta", None)
    if mean_long_f1_delta is not None and _float(row, "mean_long_f1") < _float(control, "mean_long_f1") + mean_long_f1_delta:
        reasons.append("mean_long_f1_delta")
    if _float(row, "top_10_lift_global") < _float(control, "top_10_lift_global") + float(gates.get("min_top_10_lift_global_delta", 0.05)):
        reasons.append("top_10_lift_global_delta")
    top_lift_floor = _optional_gate_float(gates, "min_top_10_lift_global", None)
    if top_lift_floor is not None and _float(row, "top_10_lift_global") < top_lift_floor:
        reasons.append("top_10_lift_global")
    worst_5_delta = _optional_gate_float(gates, "min_worst_5_rank_ic_delta", None)
    if worst_5_delta is not None and _float(row, "worst_5_rank_ic_mean") < _float(control, "worst_5_rank_ic_mean") + worst_5_delta:
        reasons.append("worst_5_rank_ic_delta")
    negative_delta = _optional_gate_float(gates, "max_negative_ic_fraction_delta", None)
    if negative_delta is not None and _float(row, "negative_ic_fraction") > _float(control, "negative_ic_fraction") + negative_delta:
        reasons.append("negative_ic_fraction")
    bad_fold_lift_delta = _optional_gate_float(gates, "min_top_10_bad_fold_lift_mean_delta", None)
    if bad_fold_lift_delta is not None and _float(row, "top_10_bad_fold_lift_mean") < _float(control, "top_10_bad_fold_lift_mean") + bad_fold_lift_delta:
        reasons.append("top_10_bad_fold_lift_mean_delta")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return not reasons, ";".join(reasons)


def _decision_rows(rows: list[dict[str, Any]], config: dict[str, Any], *, scope: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    settings = experiment_settings(config)
    control_profile = settings["control_profile"]
    control = next(row for row in rows if row["profile"] == control_profile)
    decided = []
    for row in rows:
        updated = dict(row)
        if row["profile"] == control_profile:
            updated["promotable"] = False
            updated["reject_reason"] = "control_profile"
        elif scope == "triage":
            passed, reason = _passes_triage(row, control, config)
            updated["promotable"] = passed
            updated["reject_reason"] = reason
        else:
            passed, reason = _passes_full(row, control, config)
            updated["promotable"] = passed
            updated["reject_reason"] = reason
        decided.append(updated)
    return decided


def _auto_full_profiles(settings: dict[str, Any], triage_rows: list[dict[str, Any]]) -> list[str]:
    control_profile = str(settings["control_profile"])
    profiles = [control_profile]
    for profile in settings.get("always_full_profiles", []) or []:
        profile = str(profile)
        if profile not in profiles:
            profiles.append(profile)

    passed_candidates = [
        row
        for row in triage_rows
        if row["profile"] != control_profile and bool(row.get("promotable"))
    ]
    passed_candidates = sorted(
        passed_candidates,
        key=lambda row: (
            _float(row, "mean_rank_ic", -np.inf),
            _float(row, "top_10_lift_global", -np.inf),
            _float(row, "worst_5_rank_ic_mean", -np.inf),
            _float(row, "top_10_positive_lift_fold_rate", -np.inf),
        ),
        reverse=True,
    )
    max_auto = settings.get("max_auto_full_candidates", None)
    if max_auto is not None:
        passed_candidates = passed_candidates[: max(0, int(max_auto))]

    for row in passed_candidates:
        profile = str(row["profile"])
        if profile not in profiles:
            profiles.append(profile)
    return profiles


def _comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "profile",
        "fold_scope",
        "feature_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "mean_long_f1",
        "test_f1_at_selected_threshold",
        "test_precision_at_selected_threshold",
        "test_recall_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "selected_threshold_mean",
        "test_f1_at_constrained_threshold",
        "test_precision_at_constrained_threshold",
        "test_recall_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "constrained_threshold_mean",
        "mean_prauc",
        "calibration_separation",
        "recent_rank_ic_mean",
        "recent_rank_ic_min",
        "negative_ic_count",
        "negative_ic_fraction",
        "worst_5_rank_ic_mean",
        "rank_ic_cvar_20",
        "bad_fold_rank_ic_mean",
        "top_10_lift_fold_mean",
        "top_10_lift_global",
        "top_10_positive_lift_fold_rate",
        "top_10_bad_fold_lift_mean",
        "mtf_leakage_passed",
        "stationarity_policy_passed",
        "passed_phase1",
        "passed_phase1_selected_threshold",
        "passed_phase1_constrained_threshold",
        "promotable",
        "reject_reason",
        "data_start",
        "data_end",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].sort_values(["fold_scope", "mean_rank_ic"], ascending=[True, False]).reset_index(drop=True)


def _best_candidate(comparison: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    candidates = comparison[
        (comparison["profile"] != control_profile)
        & (comparison["fold_scope"] == "full")
        & (comparison["promotable"].astype(bool))
    ].copy()
    if candidates.empty:
        return {}
    candidates = candidates.sort_values(
        ["passed_phase1", "mean_rank_ic", "top_10_lift_global", "worst_5_rank_ic_mean"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0].to_dict()


def _comparison_markdown(comparison: pd.DataFrame, decision: dict[str, Any]) -> str:
    lines = ["# Experiment Profile Comparison", ""]
    if comparison.empty:
        lines.append("No profile runs were found.")
    else:
        display_cols = [
            "profile",
            "fold_scope",
            "feature_count",
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "worst_5_rank_ic_mean",
            "mean_long_f1",
            "test_f1_at_selected_threshold",
            "test_f1_at_constrained_threshold",
            "test_pred_long_rate_at_constrained_threshold",
            "top_10_lift_global",
            "top_10_bad_fold_lift_mean",
            "passed_phase1_selected_threshold",
            "passed_phase1_constrained_threshold",
            "promotable",
            "reject_reason",
        ]
        visible = comparison[display_cols].copy()
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for _, row in visible.iterrows():
            values = [str(row[column]) for column in display_cols]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(decision), indent=2, sort_keys=True)])
    return "\n".join(lines)


def _write_decision_files(run_dir: Path, comparison: pd.DataFrame, decision: dict[str, Any]) -> None:
    comparison.to_csv(run_dir / "profile_comparison.csv", index=False)
    (run_dir / "profile_comparison.md").write_text(_comparison_markdown(comparison, decision), encoding="utf-8")
    _write_json(run_dir / "decision_report.json", decision)
    _write_json(run_dir / "best_candidate.json", decision.get("best_candidate") or {})


def _experiment_selection_frame(settings: dict[str, Any]) -> pd.DataFrame:
    columns = ["profile", "role", "selected", "expected_fold_scope", "skip_reason"]
    rows: list[dict[str, Any]] = [
        {
            "profile": str(settings["control_profile"]),
            "role": "control_profile",
            "selected": True,
            "expected_fold_scope": "triage",
            "skip_reason": "",
        }
    ]
    for role, key in (
        ("candidate_profile", "candidate_profiles"),
        ("always_full_profile", "always_full_profiles"),
        ("seed_audit_profile", "seed_audit_profiles"),
    ):
        values = settings.get(key, [])
        if key == "seed_audit_profiles":
            values = (settings.get("seed_audit", {}) or {}).get("profiles", [])
        expected_scope = {
            "candidate_profile": "triage",
            "always_full_profile": "full",
            "seed_audit_profile": "seed_audit",
        }[role]
        for profile in values or []:
            rows.append(
                {
                    "profile": str(profile),
                    "role": role,
                    "selected": True,
                    "expected_fold_scope": expected_scope,
                    "skip_reason": "",
                }
            )
    for skipped in settings.get("skipped_profiles", []) or []:
        rows.append(
            {
                "profile": str(skipped.get("profile", "")),
                "role": str(skipped.get("role", "skipped_profile")),
                "selected": False,
                "expected_fold_scope": "",
                "skip_reason": str(skipped.get("skip_reason", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def _experiment_selection_markdown(selection: pd.DataFrame) -> str:
    lines = ["# Experiment Selection", ""]
    if selection.empty:
        lines.append("No profile selection metadata was produced.")
        return "\n".join(lines)
    lines.append("| profile | role | selected | expected_fold_scope | skip_reason |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in selection.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(bool(row["selected"])),
                    str(row.get("expected_fold_scope", "")),
                    str(row.get("skip_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_experiment_selection(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    selection = _experiment_selection_frame(settings)
    selection.to_csv(path / "experiment_selection.csv", index=False)
    (path / "experiment_selection.md").write_text(_experiment_selection_markdown(selection), encoding="utf-8")
    _write_json(
        path / "experiment_selection.json",
        {
            "control_profile": settings.get("control_profile", ""),
            "selected_profiles": selection.loc[selection["selected"].astype(bool), "profile"].drop_duplicates().tolist(),
            "skipped_profiles": settings.get("skipped_profiles", []) or [],
            "rows": selection.to_dict(orient="records"),
        },
    )
    return selection


def _missing_selected_profiles(selection: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    columns = ["profile", "role", "expected_fold_scope", "reason"]
    if selection.empty:
        return pd.DataFrame(columns=columns)
    completed = {
        (str(row["profile"]), str(row["fold_scope"]))
        for _, row in comparison.iterrows()
        if str(row.get("fold_scope", "")) in {"triage", "full"}
    }
    rows = []
    comparable_scopes = {"triage", "full"}
    for _, row in selection.iterrows():
        if not bool(row.get("selected", False)):
            continue
        scope = str(row.get("expected_fold_scope", ""))
        if scope not in comparable_scopes:
            continue
        profile = str(row.get("profile", ""))
        if (profile, scope) in completed:
            continue
        rows.append(
            {
                "profile": profile,
                "role": str(row.get("role", "")),
                "expected_fold_scope": scope,
                "reason": "missing_selected_profile_output",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _missing_selected_markdown(missing: pd.DataFrame) -> str:
    lines = ["# Missing Selected Profiles", ""]
    if missing.empty:
        lines.append("All selected comparison profiles have completed outputs.")
        return "\n".join(lines)
    lines.append("| profile | role | expected_fold_scope | reason |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in missing.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(row["expected_fold_scope"]),
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_missing_selected_profiles(path: Path, missing: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "missing_selected_profiles.md").write_text(_missing_selected_markdown(missing), encoding="utf-8")
    _write_json(path / "missing_selected_profiles.json", {"rows": missing.to_dict(orient="records")})


def _parquet_timestamps(path: Path) -> pd.Series:
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except (TypeError, ValueError):
        frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["timestamp"], utc=True).dropna()


def _default_holdout_path(config: dict[str, Any], holdout: dict[str, Any]) -> Path | None:
    explicit = str(holdout.get("holdout_path") or "").strip()
    if explicit:
        return Path(explicit)
    data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
    if not data_dir:
        return None
    filename = str(holdout.get("holdout_filename") or "holdout_1h.parquet")
    return Path(str(data_dir)) / "processed" / filename


def _resolve_holdout_settings(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Attach durable holdout metadata from config/default Drive paths.

    Notebook 04 injects holdout metadata into its in-memory config before training,
    but notebook 05 may be run in a fresh session. This resolver lets diagnostics
    recover the reserved holdout from config and the standard parquet location.
    """

    updated = copy.deepcopy(settings)
    holdout = copy.deepcopy(updated.get("holdout") or _cfg(config, ["experiments", "holdout"], {}) or {})
    if not holdout:
        return updated

    holdout.setdefault("enabled", True)
    holdout.setdefault("policy", "profile_selection_only_before_holdout; holdout is reserved for one-shot final validation")
    holdout_path = _default_holdout_path(config, holdout)
    if holdout_path is not None:
        holdout["holdout_path"] = str(holdout_path)
        if holdout_path.exists():
            timestamps = _parquet_timestamps(holdout_path)
            if not timestamps.empty:
                holdout.setdefault("holdout_rows", int(len(timestamps)))
                holdout.setdefault("holdout_bars", int(holdout.get("holdout_rows", len(timestamps))))
                holdout.setdefault("holdout_data_start", str(timestamps.min()))
                holdout.setdefault("holdout_data_end", str(timestamps.max()))

                data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
                labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet" if data_dir else None
                if labeled_path is not None and labeled_path.exists():
                    labeled_timestamps = _parquet_timestamps(labeled_path)
                    if not labeled_timestamps.empty:
                        holdout_start = pd.to_datetime(holdout["holdout_data_start"], utc=True)
                        selection_timestamps = labeled_timestamps.loc[labeled_timestamps < holdout_start]
                        if not selection_timestamps.empty:
                            holdout.setdefault("selection_rows", int(len(selection_timestamps)))
                            holdout.setdefault("selection_data_start", str(selection_timestamps.min()))
                            holdout.setdefault("selection_data_end", str(selection_timestamps.max()))

    updated["holdout"] = holdout
    return updated


def _selection_frame_before_holdout(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)) or "timestamp" not in frame.columns:
        return frame
    holdout_start = holdout.get("holdout_data_start")
    if not holdout_start:
        return frame
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = pd.to_datetime(holdout_start, utc=True)
    if not (timestamps >= start).any():
        return frame
    return frame.loc[timestamps < start].copy().reset_index(drop=True)


def _holdout_reservation_frame(settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    columns = [
        "enabled",
        "holdout_bars",
        "selection_rows",
        "holdout_rows",
        "selection_data_start",
        "selection_data_end",
        "holdout_data_start",
        "holdout_data_end",
        "holdout_path",
        "policy",
    ]
    if not holdout:
        return pd.DataFrame(columns=columns)
    row = {column: holdout.get(column, "") for column in columns}
    row["enabled"] = bool(holdout.get("enabled", False))
    return pd.DataFrame([row], columns=columns)


def _holdout_reservation_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Holdout Reservation", ""]
    if frame.empty:
        lines.append("No holdout reservation metadata was attached to this experiment run.")
        return "\n".join(lines)
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    row = frame.iloc[0].to_dict()
    for key, value in row.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _write_holdout_reservation(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    frame = _holdout_reservation_frame(settings)
    frame.to_csv(path / "holdout_reservation.csv", index=False)
    (path / "holdout_reservation.md").write_text(_holdout_reservation_markdown(frame), encoding="utf-8")
    _write_json(path / "holdout_reservation.json", {"rows": frame.to_dict(orient="records")})
    return frame


def _read_holdout_context(settings: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return pd.DataFrame(), None
    holdout_path = Path(str(holdout.get("holdout_path", "")))
    if not holdout_path.exists():
        return pd.DataFrame(), None

    holdout_frame = pd.read_parquet(holdout_path).copy()
    if holdout_frame.empty or "timestamp" not in holdout_frame.columns:
        return pd.DataFrame(), None
    holdout_frame["timestamp"] = pd.to_datetime(holdout_frame["timestamp"], utc=True)
    holdout_start = pd.to_datetime(holdout.get("holdout_data_start", holdout_frame["timestamp"].min()), utc=True)

    seq_len = int(_cfg(config, ["model", "seq_len"], 64))
    context_rows = max(seq_len - 1, 0)
    context = pd.DataFrame()
    data_dir = _cfg(config, ["paths", "data_dir"], None)
    if data_dir:
        labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet"
        if labeled_path.exists() and context_rows:
            full = pd.read_parquet(labeled_path)
            if "timestamp" in full.columns:
                full = full.copy()
                full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
                selection_end = pd.to_datetime(holdout.get("selection_data_end", holdout_start), utc=True)
                context = full.loc[full["timestamp"] <= selection_end].tail(context_rows).copy()

    if not context.empty:
        frame = pd.concat([context, holdout_frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    else:
        frame = holdout_frame.sort_values("timestamp").reset_index(drop=True)
    return frame, holdout_start


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict_holdout_for_profile(
    *,
    scope_dir: Path,
    manifest: dict[str, Any],
    holdout_context: pd.DataFrame,
    holdout_start: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    profile = str(manifest["profile"])
    cfg = profile_config(config, profile)
    feature_columns = list(manifest["feature_columns"])
    required = [*feature_columns, *list(_cfg(cfg, ["hmm", "features"], []) or []), "label"]
    forward_column = f"fwd_return_{int(_cfg(cfg, ['labeling', 'max_holding_bars'], 10))}h"
    if forward_column not in holdout_context.columns:
        forward_column = "fwd_return_10h"
    required.append(forward_column)
    missing = [column for column in dict.fromkeys(required) if column not in holdout_context.columns]
    if missing:
        raise ValueError(f"Holdout frame is missing columns for {profile}: {missing}")

    torch_device = _device(None)
    batch_size = int(_cfg(cfg, ["training", "batch_size"], 256))
    rows = []
    model_paths = sorted(scope_dir.glob("model_fold_*.pt"))
    for model_path in model_paths:
        fold = int(model_path.stem.rsplit("_", 1)[-1])
        scaler_path = scope_dir / f"scaler_fold_{fold:03d}.pkl"
        hmm_path = scope_dir / f"hmm_fold_{fold:03d}.pkl"
        if not scaler_path.exists() or not hmm_path.exists():
            continue

        part = holdout_context.copy().reset_index(drop=True)
        scaler = joblib.load(scaler_path)
        part.loc[:, feature_columns] = scaler.transform(part[feature_columns])
        hmm = joblib.load(hmm_path)
        part = _add_regime_probs(part, hmm, cfg)
        dataset = _make_dataset(part, feature_columns, cfg)

        checkpoint = _load_torch_checkpoint(model_path, torch_device)
        model = _build_model(len(feature_columns), cfg).to(torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction = _predict_dataset(model, dataset, part, batch_size=batch_size, device=torch_device)
        prediction = prediction.loc[pd.to_datetime(prediction["timestamp"], utc=True) >= holdout_start].copy()
        if prediction.empty:
            continue
        prediction["split"] = "test"
        prediction["fold"] = fold
        prediction["model_fold"] = fold
        prediction["profile"] = profile
        rows.append(prediction)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_holdout_predictions(predictions: pd.DataFrame, *, profile: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    group_keys = ["timestamp"]
    first_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
            "forward_return",
            "tb_return",
            "hit_type",
            "4h_source_timestamp",
            "4h_available_timestamp",
        )
        if column in frame.columns
    ]
    aggregations: dict[str, Any] = {column: (column, "first") for column in first_columns}
    aggregations["prob_long"] = ("prob_long", "mean")
    aggregations["model_fold_count"] = ("model_fold", "nunique")
    for column in [column for column in frame.columns if column.startswith("regime_prob_")]:
        aggregations[column] = (column, "mean")
    out = frame.groupby(group_keys, as_index=False).agg(**aggregations)
    out["split"] = "test"
    out["fold"] = 0
    out["source_row_position"] = np.arange(len(out))
    out["profile"] = profile
    return out.sort_values("timestamp").reset_index(drop=True)


def _holdout_markdown(holdout_evaluation: pd.DataFrame, holdout_decision: dict[str, Any]) -> str:
    lines = ["# Holdout Evaluation", ""]
    if holdout_evaluation.empty:
        lines.append("No holdout evaluation was produced.")
        if holdout_decision:
            lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "mean_rank_ic",
        "mean_long_f1",
        "mean_prauc",
        "calibration_separation",
        "top_10_lift_global",
        "top_10_forward_return_global",
        "cv_policy_name",
        "cv_policy_lift_vs_base",
        "cv_policy_forward_return",
        "holdout_cv_threshold_f1",
        "holdout_cv_threshold_pred_long_rate",
        "holdout_cv_threshold_source",
        "holdout_policy_name",
        "holdout_policy_selection_rate",
        "holdout_policy_lift_vs_base",
        "holdout_policy_forward_return",
        "holdout_policy_pass",
        "holdout_policy_consistency_pass",
        "holdout_policy_consistency_reject_reason",
        "mtf_leakage_passed",
        "holdout_signal_pass",
        "holdout_signal_reject_reason",
        "holdout_threshold_pass",
        "holdout_threshold_reject_reason",
        "holdout_soft_pass",
        "holdout_reject_reason",
        "frozen_selection",
    ]
    visible = holdout_evaluation[[column for column in display_cols if column in holdout_evaluation.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
    return "\n".join(lines)


def _write_holdout_files(
    path: Path,
    *,
    holdout_evaluation: pd.DataFrame,
    holdout_score_bands: pd.DataFrame,
    holdout_thresholds: pd.DataFrame,
    holdout_decision: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    holdout_evaluation.to_csv(path / "holdout_evaluation.csv", index=False)
    holdout_score_bands.to_csv(path / "holdout_score_band_summary.csv", index=False)
    holdout_thresholds.to_csv(path / "holdout_threshold_summary.csv", index=False)
    policy_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_selection_rate",
            "cv_policy_precision",
            "cv_policy_f1",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_positive_forward_return_fold_rate",
            "cv_policy_pass",
            "cv_policy_reject_reason",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_source",
            "holdout_policy_selection_rate",
            "holdout_policy_precision",
            "holdout_policy_recall",
            "holdout_policy_f1",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_selection_rate_delta_vs_cv",
            "holdout_policy_precision_delta_vs_cv",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_policy_reject_reason",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
            "holdout_signal_pass",
            "holdout_signal_reject_reason",
            "holdout_threshold_pass",
            "holdout_threshold_reject_reason",
            "holdout_soft_pass",
            "holdout_reject_reason",
            "frozen_selection",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_evaluation = (
        holdout_evaluation[policy_columns].copy()
        if policy_columns
        else pd.DataFrame()
    )
    holdout_policy_evaluation.to_csv(path / "holdout_policy_evaluation.csv", index=False)
    consistency_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "frozen_selection",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_pass",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_signal_pass",
            "holdout_threshold_pass",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_consistency = (
        holdout_evaluation[consistency_columns].copy()
        if consistency_columns
        else pd.DataFrame()
    )
    holdout_policy_consistency.to_csv(path / "holdout_policy_consistency.csv", index=False)
    (path / "holdout_policy_consistency.md").write_text(
        _table_markdown("Holdout Policy Consistency", holdout_policy_consistency),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_policy_consistency.json",
        {"rows": holdout_policy_consistency.to_dict(orient="records")},
    )
    (path / "holdout_evaluation.md").write_text(
        _holdout_markdown(holdout_evaluation, holdout_decision),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_evaluation.json",
        {
            "decision": holdout_decision,
            "rows": holdout_evaluation.to_dict(orient="records"),
            "score_bands": holdout_score_bands.to_dict(orient="records"),
            "thresholds": holdout_thresholds.to_dict(orient="records"),
            "policy_evaluation": holdout_policy_evaluation.to_dict(orient="records"),
            "policy_consistency": holdout_policy_consistency.to_dict(orient="records"),
        },
    )


def _threshold_summary_value(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    matched = threshold_summary.loc[threshold_summary["metric"].astype(str) == metric, "mean"]
    if matched.empty:
        return np.nan
    try:
        return float(matched.iloc[0])
    except (TypeError, ValueError):
        return np.nan


def _cv_selected_threshold(entry: dict[str, Any] | None) -> tuple[float, str]:
    if not entry:
        return 0.5, "fallback_0.50_missing_cv_entry"
    diagnostics = entry.get("diagnostics", {}) or {}
    row = diagnostics.get("row", {}) or {}
    threshold = _float(row, "constrained_threshold_mean", np.nan)
    if np.isfinite(threshold):
        return threshold, "cv_constrained_threshold"
    threshold = _threshold_summary_value(diagnostics.get("threshold_summary"), "constrained_threshold")
    if np.isfinite(threshold):
        return threshold, "cv_constrained_threshold"
    threshold = _float(row, "selected_threshold_mean", np.nan)
    if np.isfinite(threshold):
        return threshold, "cv_selected_threshold"
    threshold = _threshold_summary_value(diagnostics.get("threshold_summary"), "selected_threshold")
    if np.isfinite(threshold):
        return threshold, "cv_selected_threshold"
    return 0.5, "fallback_0.50_missing_cv_threshold"


def _binary_metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = pd.to_numeric(scores, errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    y_pred = (y_score >= float(threshold)).astype(int)
    true_positive = float(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = float(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
    }


def _attach_holdout_cv_threshold_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    threshold, source = _cv_selected_threshold(cv_entry)
    metrics = _binary_metrics_at_threshold(predictions["label"], predictions["prob_long"], threshold)
    row["holdout_cv_threshold"] = float(threshold)
    row["holdout_cv_threshold_source"] = source
    row["holdout_cv_threshold_f1"] = metrics["f1"]
    row["holdout_cv_threshold_precision"] = metrics["precision"]
    row["holdout_cv_threshold_recall"] = metrics["recall"]
    row["holdout_cv_threshold_pred_long_rate"] = metrics["pred_long_rate"]
    return row


def _cv_score_policy(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {}
    diagnostics = entry.get("diagnostics", {}) or {}
    selection = diagnostics.get("score_policy_selection")
    if isinstance(selection, pd.DataFrame) and not selection.empty:
        return selection.iloc[0].to_dict()
    grid = diagnostics.get("score_policy_grid")
    if isinstance(grid, pd.DataFrame) and not grid.empty:
        chosen = select_score_policy(grid, entry.get("config", {}) or {})
        if not chosen.empty:
            return chosen.iloc[0].to_dict()
    return {}


def _policy_metrics_from_mask(predictions: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    selected = predictions.loc[mask].copy()
    base_long_rate = float(predictions["label"].mean()) if len(predictions) else np.nan
    if selected.empty:
        return {
            "selection_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "lift_vs_base": np.nan,
            "forward_return": np.nan,
        }
    true_positive = float(selected["label"].astype(int).sum())
    total_positive = float(predictions["label"].astype(int).sum())
    precision = float(selected["label"].mean())
    recall = true_positive / max(total_positive, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "selection_rate": float(len(selected) / len(predictions)) if len(predictions) else np.nan,
        "precision": precision,
        "recall": float(recall),
        "f1": float(f1),
        "lift_vs_base": float(precision / base_long_rate) if base_long_rate and base_long_rate > 0 else np.nan,
        "forward_return": float(selected["forward_return"].mean()) if "forward_return" in selected.columns else np.nan,
    }


def _evaluate_score_policy_on_holdout(
    predictions: pd.DataFrame,
    policy: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not policy:
        return {"source": "missing_cv_policy", "reject_reason": "missing_cv_policy"}
    policy_type = str(policy.get("policy_type", ""))
    policy_name = str(policy.get("policy_name", ""))
    if policy_type == "score_band":
        score_bins = int(_cfg(config, ["validation", "score_lift_bins"], _cfg(config, ["validation", "calibration_bins"], 10)))
        score_bands = _cfg(config, ["validation", "score_bands"], None)
        band_rows = score_band_diagnostics(predictions, bins=score_bins, bands=score_bands)
        matched = band_rows.loc[band_rows["band"].astype(str) == policy_name]
        if matched.empty:
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_holdout_band",
            }
        item = matched.iloc[0].to_dict()
        metrics = {
            "selection_rate": _float(item, "selection_rate"),
            "precision": _float(item, "actual_long_rate"),
            "recall": _float(item, "recall"),
            "f1": _float(item, "f1"),
            "lift_vs_base": _float(item, "lift_vs_base"),
            "forward_return": _float(item, "mean_forward_return"),
        }
    elif policy_type == "threshold_cap":
        threshold = _float(policy, "threshold_mean", np.nan)
        if not np.isfinite(threshold):
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_cv_threshold_mean",
            }
        mask = pd.to_numeric(predictions["prob_long"], errors="coerce") >= threshold
        metrics = _policy_metrics_from_mask(predictions, mask)
    else:
        return {
            "name": policy_name,
            "type": policy_type,
            "source": "cv_score_policy_selection",
            "reject_reason": "unknown_policy_type",
        }

    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    policy_cfg = _cfg(config, ["validation", "policy_selection"], {}) or {}
    max_selection_rate = float(policy_cfg.get("max_selection_rate", threshold_cfg.get("max_pred_long_rate", 0.70)))
    min_precision = float(policy_cfg.get("min_precision", threshold_cfg.get("min_precision", 0.30)))
    min_lift = float(policy_cfg.get("min_lift_vs_base", 1.0))
    min_forward_return = float(policy_cfg.get("min_forward_return", 0.0))
    reasons = []
    if metrics["selection_rate"] > max_selection_rate:
        reasons.append("selection_rate")
    if metrics["precision"] < min_precision:
        reasons.append("precision")
    if not np.isfinite(metrics["lift_vs_base"]) or metrics["lift_vs_base"] <= min_lift:
        reasons.append("lift_vs_base")
    if not np.isfinite(metrics["forward_return"]) or metrics["forward_return"] <= min_forward_return:
        reasons.append("forward_return")
    return {
        "name": policy_name,
        "type": policy_type,
        "source": "cv_score_policy_selection",
        **metrics,
        "pass": len(reasons) == 0,
        "reject_reason": ";".join(reasons),
    }


def _attach_holdout_policy_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = _cv_score_policy(cv_entry)
    row["cv_policy_name"] = str(policy.get("policy_name", ""))
    row["cv_policy_type"] = str(policy.get("policy_type", ""))
    row["cv_policy_selection_rate"] = _float(policy, "selection_rate")
    row["cv_policy_precision"] = _float(policy, "precision")
    row["cv_policy_recall"] = _float(policy, "recall")
    row["cv_policy_f1"] = _float(policy, "f1")
    row["cv_policy_lift_vs_base"] = _float(policy, "lift_vs_base")
    row["cv_policy_forward_return"] = _float(policy, "forward_return")
    row["cv_policy_positive_lift_fold_rate"] = _float(policy, "positive_lift_fold_rate")
    row["cv_policy_positive_forward_return_fold_rate"] = _float(policy, "positive_forward_return_fold_rate")
    row["cv_policy_pass"] = bool(policy.get("policy_pass", False))
    row["cv_policy_reject_reason"] = str(policy.get("policy_reject_reason", ""))
    metrics = _evaluate_score_policy_on_holdout(predictions, policy, config)
    row["holdout_policy_name"] = metrics.get("name", "")
    row["holdout_policy_type"] = metrics.get("type", "")
    row["holdout_policy_source"] = metrics.get("source", "")
    row["holdout_policy_selection_rate"] = metrics.get("selection_rate", np.nan)
    row["holdout_policy_precision"] = metrics.get("precision", np.nan)
    row["holdout_policy_recall"] = metrics.get("recall", np.nan)
    row["holdout_policy_f1"] = metrics.get("f1", np.nan)
    row["holdout_policy_lift_vs_base"] = metrics.get("lift_vs_base", np.nan)
    row["holdout_policy_forward_return"] = metrics.get("forward_return", np.nan)
    row["holdout_policy_pass"] = bool(metrics.get("pass", False))
    row["holdout_policy_reject_reason"] = metrics.get("reject_reason", "")
    return row


def _attach_holdout_policy_consistency(row: dict[str, Any]) -> dict[str, Any]:
    row["holdout_policy_selection_rate_delta_vs_cv"] = (
        _float(row, "holdout_policy_selection_rate") - _float(row, "cv_policy_selection_rate")
    )
    row["holdout_policy_precision_delta_vs_cv"] = (
        _float(row, "holdout_policy_precision") - _float(row, "cv_policy_precision")
    )
    row["holdout_policy_lift_delta_vs_cv"] = (
        _float(row, "holdout_policy_lift_vs_base") - _float(row, "cv_policy_lift_vs_base")
    )
    row["holdout_policy_forward_return_delta_vs_cv"] = (
        _float(row, "holdout_policy_forward_return") - _float(row, "cv_policy_forward_return")
    )

    reasons = []
    if not str(row.get("cv_policy_name", "")).strip():
        reasons.append("missing_cv_policy")
    if not bool(row.get("cv_policy_pass", False)):
        reasons.append("cv_policy")
    if str(row.get("cv_policy_name", "")) != str(row.get("holdout_policy_name", "")):
        reasons.append("policy_name_mismatch")
    if str(row.get("cv_policy_type", "")) != str(row.get("holdout_policy_type", "")):
        reasons.append("policy_type_mismatch")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if not bool(row.get("holdout_signal_pass", False)):
        reasons.append("holdout_signal")
    row["holdout_policy_consistency_pass"] = len(reasons) == 0
    row["holdout_policy_consistency_reject_reason"] = ";".join(reasons)
    return row


def _holdout_signal_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    reasons = []
    if float(row.get("mean_rank_ic", 0.0)) <= target_rank_ic:
        reasons.append("mean_rank_ic")
    if float(row.get("top_10_lift_global", 0.0)) <= 1.0:
        reasons.append("top_10_lift_global")
    if float(row.get("top_10_forward_return_global", 0.0)) <= 0.0:
        reasons.append("top_10_forward_return_global")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if float(row.get("calibration_separation", 0.0)) <= 0.0:
        reasons.append("calibration_separation")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", True)):
        reasons.append("stationarity_policy")
    return reasons


def _holdout_threshold_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    reasons = []
    if float(row.get("holdout_cv_threshold_f1", 0.0)) <= min_long_f1:
        reasons.append("holdout_cv_threshold_f1")
    if float(row.get("holdout_cv_threshold_pred_long_rate", 1.0)) > max_pred_long_rate:
        reasons.append("holdout_cv_threshold_pred_long_rate")
    return reasons


def _attach_holdout_soft_pass(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    signal_reasons = _holdout_signal_pass_reasons(row, config)
    threshold_reasons = _holdout_threshold_pass_reasons(row, config)
    reasons = [*signal_reasons, *threshold_reasons]
    row["holdout_signal_pass"] = len(signal_reasons) == 0
    row["holdout_signal_reject_reason"] = ";".join(signal_reasons)
    row["holdout_threshold_pass"] = len(threshold_reasons) == 0
    row["holdout_threshold_reject_reason"] = ";".join(threshold_reasons)
    row["holdout_soft_pass"] = len(reasons) == 0
    row["holdout_reject_reason"] = ";".join(reasons)
    return row


def _evaluate_holdout_candidates(
    *,
    profile_entries: list[dict[str, Any]],
    cv_blend_entries: list[dict[str, Any]] | None = None,
    settings: dict[str, Any],
    config: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    holdout_context, holdout_start = _read_holdout_context(settings, config)
    if holdout_context.empty or holdout_start is None:
        holdout_decision = {
            "available": False,
            "reason": "missing_holdout_frame_or_metadata",
            "policy": "holdout result must remain separate from profile selection",
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), holdout_decision, []

    full_entries = [
        entry
        for entry in profile_entries
        if str(entry.get("fold_scope", "")) == "full"
        and not str(entry.get("profile", "")).startswith("blend_")
    ]
    holdout_entries: list[dict[str, Any]] = []
    score_band_rows = []
    threshold_rows = []
    evaluation_rows = []
    cv_entry_by_profile = {
        str(entry.get("profile", "")): entry
        for entry in [*profile_entries, *(cv_blend_entries or [])]
        if str(entry.get("fold_scope", "")) == "full" or str(entry.get("fold_scope", "")).startswith("blend_")
    }

    for entry in full_entries:
        scope_dir = Path(entry["scope_dir"])
        manifest = _read_json(scope_dir / "training_manifest.json")
        raw_predictions = _predict_holdout_for_profile(
            scope_dir=scope_dir,
            manifest=manifest,
            holdout_context=holdout_context,
            holdout_start=holdout_start,
            config=config,
        )
        predictions = _aggregate_holdout_predictions(raw_predictions, profile=str(entry["profile"]))
        if predictions.empty:
            continue
        diagnostics = summarize_profile_predictions(
            predictions,
            config,
            profile=str(entry["profile"]),
            feature_columns=list(entry["feature_columns"]),
            fold_scope="holdout_profile",
        )
        row = dict(diagnostics["row"])
        row["candidate"] = str(entry["profile"])
        row["candidate_type"] = "profile"
        row["source_profiles"] = str(entry["profile"])
        row["blend_method"] = ""
        row["blend_weights"] = ""
        cv_entry = cv_entry_by_profile.get(str(entry["profile"]))
        row = _attach_holdout_cv_threshold_metrics(row, predictions, cv_entry)
        row = _attach_holdout_policy_metrics(row, predictions, cv_entry, config)
        row = _attach_holdout_soft_pass(row, config)
        row = _attach_holdout_policy_consistency(row)
        evaluation_rows.append(row)
        bands = diagnostics["score_band_summary"].copy()
        if not bands.empty:
            bands.insert(0, "candidate", row["candidate"])
            score_band_rows.append(bands)
        thresholds = diagnostics["threshold_summary"].copy()
        if not thresholds.empty:
            thresholds.insert(0, "candidate", row["candidate"])
            threshold_rows.append(thresholds)
        holdout_entries.append(
            {
                "profile": str(entry["profile"]),
                "fold_scope": "holdout_profile",
                "feature_columns": list(entry["feature_columns"]),
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": row,
                "config": entry.get("config", config),
            }
        )

    blend_source_entries = [{**entry, "fold_scope": "full"} for entry in holdout_entries]
    blend_entries = _profile_blend_entries(blend_source_entries, config)
    for entry in blend_entries:
        diagnostics = entry["diagnostics"]
        row = dict(diagnostics["row"])
        row["candidate"] = str(entry["profile"])
        row["candidate_type"] = "blend"
        row["source_profiles"] = row.get("blend_profiles", "")
        row["blend_method"] = row.get("blend_method", "")
        row["blend_weights"] = row.get("blend_weights", "")
        cv_entry = cv_entry_by_profile.get(str(entry["profile"]))
        row = _attach_holdout_cv_threshold_metrics(row, entry["predictions"], cv_entry)
        row = _attach_holdout_policy_metrics(row, entry["predictions"], cv_entry, config)
        row = _attach_holdout_soft_pass(row, config)
        row = _attach_holdout_policy_consistency(row)
        evaluation_rows.append(row)
        bands = diagnostics["score_band_summary"].copy()
        if not bands.empty:
            bands.insert(0, "candidate", row["candidate"])
            score_band_rows.append(bands)
        thresholds = diagnostics["threshold_summary"].copy()
        if not thresholds.empty:
            thresholds.insert(0, "candidate", row["candidate"])
            threshold_rows.append(thresholds)

    holdout_evaluation = pd.DataFrame(evaluation_rows)
    holdout_score_bands = pd.concat(score_band_rows, ignore_index=True) if score_band_rows else pd.DataFrame()
    holdout_thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    if holdout_evaluation.empty:
        holdout_decision = {
            "available": False,
            "reason": "no_holdout_predictions",
            "policy": "holdout result must remain separate from profile selection",
        }
        return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries

    frozen_selection = str(settings.get("control_profile", ""))
    best_blend = decision.get("best_profile_blend") or {}
    best_candidate = decision.get("best_candidate") or {}
    if best_blend:
        frozen_selection = str(best_blend.get("profile") or frozen_selection)
    elif best_candidate:
        frozen_selection = str(best_candidate.get("profile") or frozen_selection)
    holdout_evaluation["frozen_selection"] = holdout_evaluation["candidate"].astype(str).eq(frozen_selection)

    sortable = holdout_evaluation.copy()
    sortable["signal_pass_sort"] = sortable["holdout_signal_pass"].astype(bool).astype(int)
    sortable["threshold_pass_sort"] = sortable["holdout_threshold_pass"].astype(bool).astype(int)
    sortable = sortable.sort_values(
        [
            "signal_pass_sort",
            "threshold_pass_sort",
            "mean_rank_ic",
            "top_10_lift_global",
            "holdout_cv_threshold_f1",
        ],
        ascending=[False, False, False, False, False],
    )
    observed_best = sortable.iloc[0].to_dict()
    frozen_rows = holdout_evaluation.loc[holdout_evaluation["frozen_selection"].astype(bool)]
    frozen_row = frozen_rows.iloc[0].to_dict() if not frozen_rows.empty else {}
    policy_sortable = holdout_evaluation.copy()
    policy_sortable["policy_consistency_sort"] = policy_sortable["holdout_policy_consistency_pass"].astype(bool).astype(int)
    policy_sortable = policy_sortable.sort_values(
        [
            "policy_consistency_sort",
            "holdout_policy_forward_return",
            "holdout_policy_lift_vs_base",
            "mean_rank_ic",
        ],
        ascending=[False, False, False, False],
    )
    observed_best_policy = policy_sortable.iloc[0].to_dict()
    observed_best_name = str(observed_best.get("candidate", ""))
    observed_best_warning = ""
    if observed_best_name and observed_best_name != frozen_selection:
        observed_best_warning = (
            "Observed-best holdout candidate is diagnostic only; do not promote it "
            "or tune blend weights against this same reserved holdout."
        )
    observed_best_policy_name = str(observed_best_policy.get("candidate", ""))
    observed_best_policy_warning = ""
    if observed_best_policy_name and observed_best_policy_name != frozen_selection:
        observed_best_policy_warning = (
            "Observed-best holdout policy candidate is diagnostic only; keep the frozen "
            "pre-holdout selection unless a future out-of-sample window confirms it."
        )
    if frozen_row and bool(frozen_row.get("holdout_policy_consistency_pass", False)):
        score_policy_recommendation = "review_frozen_score_band_policy"
    elif bool(observed_best_policy.get("holdout_policy_consistency_pass", False)):
        score_policy_recommendation = "holdout_only_diagnostic_policy_candidate"
    else:
        score_policy_recommendation = "keep_control_profile"
    holdout_decision = {
        "available": True,
        "policy": "one_shot_final_validation; do not tune profiles or weights against this same holdout",
        "holdout_start": str(pd.to_datetime(holdout_start, utc=True)),
        "holdout_rows": int(len(holdout_context.loc[pd.to_datetime(holdout_context["timestamp"], utc=True) >= holdout_start])),
        "candidate_count": int(len(holdout_evaluation)),
        "frozen_selection": frozen_selection,
        "frozen_selection_metrics": _json_ready(frozen_row),
        "frozen_policy_validation": _json_ready(frozen_row),
        "observed_best_holdout_candidate": _json_ready(observed_best),
        "observed_best_holdout_warning": observed_best_warning,
        "observed_best_policy_candidate": _json_ready(observed_best_policy),
        "observed_best_policy_warning": observed_best_policy_warning,
        "score_policy_recommendation": score_policy_recommendation,
    }
    return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries


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


def _write_seed_audit_files(path: Path, seed_audit: pd.DataFrame, seed_stability: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    seed_audit.to_csv(path / "seed_audit.csv", index=False)
    seed_stability.to_csv(path / "seed_stability.csv", index=False)
    (path / "seed_audit.md").write_text(_seed_audit_markdown(seed_audit, seed_stability), encoding="utf-8")
    _write_json(path / "seed_audit.json", {"rows": seed_audit.to_dict(orient="records")})
    _write_json(path / "seed_stability.json", {"rows": seed_stability.to_dict(orient="records")})


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
            "test_f1_at_constrained_threshold",
            _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
        ) - _metric_or(
            control,
            "test_f1_at_constrained_threshold",
            _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
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


def _write_experiment_bundle(
    *,
    output_dir: Path,
    run_id: str,
    report_dir: Path,
    zip_paths: list[str],
) -> tuple[Path, Path]:
    bundle_path = output_dir / f"phase1_experiment_bundle_{run_id}.zip"
    latest_path = output_dir / "phase1_latest_experiment_bundle.zip"
    summary_files = [
        "profile_comparison.csv",
        "profile_comparison.md",
        "profile_delta_vs_control.csv",
        "seed_audit.csv",
        "seed_audit.md",
        "seed_audit.json",
        "seed_stability.csv",
        "seed_stability.json",
        "seed_ensemble.csv",
        "seed_ensemble.md",
        "seed_ensemble.json",
        "profile_blend.csv",
        "profile_blend.md",
        "profile_blend.json",
        "profile_fold_metrics.csv",
        "profile_threshold_summary.csv",
        "profile_threshold_grid_summary.csv",
        "profile_score_band_summary.csv",
        "profile_score_policy_grid.csv",
        "profile_score_policy_selection.csv",
        "profile_feature_groups.csv",
        "experiment_selection.csv",
        "experiment_selection.md",
        "experiment_selection.json",
        "missing_selected_profiles.csv",
        "missing_selected_profiles.md",
        "missing_selected_profiles.json",
        "holdout_reservation.csv",
        "holdout_reservation.md",
        "holdout_reservation.json",
        "holdout_evaluation.csv",
        "holdout_evaluation.md",
        "holdout_evaluation.json",
        "holdout_score_band_summary.csv",
        "holdout_threshold_summary.csv",
        "holdout_policy_evaluation.csv",
        "holdout_policy_consistency.csv",
        "holdout_policy_consistency.md",
        "holdout_policy_consistency.json",
        "decision_report.json",
        "best_candidate.json",
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in summary_files:
            path = report_dir / filename
            if path.exists():
                archive.write(path, f"{run_id}/{filename}")
        for item in zip_paths:
            path = Path(item)
            if path.exists():
                archive.write(path, f"{run_id}/diagnostics/{path.name}")
    shutil.copyfile(bundle_path, latest_path)
    return bundle_path, latest_path


def _write_experiment_slim_bundle(*, output_dir: Path, run_id: str, report_dir: Path) -> tuple[Path, Path]:
    slim_path = output_dir / f"phase1_experiment_slim_bundle_{run_id}.zip"
    latest_slim_path = output_dir / "phase1_latest_experiment_slim_bundle.zip"
    with zipfile.ZipFile(slim_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(report_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
                archive.write(path, f"{run_id}/{path.name}")
    shutil.copyfile(slim_path, latest_slim_path)
    return slim_path, latest_slim_path


def run_experiment_matrix(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    run_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    settings = experiment_settings(config)
    settings = _resolve_holdout_settings(settings, config)
    frame = _selection_frame_before_holdout(frame, settings)
    settings = _preflight_experiment_profiles(settings, frame, config)
    signature = _experiment_signature(config, settings)
    signature_hash = _hash_payload(signature)
    run_id, run_id_source = resolve_experiment_run_id(checkpoint_dir, config, settings, run_id)
    run_dir = experiment_root(checkpoint_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "experiment_manifest.json",
        {
            "run_id": run_id,
            "run_id_source": run_id_source,
            "signature": signature,
            "signature_hash": signature_hash,
            "settings": settings,
        },
    )
    experiment_selection = _write_experiment_selection(run_dir, settings)
    holdout_reservation = _write_holdout_reservation(run_dir, settings)

    triage_fold_ids = [int(fold_id) for fold_id in settings.get("triage_fold_ids", [])]
    resume_existing = bool(settings.get("resume_existing", True))
    force_retrain = bool(settings.get("force_retrain", False))
    rows: list[dict[str, Any]] = []
    profile_results = []
    for profile in settings["profiles"]:
        result = run_profile_experiment(
            frame,
            config,
            profile=profile,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            fold_scope="triage",
            fold_ids=triage_fold_ids or None,
            resume_existing=resume_existing,
            force_retrain=force_retrain,
            device=device,
        )
        profile_results.append(result)
        rows.append(result["summary"])

    triage_rows = _decision_rows(rows, config, scope="triage")
    full_profiles_setting = settings.get("full_cv_profiles", "auto")
    if full_profiles_setting == "auto":
        full_profiles = _auto_full_profiles(settings, triage_rows)
    else:
        full_profiles = [str(profile) for profile in full_profiles_setting]

    full_rows = []
    for profile in dict.fromkeys(full_profiles):
        result = run_profile_experiment(
            frame,
            config,
            profile=profile,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            fold_scope="full",
            fold_ids=None,
            resume_existing=resume_existing,
            force_retrain=force_retrain,
            device=device,
        )
        profile_results.append(result)
        full_rows.append(result["summary"])

    full_rows = _decision_rows(full_rows, config, scope="full") if full_rows else []
    comparison = _comparison_frame([*triage_rows, *full_rows])
    seed_results: list[dict[str, Any]] = []
    seed_audit_cfg = settings.get("seed_audit", {}) or {}
    if bool(seed_audit_cfg.get("enabled", False)):
        audit_profiles = [str(profile) for profile in seed_audit_cfg.get("profiles", []) or [settings["control_profile"]]]
        audit_seeds = [int(seed) for seed in seed_audit_cfg.get("seeds", []) or []]
        audit_fold_ids = seed_audit_cfg.get("fold_ids", triage_fold_ids)
        audit_fold_ids = [int(fold_id) for fold_id in audit_fold_ids] if audit_fold_ids else None
        for profile in audit_profiles:
            for seed in audit_seeds:
                seed_cfg = copy.deepcopy(config)
                _set_cfg(seed_cfg, ["project", "random_seed"], seed)
                result = run_profile_experiment(
                    frame,
                    seed_cfg,
                    profile=profile,
                    checkpoint_dir=checkpoint_dir,
                    run_id=run_id,
                    fold_scope=_seed_audit_scope(seed),
                    fold_ids=audit_fold_ids,
                    resume_existing=resume_existing,
                    force_retrain=force_retrain,
                    device=device,
                )
                result["summary"]["seed"] = seed
                seed_results.append(result)
    seed_ensemble_results = _seed_ensemble_entries(seed_results, config)
    profile_blend_results = _profile_blend_entries(profile_results, config)
    all_results = [*profile_results, *seed_results, *seed_ensemble_results, *profile_blend_results]
    seed_audit, seed_stability = _seed_audit_entries_to_frames(all_results)
    seed_ensemble = _seed_ensemble_frame(all_results)
    profile_blend = _profile_blend_frame(all_results)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, config, settings["control_profile"])
    _write_seed_audit_files(run_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    profile_delta = _profile_delta_vs_control(profile_results, settings["control_profile"])
    best = _best_candidate(comparison, settings["control_profile"])
    blend_leaders = _profile_blend_leaders(profile_blend)
    best_blend = _best_profile_blend(profile_blend)
    missing_selected = _missing_selected_profiles(experiment_selection, comparison)
    _write_missing_selected_profiles(run_dir, missing_selected)
    decision = {
        "run_id": run_id,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "full_profiles": full_profiles,
        "seed_audit_profiles": [str(profile) for profile in seed_audit_cfg.get("profiles", [])] if seed_audit_cfg else [],
        "seed_audit_seeds": [int(seed) for seed in seed_audit_cfg.get("seeds", [])] if seed_audit_cfg else [],
        "skipped_profiles": settings.get("skipped_profiles", []) or [],
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout": settings.get("holdout", {}) or {},
        "recommendation": "fix_missing_selected_profiles"
        if not missing_selected.empty
        else ("promote_best_candidate" if best else ("review_profile_blend" if best_blend else "keep_control_profile")),
    }
    _write_decision_files(run_dir, comparison, decision)
    _write_profile_delta(run_dir, profile_delta)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "profile_results": all_results,
        "comparison": comparison,
        "profile_delta": profile_delta,
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "missing_selected_profiles": missing_selected,
        "decision": decision,
    }


def _profile_dirs(run_dir: Path) -> list[Path]:
    paths = []
    for profile_dir in run_dir.iterdir():
        if not profile_dir.is_dir():
            continue
        for scope_dir in profile_dir.iterdir():
            if (scope_dir / "training_manifest.json").exists() and (scope_dir / "predictions_all.parquet").exists():
                paths.append(scope_dir)
    return sorted(paths)


def write_experiment_diagnostics(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_dir = experiment_root(checkpoint_dir) / run_id if run_id else latest_experiment_run(checkpoint_dir)
    run_manifest_path = run_dir / "experiment_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    settings = copy.deepcopy(run_manifest.get("settings") or experiment_settings(config))
    settings = _resolve_holdout_settings(settings, config)
    diagnostic_config = copy.deepcopy(config)
    experiment_cfg = copy.deepcopy(_cfg(diagnostic_config, ["experiments"], default={}) or {})
    experiment_cfg.update(settings)
    _set_cfg(diagnostic_config, ["experiments"], experiment_cfg)
    scope_dirs = _profile_dirs(run_dir)
    if not scope_dirs:
        root = experiment_root(checkpoint_dir)
        recent_runs = sorted([path.name for path in root.glob("*") if path.is_dir()], reverse=True)[:8]
        hint = (
            f"No completed profile runs found under {run_dir}.\n"
            "This usually means notebook `04_training_walk_forward.ipynb` has not finished (or wrote to a different CHECKPT_DIR).\n"
            f"Expected files like: {run_dir}/<profile>/<fold_scope>/training_manifest.json and predictions_all.parquet.\n"
            f"Recent experiment run directories: {recent_runs}"
        )
        raise FileNotFoundError(hint)

    entries = []
    for scope_dir in scope_dirs:
        manifest = _read_json(scope_dir / "training_manifest.json")
        profile = str(manifest["profile"])
        fold_scope = str(manifest["fold_scope"])
        feature_columns = list(manifest["feature_columns"])
        predictions = pd.read_parquet(scope_dir / "predictions_all.parquet")
        diagnostics = summarize_profile_predictions(
            predictions,
            diagnostic_config,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope=fold_scope,
        )
        entries.append(
            {
                "scope_dir": scope_dir,
                "profile": profile,
                "fold_scope": fold_scope,
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
            }
        )
    profile_entries = list(entries)
    seed_ensemble_entries = _seed_ensemble_entries(profile_entries, diagnostic_config)
    profile_blend_entries = _profile_blend_entries(profile_entries, diagnostic_config)
    entries = [*profile_entries, *seed_ensemble_entries, *profile_blend_entries]

    rows = [entry["diagnostics"]["row"] for entry in entries]
    triage_rows = _decision_rows(
        [row for row in rows if row.get("fold_scope") == "triage"],
        diagnostic_config,
        scope="triage",
    )
    full_rows = _decision_rows(
        [row for row in rows if row.get("fold_scope") == "full"],
        diagnostic_config,
        scope="full",
    )
    comparison = _comparison_frame([*triage_rows, *full_rows])
    profile_delta = _profile_delta_vs_control(profile_entries, settings["control_profile"])
    seed_audit, seed_stability = _seed_audit_entries_to_frames(entries)
    seed_ensemble = _seed_ensemble_frame(entries)
    profile_blend = _profile_blend_frame(entries)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, diagnostic_config, settings["control_profile"])
    decision_lookup = {
        (str(row["profile"]), str(row["fold_scope"])): row
        for row in [*triage_rows, *full_rows]
    }

    zip_paths = []
    for entry in entries:
        profile = str(entry["profile"])
        fold_scope = str(entry["fold_scope"])
        feature_columns = list(entry["feature_columns"])
        predictions = entry["predictions"]
        diagnostics = entry["diagnostics"]
        entry_config = entry.get("config", config)
        decided = decision_lookup.get((profile, fold_scope), {})
        ledger = diagnostics["ledger"].copy()
        for column in ("promotable", "reject_reason"):
            if column in decided:
                ledger.loc[:, column] = decided[column]
        zip_path = write_phase1_diagnostic_bundle(
            output_dir=Path(output_dir) / "experiments" / run_dir.name / _slug(profile) / fold_scope,
            report=diagnostics["report"],
            predictions=_test_predictions(predictions),
            calibration=diagnostics["calibration"],
            calibrated_report=diagnostics.get("calibrated_report"),
            calibrated_calibration=diagnostics.get("calibrated_calibration"),
            calibrated_predictions=diagnostics.get("calibrated_predictions"),
            fold_metrics=diagnostics["fold_metrics"],
            regime_metrics=diagnostics["regime_metrics"],
            regime_by_fold=diagnostics["regime_by_fold"],
            bad_fold_regime=diagnostics["bad_fold_regime"],
            threshold_metrics=diagnostics["threshold_metrics"],
            threshold_summary=diagnostics["threshold_summary"],
            mtf_leakage=diagnostics["mtf_leakage"],
            stationarity_policy=diagnostics["stationarity_policy"],
            score_lift=diagnostics["score_lift"],
            score_lift_by_fold=diagnostics["score_lift_by_fold"],
            score_band_lift=diagnostics["score_band_lift"],
            score_band_by_fold=diagnostics["score_band_by_fold"],
            score_band_summary=diagnostics["score_band_summary"],
            recent_fold_summary=diagnostics["recent_fold_summary"],
            feature_groups=diagnostics["feature_groups"],
            feature_profile=diagnostics["feature_profile"],
            experiment_ledger=ledger,
            model_feature_columns=feature_columns,
            config=profile_config(entry_config, profile),
            prefix=f"phase1_diagnostics_{_slug(profile)}_{fold_scope}",
        )
        zip_paths.append(str(zip_path))

    best = _best_candidate(comparison, settings["control_profile"])
    blend_leaders = _profile_blend_leaders(profile_blend)
    best_blend = _best_profile_blend(profile_blend)
    report_dir = Path(output_dir) / "experiments" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    experiment_selection = _write_experiment_selection(report_dir, settings)
    holdout_reservation = _write_holdout_reservation(report_dir, settings)
    missing_selected = _missing_selected_profiles(experiment_selection, comparison)
    _write_missing_selected_profiles(report_dir, missing_selected)
    decision = {
        "run_id": run_dir.name,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "full_profiles": [str(profile) for profile in settings.get("always_full_profiles", [])],
        "seed_audit_profiles": [
            str(profile) for profile in (settings.get("seed_audit", {}) or {}).get("profiles", [])
        ],
        "seed_audit_seeds": [int(seed) for seed in (settings.get("seed_audit", {}) or {}).get("seeds", [])],
        "skipped_profiles": settings.get("skipped_profiles", []) or [],
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout": settings.get("holdout", {}) or {},
        "recommendation": "fix_missing_selected_profiles"
        if not missing_selected.empty
        else ("promote_best_candidate" if best else ("review_profile_blend" if best_blend else "keep_control_profile")),
        "diagnostic_zips": zip_paths,
    }
    holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries = (
        _evaluate_holdout_candidates(
            profile_entries=profile_entries,
            cv_blend_entries=profile_blend_entries,
            settings=settings,
            config=diagnostic_config,
            decision=decision,
        )
    )
    decision["holdout_evaluation"] = holdout_decision
    decision["holdout_evaluation_available"] = bool(holdout_decision.get("available", False))
    bundle_path = Path(output_dir) / f"phase1_experiment_bundle_{run_dir.name}.zip"
    latest_bundle_path = Path(output_dir) / "phase1_latest_experiment_bundle.zip"
    slim_bundle_path = Path(output_dir) / f"phase1_experiment_slim_bundle_{run_dir.name}.zip"
    latest_slim_bundle_path = Path(output_dir) / "phase1_latest_experiment_slim_bundle.zip"
    decision["bundle_zip"] = str(bundle_path)
    decision["latest_bundle_zip"] = str(latest_bundle_path)
    decision["slim_bundle_zip"] = str(slim_bundle_path)
    decision["latest_slim_bundle_zip"] = str(latest_slim_bundle_path)
    _write_decision_files(report_dir, comparison, decision)
    _write_profile_delta(report_dir, profile_delta)
    _write_seed_audit_files(report_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(report_dir, seed_ensemble)
    _write_profile_blend_files(report_dir, profile_blend)
    _write_profile_diagnostic_summaries(report_dir, entries)
    _write_holdout_files(
        report_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
    )
    _write_decision_files(run_dir, comparison, decision)
    _write_profile_delta(run_dir, profile_delta)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_profile_diagnostic_summaries(run_dir, entries)
    _write_experiment_selection(run_dir, settings)
    _write_holdout_reservation(run_dir, settings)
    _write_missing_selected_profiles(run_dir, missing_selected)
    _write_holdout_files(
        run_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
    )
    bundle_path, latest_bundle_path = _write_experiment_bundle(
        output_dir=Path(output_dir),
        run_id=run_dir.name,
        report_dir=report_dir,
        zip_paths=zip_paths,
    )
    slim_bundle_path, latest_slim_bundle_path = _write_experiment_slim_bundle(
        output_dir=Path(output_dir),
        run_id=run_dir.name,
        report_dir=report_dir,
    )
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "comparison": comparison,
        "profile_delta": profile_delta,
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "holdout_evaluation": holdout_evaluation,
        "holdout_score_bands": holdout_score_bands,
        "holdout_thresholds": holdout_thresholds,
        "holdout_entries": holdout_entries,
        "missing_selected_profiles": missing_selected,
        "decision": decision,
        "zip_paths": zip_paths,
        "bundle_zip": str(bundle_path),
        "latest_bundle_zip": str(latest_bundle_path),
        "slim_bundle_zip": str(slim_bundle_path),
        "latest_slim_bundle_zip": str(latest_slim_bundle_path),
    }
