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

import numpy as np
import pandas as pd

from yenibot.diagnostics import (
    attach_threshold_summary_to_phase1_report,
    calibration_table,
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
    score_lift_by_fold_diagnostics,
    score_lift_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)
from yenibot.features import filter_feature_columns, resolve_feature_profile, select_feature_columns
from yenibot.training import run_walk_forward_training


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
    threshold_metrics = threshold_diagnostics(predictions)
    threshold_summary = threshold_summary_diagnostics(threshold_metrics)
    report = attach_threshold_summary_to_phase1_report(report, threshold_summary, profile_cfg)
    score_bins = int(_cfg(profile_cfg, ["validation", "score_lift_bins"], _cfg(profile_cfg, ["validation", "calibration_bins"], 10)))
    score_bands = _cfg(profile_cfg, ["validation", "score_bands"], None)
    score_lift = score_lift_diagnostics(test_predictions, bins=score_bins)
    score_lift_by_fold = score_lift_by_fold_diagnostics(test_predictions, bins=score_bins)
    score_band_lift = score_band_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_by_fold = score_band_by_fold_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_summary = score_band_summary_diagnostics(score_band_by_fold)
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
        "fold_metrics": fold_metrics,
        "regime_metrics": regime_metrics,
        "regime_by_fold": regime_by_fold,
        "bad_fold_regime": bad_fold_regime,
        "threshold_metrics": threshold_metrics,
        "threshold_summary": threshold_summary,
        "score_lift": score_lift,
        "score_lift_by_fold": score_lift_by_fold,
        "score_band_lift": score_band_lift,
        "score_band_by_fold": score_band_by_fold,
        "score_band_summary": score_band_summary,
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
    selected_f1 = _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1"))
    control_selected_f1 = _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1"))
    selected_f1_floor = _optional_gate_float(gates, "min_selected_threshold_f1", None)
    if selected_f1_floor is not None and selected_f1 < selected_f1_floor:
        reasons.append("selected_threshold_f1")
    selected_f1_delta = _optional_gate_float(gates, "min_selected_threshold_f1_delta", None)
    if selected_f1_delta is not None and selected_f1 < control_selected_f1 + selected_f1_delta:
        reasons.append("selected_threshold_f1_delta")
    mean_long_f1_delta = _optional_gate_float(gates, "min_long_f1_delta", None)
    if mean_long_f1_delta is not None and _float(row, "mean_long_f1") < _float(control, "mean_long_f1") + mean_long_f1_delta:
        reasons.append("mean_long_f1_delta")
    if _float(row, "top_10_lift_global") < _float(control, "top_10_lift_global") + float(gates.get("min_top_10_lift_global_delta", 0.05)):
        reasons.append("top_10_lift_global_delta")
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
        "selected_threshold_mean",
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
            "top_10_lift_global",
            "top_10_bad_fold_lift_mean",
            "passed_phase1_selected_threshold",
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
    columns = ["profile", "role", "selected", "skip_reason"]
    rows: list[dict[str, Any]] = [
        {
            "profile": str(settings["control_profile"]),
            "role": "control_profile",
            "selected": True,
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
        for profile in values or []:
            rows.append({"profile": str(profile), "role": role, "selected": True, "skip_reason": ""})
    for skipped in settings.get("skipped_profiles", []) or []:
        rows.append(
            {
                "profile": str(skipped.get("profile", "")),
                "role": str(skipped.get("role", "skipped_profile")),
                "selected": False,
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
    lines.append("| profile | role | selected | skip_reason |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in selection.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(bool(row["selected"])),
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
        frame = frame.merge(thresholds[["fold", "test_f1_at_selected_threshold"]], on="fold", how="left")
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


def _profile_blend_predictions(entries: list[dict[str, Any]], *, method: str) -> pd.DataFrame:
    if len(entries) < 2:
        return pd.DataFrame()

    frames = []
    profiles = []
    for entry in entries:
        prediction = entry["predictions"].copy()
        profile = str(entry["profile"])
        prediction["_blend_profile"] = profile
        if method == "rank_mean":
            prediction["_blend_score"] = _rank_score_by_fold(prediction)
        elif method == "prob_mean":
            prediction["_blend_score"] = prediction["prob_long"].astype(float)
        else:
            raise ValueError(f"Unknown profile blend method: {method}")
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
    stats = grouped["_blend_score"].agg(
        prob_long_blend="mean",
        prob_long_profile_std="std",
        prob_long_profile_min="min",
        prob_long_profile_max="max",
        blend_profile_count="count",
    ).reset_index()
    stats = stats.loc[stats["blend_profile_count"] == profile_count].copy()
    if stats.empty:
        return pd.DataFrame()

    base = grouped.first().reset_index()
    drop_columns = ["_blend_profile", "_blend_score", "prob_long_blend"]
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
    base["blend_profiles"] = ",".join(sorted(set(profiles)))
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

    combos = list(combinations(full_entries, 2))
    if len(full_entries) > 2:
        combos.append(tuple(full_entries))

    blend_entries = []
    for combo in combos:
        profiles = [str(entry["profile"]) for entry in combo]
        feature_columns = sorted({column for entry in combo for column in entry.get("feature_columns", [])})
        combo_hash = _hash_payload({"profiles": profiles})[:10]
        for method in ("prob_mean", "rank_mean"):
            predictions = _profile_blend_predictions(list(combo), method=method)
            if predictions.empty:
                continue
            profile = f"blend_{method}_{combo_hash}"
            blend_cfg = _blend_entry_config(
                config,
                profile,
                feature_columns,
                description=f"Diagnostic {method} blend of: {', '.join(profiles)}",
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

    rows = []
    for _, item in reviewed.iterrows():
        row = item.to_dict()
        row["control_profile"] = control_profile
        row["mean_rank_ic_delta_vs_control"] = _float(row, "mean_rank_ic") - _float(control, "mean_rank_ic")
        row["std_rank_ic_delta_vs_control"] = _float(row, "std_rank_ic") - _float(control, "std_rank_ic")
        row["positive_ic_fraction_delta_vs_control"] = _float(row, "positive_ic_fraction") - _float(control, "positive_ic_fraction")
        row["top_10_lift_global_delta_vs_control"] = _float(row, "top_10_lift_global") - _float(control, "top_10_lift_global")
        row["selected_threshold_f1_delta_vs_control"] = _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")) - _metric_or(
            control,
            "test_f1_at_selected_threshold",
            _float(control, "mean_long_f1"),
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
        row["tail_lift_eligible"] = not tail_lift_reasons
        row["tail_lift_reason"] = ";".join(tail_lift_reasons)
        row["stability_eligible"] = not stability_reasons
        row["stability_reason"] = ";".join(stability_reasons)
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
        "tail_lift_leader": _select_profile_blend_leader(profile_blend, "tail_lift"),
        "stability_leader": _select_profile_blend_leader(profile_blend, "stability"),
    }
    return {key: value for key, value in leaders.items() if value}


def _mark_profile_blend_leaders(profile_blend: pd.DataFrame) -> pd.DataFrame:
    if profile_blend.empty:
        return profile_blend
    marked = profile_blend.copy()
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
        if bool(row.get("tail_lift_leader", False)):
            item_roles.append("tail_lift")
        if bool(row.get("stability_leader", False)):
            item_roles.append("stability")
        roles.append(",".join(item_roles))
    marked["leader_roles"] = roles
    return marked


def _best_profile_blend(profile_blend: pd.DataFrame) -> dict[str, Any]:
    leaders = _profile_blend_leaders(profile_blend)
    return leaders.get("tail_lift_leader") or leaders.get("stability_leader") or {}


def _profile_blend_markdown(profile_blend: pd.DataFrame) -> str:
    lines = ["# Profile Blend Diagnostics", ""]
    if profile_blend.empty:
        lines.append("No full-profile blends were produced.")
        return "\n".join(lines)
    display_cols = [
        "profile",
        "blend_method",
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
        "reviewable",
        "review_reason",
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
        "experiment_selection.csv",
        "experiment_selection.md",
        "experiment_selection.json",
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


def run_experiment_matrix(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    run_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    settings = experiment_settings(config)
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
        "recommendation": "promote_best_candidate"
        if best
        else ("review_profile_blend" if best_blend else "keep_control_profile"),
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
    settings = experiment_settings(config)
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
            config,
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
    seed_ensemble_entries = _seed_ensemble_entries(profile_entries, config)
    profile_blend_entries = _profile_blend_entries(profile_entries, config)
    entries = [*profile_entries, *seed_ensemble_entries, *profile_blend_entries]

    rows = [entry["diagnostics"]["row"] for entry in entries]
    triage_rows = _decision_rows([row for row in rows if row.get("fold_scope") == "triage"], config, scope="triage")
    full_rows = _decision_rows([row for row in rows if row.get("fold_scope") == "full"], config, scope="full")
    comparison = _comparison_frame([*triage_rows, *full_rows])
    profile_delta = _profile_delta_vs_control(profile_entries, settings["control_profile"])
    seed_audit, seed_stability = _seed_audit_entries_to_frames(entries)
    seed_ensemble = _seed_ensemble_frame(entries)
    profile_blend = _profile_blend_frame(entries)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, config, settings["control_profile"])
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
    decision = {
        "run_id": run_dir.name,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "recommendation": "promote_best_candidate"
        if best
        else ("review_profile_blend" if best_blend else "keep_control_profile"),
        "diagnostic_zips": zip_paths,
    }
    report_dir = Path(output_dir) / "experiments" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(output_dir) / f"phase1_experiment_bundle_{run_dir.name}.zip"
    latest_bundle_path = Path(output_dir) / "phase1_latest_experiment_bundle.zip"
    decision["bundle_zip"] = str(bundle_path)
    decision["latest_bundle_zip"] = str(latest_bundle_path)
    decision["skipped_profiles"] = settings.get("skipped_profiles", []) or []
    _write_decision_files(report_dir, comparison, decision)
    _write_profile_delta(report_dir, profile_delta)
    _write_seed_audit_files(report_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(report_dir, seed_ensemble)
    _write_profile_blend_files(report_dir, profile_blend)
    experiment_selection = _write_experiment_selection(report_dir, settings)
    _write_decision_files(run_dir, comparison, decision)
    _write_profile_delta(run_dir, profile_delta)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_experiment_selection(run_dir, settings)
    bundle_path, latest_bundle_path = _write_experiment_bundle(
        output_dir=Path(output_dir),
        run_id=run_dir.name,
        report_dir=report_dir,
        zip_paths=zip_paths,
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
        "decision": decision,
        "zip_paths": zip_paths,
        "bundle_zip": str(bundle_path),
        "latest_bundle_zip": str(latest_bundle_path),
    }
