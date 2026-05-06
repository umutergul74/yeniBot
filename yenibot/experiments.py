from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.diagnostics import (
    calibration_table,
    experiment_ledger_diagnostics,
    feature_group_diagnostics,
    feature_profile_diagnostics,
    fold_diagnostics,
    mtf_leakage_diagnostics,
    phase1_report,
    recent_fold_diagnostics,
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


def experiment_settings(config: dict[str, Any]) -> dict[str, Any]:
    experiments = copy.deepcopy(_cfg(config, ["experiments"], {}) or {})
    control = str(experiments.get("control_profile") or _cfg(config, ["features", "active_profile"]))
    candidates = [str(profile) for profile in experiments.get("candidate_profiles", [])]
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
    experiments.setdefault("max_auto_full_candidates", None)
    experiments.setdefault("resume_existing", True)
    experiments.setdefault("force_retrain", False)
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
    threshold_metrics = threshold_diagnostics(predictions)
    threshold_summary = threshold_summary_diagnostics(threshold_metrics)
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
        recent_fold_summary=recent,
        threshold_summary=threshold_summary,
        score_band_lift=score_band_lift,
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
        "top_10_lift_fold_mean",
        "top_10_lift_global",
        "top_10_positive_lift_fold_rate",
        "mtf_leakage_passed",
        "stationarity_policy_passed",
        "passed_phase1",
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
        ["passed_phase1", "mean_rank_ic", "top_10_lift_global"],
        ascending=[False, False, False],
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
            "mean_long_f1",
            "test_f1_at_selected_threshold",
            "top_10_lift_global",
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
    best = _best_candidate(comparison, settings["control_profile"])
    decision = {
        "run_id": run_id,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "full_profiles": full_profiles,
        "recommendation": "promote_best_candidate" if best else "keep_control_profile",
    }
    _write_decision_files(run_dir, comparison, decision)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "profile_results": profile_results,
        "comparison": comparison,
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
    entries = []
    for scope_dir in _profile_dirs(run_dir):
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

    rows = [entry["diagnostics"]["row"] for entry in entries]
    triage_rows = _decision_rows([row for row in rows if row.get("fold_scope") == "triage"], config, scope="triage")
    full_rows = _decision_rows([row for row in rows if row.get("fold_scope") == "full"], config, scope="full")
    comparison = _comparison_frame([*triage_rows, *full_rows])
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
            config=profile_config(config, profile),
            prefix=f"phase1_diagnostics_{_slug(profile)}_{fold_scope}",
        )
        zip_paths.append(str(zip_path))

    best = _best_candidate(comparison, settings["control_profile"])
    decision = {
        "run_id": run_dir.name,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "recommendation": "promote_best_candidate" if best else "keep_control_profile",
        "diagnostic_zips": zip_paths,
    }
    report_dir = Path(output_dir) / "experiments" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(output_dir) / f"phase1_experiment_bundle_{run_dir.name}.zip"
    latest_bundle_path = Path(output_dir) / "phase1_latest_experiment_bundle.zip"
    decision["bundle_zip"] = str(bundle_path)
    decision["latest_bundle_zip"] = str(latest_bundle_path)
    _write_decision_files(report_dir, comparison, decision)
    _write_decision_files(run_dir, comparison, decision)
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
        "decision": decision,
        "zip_paths": zip_paths,
        "bundle_zip": str(bundle_path),
        "latest_bundle_zip": str(latest_bundle_path),
    }
