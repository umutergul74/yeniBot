"""Per-profile training execution, prediction summaries, and promotion gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from yenibot.diagnostics import (
    attach_threshold_summary_to_phase1_report,
    calibration_table,
    calibrate_split_probabilities_from_val,
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
)
from yenibot.features import filter_feature_columns, select_feature_columns
from yenibot.training import run_walk_forward_training

from yenibot.experiment.common import (
    _cfg,
    _float,
    _hash_payload,
    _json_ready,
    _metric_or,
    _optional_float,
    _optional_gate_float,
    _write_json,
)

from yenibot.experiment.configuration import (
    _frame_window,
    _is_complete,
    _manifest_path,
    _training_signature,
    experiment_settings,
    profile_config,
    profile_run_dir,
)

__all__ = [
    '_test_predictions',
    '_threshold_guard_from_report',
    '_threshold_summary_metric',
    '_threshold_selection_score',
    '_threshold_candidate_is_guarded',
    '_select_official_threshold_candidate',
    '_apply_official_threshold_fields',
    'summarize_profile_predictions',
    'run_profile_experiment',
    '_passes_triage',
    '_passes_full',
    '_decision_rows',
    '_auto_full_profiles',
    '_comparison_frame',
    '_best_candidate',
    '_comparison_markdown',
    '_write_decision_files',
    '_threshold_summary_value',
    '_cv_selected_threshold',
    '_cv_score_policy',
]

def _test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "split" in predictions.columns:
        return predictions[predictions["split"] == "test"].copy()
    return predictions.copy()

def _threshold_guard_from_report(report: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    guarded = report.get("threshold_guarded", {}) or {}
    source = str(guarded.get("threshold_source", ""))
    if prefix and source:
        source = f"{prefix}_{source}"
    return {
        "threshold_source": source,
        "reject_reason": str(guarded.get("reject_reason", "")),
        "threshold_mean": guarded.get("threshold_mean", np.nan),
        "f1": guarded.get("test_f1_at_guarded_threshold", np.nan),
        "precision": guarded.get("test_precision_at_guarded_threshold", np.nan),
        "recall": guarded.get("test_recall_at_guarded_threshold", np.nan),
        "pred_long_rate": guarded.get("test_pred_long_rate_at_guarded_threshold", np.nan),
        "passed": bool(report.get("passed_threshold_guarded", False)),
    }

def _threshold_summary_metric(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    row = threshold_summary.loc[threshold_summary["metric"].astype(str) == str(metric)]
    if row.empty:
        return np.nan
    return _float(row.iloc[0].to_dict(), "mean")

def _threshold_selection_score(
    threshold_summary: pd.DataFrame | None,
    source: str,
) -> float:
    if "constrained" in str(source):
        score = _threshold_summary_metric(threshold_summary, "source_constrained_f1")
        if np.isfinite(score):
            return score
    score = _threshold_summary_metric(threshold_summary, "source_best_f1")
    if np.isfinite(score):
        return score
    return np.nan

def _threshold_candidate_is_guarded(
    candidate: dict[str, Any],
    *,
    max_pred_long_rate: float,
    min_precision: float,
) -> bool:
    f1 = _optional_float(candidate.get("f1"))
    precision = _optional_float(candidate.get("precision"))
    pred_rate = _optional_float(candidate.get("pred_long_rate"))
    return bool(
        f1 is not None
        and precision is not None
        and pred_rate is not None
        and pred_rate <= max_pred_long_rate
        and precision >= min_precision
    )

def _select_official_threshold_candidate(
    *,
    raw_report: dict[str, Any],
    raw_threshold_summary: pd.DataFrame | None,
    calibrated_threshold_report: dict[str, Any] | None,
    calibrated_threshold_summary: pd.DataFrame | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    raw_candidate = _threshold_guard_from_report(raw_report)
    raw_candidate["candidate_order"] = 0
    raw_candidate["selection_score"] = _threshold_selection_score(
        raw_threshold_summary,
        str(raw_candidate.get("threshold_source", "")),
    )
    candidates = [raw_candidate]
    if calibrated_threshold_report:
        calibrated_candidate = _threshold_guard_from_report(calibrated_threshold_report, prefix="calibrated")
        calibrated_candidate["candidate_order"] = 1
        calibrated_candidate["selection_score"] = _threshold_selection_score(
            calibrated_threshold_summary,
            str(calibrated_candidate.get("threshold_source", "")),
        )
        candidates.append(calibrated_candidate)
    guarded_candidates = [
        candidate
        for candidate in candidates
        if _threshold_candidate_is_guarded(
            candidate,
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    ]
    if guarded_candidates:
        selected = max(
            guarded_candidates,
            key=lambda item: (
                _optional_float(item.get("selection_score")) or -np.inf,
                -int(item.get("candidate_order", 999) or 999),
            ),
        )
    else:
        selected = candidates[0]
    selected = dict(selected)
    selected["uses_calibration"] = str(selected.get("threshold_source", "")).startswith("calibrated_")
    selected["candidate_count"] = len(candidates)
    selected["guarded_candidate_count"] = len(guarded_candidates)
    return selected

def _apply_official_threshold_fields(
    row: dict[str, Any],
    ledger: pd.DataFrame,
    *,
    official: dict[str, Any],
    calibrated: dict[str, Any] | None = None,
) -> None:
    calibrated = calibrated or {}
    additions = {
        "official_threshold_source": str(official.get("threshold_source", "")),
        "official_threshold_reason": str(official.get("reject_reason", "")),
        "official_threshold_mean": official.get("threshold_mean", np.nan),
        "test_f1_at_official_threshold": official.get("f1", np.nan),
        "test_precision_at_official_threshold": official.get("precision", np.nan),
        "test_recall_at_official_threshold": official.get("recall", np.nan),
        "test_pred_long_rate_at_official_threshold": official.get("pred_long_rate", np.nan),
        "official_threshold_uses_calibration": bool(official.get("uses_calibration", False)),
        "official_threshold_candidate_count": int(official.get("candidate_count", 1) or 1),
        "official_threshold_guarded_candidate_count": int(official.get("guarded_candidate_count", 0) or 0),
        "official_threshold_selection_score": official.get("selection_score", np.nan),
        "calibrated_guarded_threshold_source": str(calibrated.get("threshold_source", "")),
        "calibrated_guarded_threshold_reason": str(calibrated.get("reject_reason", "")),
        "calibrated_guarded_threshold_mean": calibrated.get("threshold_mean", np.nan),
        "test_f1_at_calibrated_guarded_threshold": calibrated.get("f1", np.nan),
        "test_precision_at_calibrated_guarded_threshold": calibrated.get("precision", np.nan),
        "test_recall_at_calibrated_guarded_threshold": calibrated.get("recall", np.nan),
        "test_pred_long_rate_at_calibrated_guarded_threshold": calibrated.get("pred_long_rate", np.nan),
    }
    row.update(additions)
    for column, value in additions.items():
        ledger.loc[:, column] = value

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
    calibrated_threshold_report = None
    calibrated_threshold_metrics = pd.DataFrame()
    calibrated_threshold_summary = pd.DataFrame()
    calibration_cfg = _cfg(profile_cfg, ["validation", "calibration"], {}) or {}
    if bool(calibration_cfg.get("enabled", False)):
        try:
            calibration_method = str(calibration_cfg.get("method", "isotonic"))
            calibrated_splits = calibrate_split_probabilities_from_val(
                predictions,
                method=calibration_method,
            )
            calibrated_predictions = calibrated_splits[calibrated_splits["split"] == "test"].copy()
            report_frame = calibrated_predictions.copy()
            report_frame["prob_long"] = report_frame["prob_long_calibrated"]
            calibrated_report = phase1_report(report_frame, profile_cfg)
            calibrated_calibration = calibration_table(
                report_frame["label"],
                report_frame["prob_long"],
                bins=int(_cfg(profile_cfg, ["validation", "calibration_bins"], 10)),
            )
            calibrated_threshold_metrics = threshold_diagnostics(
                calibrated_splits,
                score_column="prob_long_calibrated",
                max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
                min_precision=float(threshold_cfg.get("min_precision", 0.30)),
            )
            calibrated_threshold_summary = threshold_summary_diagnostics(calibrated_threshold_metrics)
            calibrated_threshold_report = attach_threshold_summary_to_phase1_report(
                dict(calibrated_report),
                calibrated_threshold_summary,
                profile_cfg,
            )
        except ValueError:
            calibrated_report = None
            calibrated_calibration = pd.DataFrame()
            calibrated_predictions = pd.DataFrame()
            calibrated_threshold_report = None
            calibrated_threshold_metrics = pd.DataFrame()
            calibrated_threshold_summary = pd.DataFrame()
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
    calibrated_guard = (
        _threshold_guard_from_report(calibrated_threshold_report, prefix="calibrated")
        if calibrated_threshold_report
        else {}
    )
    official_threshold = _select_official_threshold_candidate(
        raw_report=report,
        raw_threshold_summary=threshold_summary,
        calibrated_threshold_report=calibrated_threshold_report,
        calibrated_threshold_summary=calibrated_threshold_summary,
        config=profile_cfg,
    )
    _apply_official_threshold_fields(
        row,
        ledger,
        official=official_threshold,
        calibrated=calibrated_guard,
    )
    min_long_f1 = float(_cfg(profile_cfg, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(profile_cfg, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    official_f1 = _optional_float(row.get("test_f1_at_official_threshold"))
    official_pred_rate = _optional_float(row.get("test_pred_long_rate_at_official_threshold"))
    official_checks = dict(report.get("checks", {}) or {})
    official_checks["long_f1"] = bool(official_f1 is not None and official_f1 > min_long_f1)
    official_checks["threshold_pred_long_rate"] = bool(
        official_pred_rate is not None and official_pred_rate <= max_pred_long_rate
    )
    row["passed_phase1_official_threshold"] = all(bool(value) for value in official_checks.values())
    ledger.loc[:, "passed_phase1_official_threshold"] = row["passed_phase1_official_threshold"]
    row["mtf_leakage_passed"] = bool(mtf.empty or mtf["passed"].all())
    row["stationarity_policy_passed"] = bool(stationarity.empty or stationarity["passed"].all())
    row["fold_count"] = int(fold_metrics["fold"].nunique()) if not fold_metrics.empty else 0
    return {
        "report": report,
        "calibration": calibration,
        "calibrated_report": calibrated_report,
        "calibrated_calibration": calibrated_calibration,
        "calibrated_predictions": calibrated_predictions,
        "calibrated_threshold_report": calibrated_threshold_report,
        "calibrated_threshold_metrics": calibrated_threshold_metrics,
        "calibrated_threshold_summary": calibrated_threshold_summary,
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
        "test_f1_at_official_threshold",
        _metric_or(
            row,
            "test_f1_at_guarded_threshold",
            _metric_or(
                row,
                "test_f1_at_constrained_threshold",
                _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
            ),
        ),
    )
    control_selected_f1 = _metric_or(
        control,
        "test_f1_at_official_threshold",
        _metric_or(
            control,
            "test_f1_at_guarded_threshold",
            _metric_or(
                control,
                "test_f1_at_constrained_threshold",
                _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
            ),
        ),
    )
    selected_f1_floor = _optional_gate_float(gates, "min_selected_threshold_f1", None)
    if selected_f1_floor is not None and selected_f1 < selected_f1_floor:
        reasons.append("official_threshold_f1")
    selected_f1_delta = _optional_gate_float(gates, "min_selected_threshold_f1_delta", None)
    if selected_f1_delta is not None and selected_f1 < control_selected_f1 + selected_f1_delta:
        reasons.append("official_threshold_f1_delta")
    threshold_checks = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_checks.get("max_pred_long_rate", 0.70))
    official_pred_rate = _float(
        row,
        "test_pred_long_rate_at_official_threshold",
        _float(row, "test_pred_long_rate_at_constrained_threshold", np.nan),
    )
    if np.isfinite(official_pred_rate) and official_pred_rate > max_pred_long_rate:
        reasons.append("official_pred_long_rate")
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
        "guarded_threshold_source",
        "guarded_threshold_reason",
        "test_f1_at_guarded_threshold",
        "test_precision_at_guarded_threshold",
        "test_recall_at_guarded_threshold",
        "test_pred_long_rate_at_guarded_threshold",
        "guarded_threshold_mean",
        "official_threshold_source",
        "official_threshold_reason",
        "test_f1_at_official_threshold",
        "test_precision_at_official_threshold",
        "test_recall_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "official_threshold_mean",
        "official_threshold_uses_calibration",
        "official_threshold_selection_score",
        "calibrated_guarded_threshold_source",
        "test_f1_at_calibrated_guarded_threshold",
        "test_pred_long_rate_at_calibrated_guarded_threshold",
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
        "passed_phase1_guarded_threshold",
        "passed_phase1_official_threshold",
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
            "test_f1_at_guarded_threshold",
            "guarded_threshold_source",
            "test_pred_long_rate_at_guarded_threshold",
            "test_f1_at_official_threshold",
            "official_threshold_source",
            "test_pred_long_rate_at_official_threshold",
            "test_f1_at_calibrated_guarded_threshold",
            "test_pred_long_rate_at_constrained_threshold",
            "top_10_lift_global",
            "top_10_bad_fold_lift_mean",
            "passed_phase1_selected_threshold",
            "passed_phase1_constrained_threshold",
            "passed_phase1_guarded_threshold",
            "passed_phase1_official_threshold",
            "passed_phase1_legacy_v3",
            "active_validation_charter",
            "historical_walk_forward_evidence_passed",
            "frozen_future_oos_evidence_passed",
            "model_evidence_passed_active_charter",
            "phase2_ready",
            "phase1_status",
            "promotable",
            "reject_reason",
        ]
        visible_cols = [column for column in display_cols if column in comparison.columns]
        visible = comparison[visible_cols].copy()
        lines.append("| " + " | ".join(visible_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible_cols)) + " |")
        for _, row in visible.iterrows():
            values = [str(row[column]) for column in visible_cols]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(decision), indent=2, sort_keys=True)])
    return "\n".join(lines)

def _write_decision_files(run_dir: Path, comparison: pd.DataFrame, decision: dict[str, Any]) -> None:
    comparison.to_csv(run_dir / "profile_comparison.csv", index=False)
    (run_dir / "profile_comparison.md").write_text(_comparison_markdown(comparison, decision), encoding="utf-8")
    _write_json(run_dir / "decision_report.json", decision)
    _write_json(run_dir / "best_candidate.json", decision.get("best_candidate") or {})

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
