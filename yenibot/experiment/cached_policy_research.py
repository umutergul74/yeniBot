"""Run a preregistered policy experiment from immutable historical predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.adaptive_ensemble import (
    aggregate_fixed_model_predictions,
    select_models_by_validation_lcb,
    split_validation_predictions,
)
from yenibot.experiment.common import _hash_payload, _write_json
from yenibot.experiment.rolling_research import (
    _paired_policy_comparison,
    _policy_metrics,
    _recency_decision_markdown,
    _research_summary,
    _select_validation_threshold,
)

__all__ = ["run_cached_adaptive_ensemble_research"]


_ARTIFACTS = {
    "summary": "adaptive_ensemble_summary.csv",
    "by_fold": "adaptive_ensemble_by_fold.csv",
    "selection_audit": "adaptive_ensemble_selection_audit.csv",
    "paired_comparison": "adaptive_ensemble_paired_comparison.csv",
    "decision": "adaptive_ensemble_decision.json",
    "manifest": "adaptive_ensemble_manifest.json",
}


def _research_config(config: dict[str, Any]) -> dict[str, Any]:
    return (
        config.get("experiments", {})
        .get("next_research_cycle", {})
        .get("adaptive_ensemble", {})
        or {}
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_completed(output_path: Path, signature: str) -> dict[str, Any] | None:
    paths = {name: output_path / filename for name, filename in _ARTIFACTS.items()}
    if not all(path.exists() for path in paths.values()):
        return None
    manifest = _read_json(paths["manifest"])
    if manifest.get("signature_hash") != signature:
        return None
    return {
        "enabled": True,
        "status": "reused",
        "summary": pd.read_csv(paths["summary"]),
        "by_fold": pd.read_csv(paths["by_fold"]),
        "selection_audit": pd.read_csv(paths["selection_audit"]),
        "paired_comparison": pd.read_csv(paths["paired_comparison"]),
        "decision": _read_json(paths["decision"]),
        "manifest": manifest,
        "output_dir": output_path,
    }


def _source_contract(
    source_path: Path,
    research_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    source_manifest_path = source_path / "recency_ensemble_manifest.json"
    schedule_path = source_path / "recency_ensemble_schedule.csv"
    if not source_manifest_path.exists() or not schedule_path.exists():
        raise FileNotFoundError(
            "Adaptive research requires the immutable historical recency manifest "
            f"and schedule under {source_path}"
        )
    source_manifest = _read_json(source_manifest_path)
    prediction_signature = str(source_manifest.get("prediction_signature_hash") or "")
    expected_signature = str(
        research_cfg.get("source_prediction_signature_hash") or ""
    )
    if not prediction_signature or prediction_signature != expected_signature:
        raise ValueError(
            "Historical prediction signature mismatch: "
            f"configured={expected_signature} observed={prediction_signature}"
        )
    if bool(source_manifest.get("failed_future_oos_used_for_policy_selection", True)):
        raise ValueError("Source research must exclude failed Future-OOS selection")
    schedule = pd.read_csv(schedule_path)
    if schedule.empty:
        raise ValueError("Historical rolling-origin schedule is empty")
    for column in ("validation_start", "validation_end", "test_start", "test_end"):
        schedule[column] = pd.to_datetime(schedule[column], utc=True, errors="raise")
    maximum_selection = pd.to_datetime(
        research_cfg["historical_selection_end"], utc=True, errors="raise"
    )
    if schedule["test_end"].max() > maximum_selection:
        raise ValueError(
            "Historical cache extends past the preregistered selection boundary: "
            f"observed={schedule['test_end'].max()} configured={maximum_selection}"
        )
    for window in research_cfg.get("excluded_evaluation_windows", []) or []:
        excluded_start = pd.to_datetime(window["start"], utc=True, errors="raise")
        excluded_end = pd.to_datetime(window["end"], utc=True, errors="raise")
        overlaps = (
            schedule["test_start"].le(excluded_end)
            & schedule["test_end"].ge(excluded_start)
        )
        if bool(overlaps.any()):
            raise ValueError(
                "Historical policy-selection schedule overlaps an excluded OOS "
                f"window: {window.get('candidate_id', '')}"
            )
    return source_manifest, schedule.sort_values("fold"), prediction_signature


def _cache_path(source_path: Path, prediction_signature: str, target_fold: int) -> Path:
    return source_path / (
        f"cross_predictions_{prediction_signature[:12]}_fold_{int(target_fold):03d}.parquet"
    )


def _window(
    raw: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    return raw.loc[timestamps.between(start, end)].copy()


def _write_markdown(
    path: Path,
    *,
    manifest: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Validation-Adaptive Ensemble Research",
                "",
                f"- Hypothesis: `{manifest.get('hypothesis_id')}`",
                f"- Status: `{decision.get('status')}`",
                f"- Control: `{decision.get('control_policy')}`",
                f"- Recommended policy: `{decision.get('recommended_policy')}`",
                f"- Source run: `{manifest.get('source_run_id')}`",
                f"- Source prediction signature: `{manifest.get('source_prediction_signature_hash')}`",
                f"- Historical selection end: `{manifest.get('historical_selection_end')}`",
                "- Model selection: first 720 validation hours only.",
                "- Purge between selection and calibration: 24 hours.",
                "- Threshold selection: disjoint trailing validation calibration only.",
                "- Failed Future-OOS rows used for selection: `False`.",
                "- Fit operations performed: `0`.",
                "- Automatic candidate freeze allowed: `False`.",
                "",
                _recency_decision_markdown(decision),
            ]
        ),
        encoding="utf-8",
    )


def run_cached_adaptive_ensemble_research(
    *,
    source_research_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    code_commit: str = "",
) -> dict[str, Any]:
    """Evaluate one preregistered adaptive policy on cached historical CV rows.

    No model, scaler, HMM, threshold, or ensemble weight is fitted on either
    failed Future-OOS window. The only labels used are the immutable historical
    validation/test rows explicitly bounded by the source schedule.
    """

    research_cfg = _research_config(config)
    if not bool(research_cfg.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    if research_cfg.get("preregistered") is not True:
        raise ValueError("Adaptive ensemble research must be preregistered")

    source_path = Path(source_research_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_manifest, schedule, prediction_signature = _source_contract(
        source_path,
        research_cfg,
    )
    signature = _hash_payload(
        {
            "hypothesis": research_cfg,
            "source_prediction_signature_hash": prediction_signature,
            "source_manifest_signature_hash": source_manifest.get("signature_hash"),
            "code_commit": str(code_commit),
        }
    )
    completed = _load_completed(output_path, signature)
    if completed is not None:
        return completed

    policy_cfg = research_cfg.get("policy", {}) or {}
    control_cfg = research_cfg.get("control", {}) or {}
    comparison_cfg = research_cfg.get("comparison", {}) or {}
    candidate_name = str(policy_cfg["name"])
    control_name = str(control_cfg["name"])
    threshold_cfg = config.get("validation", {}).get("threshold_checks", {}) or {}
    rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []

    for schedule_row in schedule.itertuples(index=False):
        target_fold = int(schedule_row.fold)
        cache_path = _cache_path(source_path, prediction_signature, target_fold)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing immutable cross-prediction cache: {cache_path}")
        raw = pd.read_parquet(cache_path)
        required = {"timestamp", "model_fold", "prob_long", "label", "forward_return"}
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"Historical cross-prediction cache is incomplete: {missing}")
        raw["model_fold"] = pd.to_numeric(raw["model_fold"], errors="raise").astype(int)
        if bool(raw["model_fold"].gt(target_fold).any()):
            raise ValueError(f"Future model found in target fold {target_fold} cache")
        validation_raw = _window(
            raw,
            start=schedule_row.validation_start,
            end=schedule_row.validation_end,
        )
        test_raw = _window(
            raw,
            start=schedule_row.test_start,
            end=schedule_row.test_end,
        )
        selector, calibration, split_meta = split_validation_predictions(
            validation_raw,
            selector_rows=int(policy_cfg["selector_rows"]),
            purge_rows=int(policy_cfg["purge_rows"]),
            min_calibration_rows=int(policy_cfg["min_calibration_rows"]),
        )
        split_rows.append({"target_fold": target_fold, **split_meta})
        candidate_folds, audit = select_models_by_validation_lcb(
            selector,
            target_fold=target_fold,
            pool_recent_k=int(policy_cfg["pool_recent_k"]),
            select_top_k=int(policy_cfg["select_top_k"]),
            block_length=int(policy_cfg["bootstrap_block_length"]),
            bootstrap_repeats=int(policy_cfg["bootstrap_repeats"]),
            confidence_level=float(policy_cfg["confidence_level"]),
            random_seed=int(policy_cfg["random_seed"]),
        )
        audit["policy_name"] = candidate_name
        audit_frames.append(audit)
        eligible = sorted(selector["model_fold"].astype(int).unique().tolist())
        control_folds = eligible[-max(1, int(control_cfg["recent_k"])) :]

        for policy_name, selected_folds in (
            (control_name, control_folds),
            (candidate_name, candidate_folds),
        ):
            calibration_predictions = aggregate_fixed_model_predictions(
                calibration,
                selected_model_folds=selected_folds,
                policy_name=policy_name,
            )
            threshold = _select_validation_threshold(
                calibration_predictions,
                max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
                min_precision=float(threshold_cfg.get("min_precision", 0.30)),
            )
            test_predictions = aggregate_fixed_model_predictions(
                test_raw,
                selected_model_folds=selected_folds,
                policy_name=policy_name,
            )
            rows.append(
                {
                    "policy_name": policy_name,
                    "policy": (
                        "validation_lcb_top_k"
                        if policy_name == candidate_name
                        else "equal_recent_k"
                    ),
                    "target_fold": target_fold,
                    "test_start": schedule_row.test_start,
                    "test_end": schedule_row.test_end,
                    "eligible_model_count": len(eligible),
                    "selected_model_count": len(selected_folds),
                    "selected_model_folds": ",".join(
                        str(item) for item in selected_folds
                    ),
                    "validation_threshold": threshold["threshold"],
                    "validation_f1": threshold["f1"],
                    "validation_pred_long_rate": threshold["pred_long_rate"],
                    "threshold_guarded": bool(threshold["guarded"]),
                    "threshold_selection_rows": split_meta["calibration_rows"],
                    "selection_and_threshold_rows_disjoint": split_meta[
                        "selection_and_calibration_disjoint"
                    ],
                    **_policy_metrics(
                        test_predictions,
                        threshold=float(threshold["threshold"]),
                    ),
                }
            )

    by_fold = pd.DataFrame(rows)
    summary = _research_summary(by_fold)
    paired, decision = _paired_policy_comparison(
        by_fold,
        summary,
        control_policy=control_name,
        comparison_config=comparison_cfg,
    )
    decision.update(
        {
            "hypothesis_id": str(research_cfg["hypothesis_id"]),
            "candidate_policy": candidate_name,
            "policy_count": 2,
            "single_candidate_confirmatory_test": True,
            "model_selection_metric": "validation_rank_ic_block_bootstrap_lower_bound",
            "model_selection_and_threshold_calibration_disjoint": True,
            "failed_future_oos_used_for_selection": False,
            "automatic_freeze_allowed": False,
        }
    )
    selection_audit = pd.concat(audit_frames, ignore_index=True)
    split_audit = pd.DataFrame(split_rows)
    manifest = {
        "status": "completed",
        "signature_hash": signature,
        "hypothesis_id": str(research_cfg["hypothesis_id"]),
        "preregistered": True,
        "code_commit": str(code_commit),
        "source_run_id": str(research_cfg["source_run_id"]),
        "source_prediction_signature_hash": prediction_signature,
        "source_manifest_signature_hash": source_manifest.get("signature_hash"),
        "historical_selection_end": str(research_cfg["historical_selection_end"]),
        "excluded_evaluation_windows": research_cfg.get(
            "excluded_evaluation_windows", []
        ),
        "fold_count": int(by_fold["target_fold"].nunique()),
        "fit_operations_performed": 0,
        "failed_future_oos_used_for_selection": False,
        "target_fold_test_labels_used_for_model_or_threshold_selection": False,
        "historical_test_labels_used_for_confirmatory_policy_comparison": True,
        "selection_and_threshold_rows_disjoint": bool(
            split_audit["selection_and_calibration_disjoint"].all()
        ),
        "source_prediction_cache_mutated": False,
        "decision_status": decision.get("status"),
        "candidate_ready_for_preregistration": bool(
            decision.get("candidate_ready_for_preregistration", False)
        ),
    }
    summary.to_csv(output_path / _ARTIFACTS["summary"], index=False)
    by_fold.to_csv(output_path / _ARTIFACTS["by_fold"], index=False)
    selection_audit.to_csv(
        output_path / _ARTIFACTS["selection_audit"], index=False
    )
    split_audit.to_csv(
        output_path / "adaptive_ensemble_validation_split_audit.csv", index=False
    )
    paired.to_csv(output_path / _ARTIFACTS["paired_comparison"], index=False)
    _write_json(output_path / _ARTIFACTS["decision"], decision)
    _write_json(output_path / _ARTIFACTS["manifest"], manifest)
    _write_markdown(
        output_path / "adaptive_ensemble_decision.md",
        manifest=manifest,
        decision=decision,
    )
    return {
        "enabled": True,
        "status": "completed",
        "summary": summary,
        "by_fold": by_fold,
        "selection_audit": selection_audit,
        "validation_split_audit": split_audit,
        "paired_comparison": paired,
        "decision": decision,
        "manifest": manifest,
        "output_dir": output_path,
    }
