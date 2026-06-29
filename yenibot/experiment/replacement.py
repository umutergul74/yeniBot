"""Build and publish a pre-registered replacement candidate after OOS failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _durable_copy_file,
    _hash_payload,
    _slug,
    _write_json,
)
from yenibot.experiment.configuration import profile_config
from yenibot.experiment.holdout import _predict_holdout_for_profile
from yenibot.experiment.rolling_research import (
    _select_validation_threshold,
    aggregate_recency_predictions,
)
from yenibot.experiment.training import run_profile_experiment
from yenibot.training import PurgedWalkForwardCV

__all__ = [
    "publish_replacement_candidate_reports",
    "run_replacement_candidate_fit",
]


def _replacement_config(config: dict[str, Any]) -> dict[str, Any]:
    return (
        _cfg(
            config,
            ["experiments", "next_research_cycle", "replacement_candidate"],
            {},
        )
        or {}
    )


def _fit_frame(
    frame: pd.DataFrame,
    *,
    anchor_data_end: str,
) -> pd.DataFrame:
    if frame.empty or "timestamp" not in frame.columns:
        raise ValueError("Replacement candidate requires timestamped labeled rows")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    anchor = pd.to_datetime(anchor_data_end, utc=True, errors="raise")
    out = frame.loc[timestamps <= anchor].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError("No labeled rows exist at or before the replacement anchor")
    observed_end = pd.to_datetime(out["timestamp"], utc=True).max()
    if observed_end != anchor:
        raise ValueError(
            "Replacement anchor must equal an available labeled timestamp: "
            f"configured={anchor.isoformat()} observed={observed_end.isoformat()}"
        )
    return out


def _folds(frame: pd.DataFrame, config: dict[str, Any]) -> list[Any]:
    cv_cfg = config.get("walk_forward", {}) or {}
    cv = PurgedWalkForwardCV(
        train_bars=int(cv_cfg["train_bars"]),
        val_bars=int(cv_cfg["val_bars"]),
        test_bars=int(cv_cfg["test_bars"]),
        step_bars=int(cv_cfg["step_bars"]),
        purge_bars=int(cv_cfg["purge_bars"]),
        embargo_bars=int(cv_cfg["embargo_bars"]),
    )
    return list(cv.split(len(frame)))


def _selection_evidence(
    run_dir: Path,
    *,
    required_policy: str,
    required_track: str,
) -> tuple[dict[str, Any], str]:
    path = run_dir / "recency_research" / "recency_ensemble_decision.json"
    if not path.exists():
        raise FileNotFoundError(
            "Replacement fit requires completed historical recency research: "
            f"{path}"
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("recommended_policy") != required_policy:
        raise ValueError(
            "Replacement policy does not match the historical decision: "
            f"configured={required_policy} "
            f"recommended={decision.get('recommended_policy')}"
        )
    if decision.get("recommended_selection_track") != required_track:
        raise ValueError(
            "Replacement selection track does not match the historical decision: "
            f"configured={required_track} "
            f"recommended={decision.get('recommended_selection_track')}"
        )
    if not bool(decision.get("candidate_ready_for_preregistration", False)):
        raise ValueError("Historical recency winner is not ready for pre-registration")
    if bool(decision.get("failed_future_oos_used_for_selection", True)):
        raise ValueError("Failed future OOS must not select the replacement policy")
    return decision, _hash_payload(decision)


def _replacement_markdown(payload: dict[str, Any]) -> str:
    pinned = payload.get("status") == "manifest_pinned_existing_candidate"
    final_note = (
        "- Candidate was fitted in the declared source run and its manifest hash is already pinned. "
        "Artifact integrity is verified separately by future-OOS preflight."
        if pinned
        else "- Promotion is blocked until the generated manifest hash is pinned in config."
    )
    return "\n".join(
        [
            "# Replacement Candidate Fit",
            "",
            f"- Status: `{payload.get('status')}`",
            f"- Candidate: `{payload.get('candidate_id')}`",
            f"- Profile: `{payload.get('profile')}`",
            f"- Policy: `{payload.get('policy_name')}`",
            f"- Selection track: `{payload.get('selection_track')}`",
            f"- Anchor: `{payload.get('anchor_data_end')}`",
            f"- Source run: `{payload.get('source_run_id')}`",
            f"- Fold scope: `{payload.get('fold_scope')}`",
            f"- Frozen model folds: `{payload.get('selected_model_folds')}`",
            f"- Validation threshold: `{payload.get('threshold', {}).get('value')}`",
            "- Policy-selection data: historical walk-forward only.",
            "- Failed future-OOS data: fit-eligible after retirement, never used for policy selection.",
            "- Future-OOS fit operations performed: `0`.",
            final_note,
            "",
        ]
    )


def _preregistration_patch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit config patch required after replacement fitting.

    The patch intentionally leaves ``expected_manifest_hash`` as a manual pin.
    That keeps the workflow two-step and prevents diagnostics-only edits from
    silently activating a future-OOS candidate.
    """

    candidate_id = str(payload.get("candidate_id", ""))
    source_run_id = str(payload.get("source_run_id", ""))
    anchor_data_end = str(payload.get("anchor_data_end", ""))
    threshold = dict(payload.get("threshold", {}) or {})
    candidate_spec = {
        "candidate_id": candidate_id,
        "candidate_type": "recency_profile",
        "profile": str(payload.get("profile", "")),
        "source_run_id": source_run_id,
        "anchor_data_end": anchor_data_end,
        "fold_scope": str(payload.get("fold_scope", "replacement_recent3")),
        "recency_policy": "equal_recent_k",
        "recent_k": int(payload.get("recent_k", 3) or 3),
        "threshold": threshold,
        "manifest_candidate_status": "preregistered_manifest_pin_pending",
        "status": "preregistered_manifest_pin_pending",
        "required_for_evaluation": True,
        "expected_manifest_hash": "<fill_after_05_generates_manifest_hash>",
    }
    return {
        "status": "replacement_fit_complete_manifest_pin_required",
        "candidate_id": candidate_id,
        "source_run_id": source_run_id,
        "anchor_data_end": anchor_data_end,
        "manual_pin_required": True,
        "same_window_selection_allowed": False,
        "failed_future_oos_used_for_policy_selection": False,
        "stage_1_unpinned_config_patch": {
            "experiments": {
                "frozen_candidates": {
                    "enabled": True,
                    "lifecycle_state": "replacement_candidate_preregistered_manifest_pin_pending",
                    "protocol_version": "v1",
                    "anchor_run_id": source_run_id,
                    "anchor_data_end": anchor_data_end,
                    "primary_candidate_id": candidate_id,
                    "artifact_policy": (
                        "reference_and_hash_existing_pre_anchor_fold_artifacts; "
                        "never_refit_during_future_oos_evaluation"
                    ),
                    "candidates": [candidate_spec],
                },
                "future_oos_validation": {
                    "enabled": True,
                    "pause_reason": (
                        "Enabled only after replacement candidate fit. "
                        "Evaluation remains blocked until expected_manifest_hash "
                        "is pinned and enough new rows mature after anchor."
                    ),
                },
                "next_research_cycle": {
                    "status": "replacement_fit_complete_manifest_pin_required",
                    "replacement_candidate": {
                        "enabled": False,
                        "status": "fit_complete_manifest_pin_required",
                        "candidate_id": candidate_id,
                    },
                },
            }
        },
        "stage_2_pin_instruction": (
            "Run notebook 05 once with the stage_1 patch, copy the generated "
            "manifest_hash from frozen_candidate_index.csv into "
            "expected_manifest_hash, then rerun notebook 05. Do not evaluate "
            "future OOS until preflight passes all manifest checks."
        ),
    }


def _preregistration_patch_markdown(payload: dict[str, Any]) -> str:
    patch = payload.get("stage_1_unpinned_config_patch", {})
    return "\n".join(
        [
            "# Replacement Candidate Preregistration Patch",
            "",
            f"- Status: `{payload.get('status')}`",
            f"- Candidate: `{payload.get('candidate_id')}`",
            f"- Source run: `{payload.get('source_run_id')}`",
            f"- Anchor: `{payload.get('anchor_data_end')}`",
            "- Manual manifest hash pin required: `True`",
            "- Failed future-OOS used for policy selection: `False`",
            "",
            "## Stage 1",
            "",
            "Apply the JSON-equivalent config patch below, then run notebook 05.",
            "",
            "```json",
            json.dumps(patch, indent=2, sort_keys=True),
            "```",
            "",
            "## Stage 2",
            "",
            str(payload.get("stage_2_pin_instruction", "")),
            "",
        ]
    )


def run_replacement_candidate_fit(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    run_id: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit the fixed recent-model policy through a new pre-registration anchor."""

    spec = _replacement_config(config)
    if not bool(spec.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}

    run_path = Path(checkpoint_dir) / "experiments" / str(run_id)
    policy_name = str(spec["policy_name"])
    selection_track = str(spec["selection_track"])
    decision, decision_hash = _selection_evidence(
        run_path,
        required_policy=policy_name,
        required_track=selection_track,
    )
    fit_frame = _fit_frame(
        frame,
        anchor_data_end=str(spec["anchor_data_end"]),
    )
    profile = str(spec["profile"])
    profile_cfg = profile_config(config, profile)
    folds = _folds(fit_frame, profile_cfg)
    recent_k = int(spec.get("recent_k", 3))
    if len(folds) < recent_k:
        raise ValueError(
            f"Replacement candidate needs {recent_k} folds; only {len(folds)} exist"
        )
    selected_folds = [int(item.fold) for item in folds[-recent_k:]]
    fold_scope = str(spec.get("fold_scope", "replacement_recent3"))
    result = run_profile_experiment(
        fit_frame,
        config,
        profile=profile,
        checkpoint_dir=checkpoint_dir,
        run_id=str(run_id),
        fold_scope=fold_scope,
        fold_ids=selected_folds,
        resume_existing=bool(spec.get("resume_existing", True)),
        force_retrain=bool(spec.get("force_retrain", False)),
        device=device,
    )

    latest_fold = folds[-1]
    seq_len = int(profile_cfg.get("model", {}).get("seq_len", 64))
    context_start = max(0, int(latest_fold.val[0]) - seq_len + 1)
    context_end = int(latest_fold.val[-1]) + 1
    context = fit_frame.iloc[context_start:context_end].copy().reset_index(drop=True)
    timestamps = pd.to_datetime(fit_frame["timestamp"], utc=True)
    validation_start = timestamps.iloc[int(latest_fold.val[0])]
    validation_end = timestamps.iloc[int(latest_fold.val[-1])]
    raw = _predict_holdout_for_profile(
        scope_dir=Path(result["output_dir"]),
        manifest={
            "profile": profile,
            "feature_columns": result["feature_columns"],
        },
        holdout_context=context,
        holdout_start=validation_start,
        holdout_end=validation_end,
        model_folds=set(selected_folds),
        config=profile_cfg,
    )
    if raw.empty:
        raise ValueError("Replacement validation cross-predictions are empty")
    validation = aggregate_recency_predictions(
        raw,
        target_fold=int(latest_fold.fold),
        policy="equal_recent_k",
        recent_k=recent_k,
    )
    threshold_cfg = profile_cfg.get("validation", {}).get("threshold_checks", {}) or {}
    threshold = _select_validation_threshold(
        validation,
        max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    payload = {
        "enabled": True,
        "status": "fit_complete_manifest_pin_required",
        "candidate_id": str(spec["candidate_id"]),
        "candidate_type": "recency_profile",
        "profile": profile,
        "policy_name": policy_name,
        "policy": "equal_recent_k",
        "recent_k": recent_k,
        "selection_track": selection_track,
        "selection_evidence_hash": decision_hash,
        "selection_decision_status": decision.get("status"),
        "failed_future_oos_used_for_policy_selection": False,
        "failed_future_oos_allowed_in_fit_after_retirement": True,
        "anchor_data_end": pd.to_datetime(
            spec["anchor_data_end"],
            utc=True,
        ).isoformat(),
        "source_run_id": str(run_id),
        "fold_scope": fold_scope,
        "selected_model_folds": selected_folds,
        "fit_rows": int(len(fit_frame)),
        "fit_data_start": pd.to_datetime(
            fit_frame["timestamp"],
            utc=True,
        ).min().isoformat(),
        "fit_data_end": pd.to_datetime(
            fit_frame["timestamp"],
            utc=True,
        ).max().isoformat(),
        "validation_start": validation_start.isoformat(),
        "validation_end": validation_end.isoformat(),
        "validation_rows": int(len(validation)),
        "threshold": {
            "value": float(threshold["threshold"]),
            "source": "replacement_recent3_common_validation_guarded_f1",
            "selected_from": "pre_anchor_latest_fold_validation_only",
            "validation_f1": float(threshold["f1"]),
            "validation_precision": float(threshold["precision"]),
            "validation_recall": float(threshold["recall"]),
            "validation_pred_long_rate": float(threshold["pred_long_rate"]),
            "guarded": bool(threshold["guarded"]),
        },
        "training_reused": bool(result.get("skipped", False)),
        "future_oos_fit_operations_performed": 0,
        "manifest_pin_required": True,
        "promotion_allowed": False,
    }
    preregistration_patch = _preregistration_patch_payload(payload)
    scope_path = Path(result["output_dir"])
    _write_json(scope_path / "replacement_candidate_fit.json", payload)
    (scope_path / "replacement_candidate_fit.md").write_text(
        _replacement_markdown(payload),
        encoding="utf-8",
    )
    _write_json(
        scope_path / "replacement_preregistration_patch.json",
        preregistration_patch,
    )
    (scope_path / "replacement_preregistration_patch.md").write_text(
        _preregistration_patch_markdown(preregistration_patch),
        encoding="utf-8",
    )
    _write_json(run_path / "replacement_candidate_fit.json", payload)
    (run_path / "replacement_candidate_fit.md").write_text(
        _replacement_markdown(payload),
        encoding="utf-8",
    )
    _write_json(
        run_path / "replacement_preregistration_patch.json",
        preregistration_patch,
    )
    (run_path / "replacement_preregistration_patch.md").write_text(
        _preregistration_patch_markdown(preregistration_patch),
        encoding="utf-8",
    )
    return {
        **payload,
        "output_dir": scope_path,
        "preregistration_patch": preregistration_patch,
        "summary": result.get("summary", {}),
    }


def publish_replacement_candidate_reports(
    run_dir: str | Path,
    report_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy compact replacement build evidence into the diagnostics bundle."""

    source = Path(run_dir)
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload_path = source / "replacement_candidate_fit.json"
    if not payload_path.exists():
        frozen = _cfg(config or {}, ["experiments", "frozen_candidates"], {}) or {}
        candidates = list(frozen.get("candidates", []) or [])
        primary_id = str(frozen.get("primary_candidate_id") or "")
        primary = next(
            (
                dict(item)
                for item in candidates
                if str(item.get("candidate_id") or "") == primary_id
            ),
            {},
        )
        expected_hash = str(primary.get("expected_manifest_hash") or "").strip()
        pinned = bool(
            primary
            and expected_hash
            and "<fill_" not in expected_hash
            and str(primary.get("status") or "").startswith("manifest_pinned")
        )
        if pinned:
            historical: dict[str, Any] = {}
            source_run_id = str(primary.get("source_run_id") or "").strip()
            historical_paths = [
                source.parent / source_run_id / "replacement_candidate_fit.json",
                (
                    source.parent
                    / source_run_id
                    / _slug(str(primary.get("profile") or ""))
                    / str(primary.get("fold_scope") or "full")
                    / "replacement_candidate_fit.json"
                ),
            ]
            for historical_path in historical_paths:
                if not historical_path.exists():
                    continue
                candidate_payload = json.loads(
                    historical_path.read_text(encoding="utf-8")
                )
                if str(candidate_payload.get("candidate_id") or "") == primary_id:
                    historical = candidate_payload
                    break
            payload = {
                **historical,
                "available": True,
                "enabled": False,
                "status": "manifest_pinned_existing_candidate",
                "candidate_id": primary_id,
                "candidate_type": primary.get("candidate_type"),
                "profile": primary.get("profile"),
                "source_run_id": primary.get("source_run_id"),
                "fold_scope": primary.get("fold_scope"),
                "anchor_data_end": primary.get("anchor_data_end"),
                "threshold": dict(primary.get("threshold", {}) or {}),
                "expected_manifest_hash": expected_hash,
                "manifest_pin_required": False,
                "promotion_allowed": False,
                "future_oos_fit_operations_performed": 0,
                "next_action": "wait_for_future_oos_preflight_and_mature_labeled_rows",
                "historical_fit_metadata_recovered": bool(historical),
            }
            patch = {
                "status": "manifest_already_pinned",
                "candidate_id": primary_id,
                "manual_pin_required": False,
                "stage_1_unpinned_config_patch": {},
                "stage_2_pin_instruction": (
                    "No config patch is required. Verify the pinned candidate "
                    "and source artifacts through future_oos_preflight.json."
                ),
            }
        else:
            payload = {
                "available": False,
                "enabled": None,
                "status": "replacement_fit_not_run",
                "manifest_pin_required": False,
                "promotion_allowed": False,
                "next_action": (
                    "run_notebook_04_with_historical_recency_research_enabled; "
                    "if historical gates pass, fit the replacement candidate"
                ),
            }
            patch = {
                "status": "replacement_fit_not_run",
                "manual_pin_required": False,
                "stage_1_unpinned_config_patch": {},
                "stage_2_pin_instruction": (
                    "No replacement fit artifact exists yet, so no frozen "
                    "candidate manifest can be pinned."
                ),
            }
        _write_json(target / "replacement_candidate_fit.json", payload)
        (target / "replacement_candidate_fit.md").write_text(
            _replacement_markdown(payload),
            encoding="utf-8",
        )
        _write_json(target / "replacement_preregistration_patch.json", patch)
        (target / "replacement_preregistration_patch.md").write_text(
            _preregistration_patch_markdown(patch),
            encoding="utf-8",
        )
        return payload
    for name in (
        "replacement_candidate_fit.json",
        "replacement_candidate_fit.md",
        "replacement_preregistration_patch.json",
        "replacement_preregistration_patch.md",
    ):
        item = source / name
        if item.exists():
            _durable_copy_file(item, target / name)
    return json.loads(payload_path.read_text(encoding="utf-8"))
