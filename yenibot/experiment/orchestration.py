"""Top-level training matrix and diagnostics orchestration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import pandas as pd
from yenibot.diagnostics import (
    write_phase1_diagnostic_bundle,
)

from yenibot.experiment.artifacts import (
    _write_experiment_bundle,
    _write_experiment_slim_bundle,
)

from yenibot.experiment.classification import (
    _causal_threshold_policy_frames,
    _classification_skill_frames,
    _threshold_score_quantile_review_frames,
    _validation_charter_proposal_frame,
    _validation_charter_review_frame,
    _write_causal_threshold_policy,
    _write_classification_skill,
    _write_threshold_score_quantile_review,
    _write_validation_charter_proposal,
    _write_validation_charter_review,
)

from yenibot.experiment.common import (
    _cfg,
    _hash_payload,
    _json_ready,
    _read_json,
    _set_cfg,
    _slug,
    _write_json,
)

from yenibot.experiment.configuration import (
    _TRAINING_EXECUTION_KEYS,
    _apply_experiment_policy_guard,
    _experiment_signature,
    _load_training_execution_summary,
    _missing_selected_profiles,
    _preflight_experiment_profiles,
    _preflight_fold_plans,
    _training_execution_summary,
    _training_execution_summary_path,
    _write_experiment_selection,
    _write_missing_selected_profiles,
    experiment_root,
    experiment_settings,
    latest_experiment_run,
    profile_config,
    resolve_experiment_run_id,
)

from yenibot.experiment.drift import (
    _feature_drift_forensics_frame,
    _feature_family_drift_summary_frame,
    _fold_reliability_gate_frame,
    _fold_reliability_gate_summary_frame,
    _probability_quality_forensics_frame,
    _probability_quality_summary_frame,
    _score_distribution_shift_frame,
    _score_distribution_shift_summary_frame,
    _score_reversal_context_audit_frame,
    _write_feature_drift_forensics,
    _write_fold_reliability_gate,
    _write_forensics_reports,
    _write_probability_quality_forensics,
    _write_score_distribution_shift,
    _write_score_reversal_context_audit,
    _write_score_separation_forensics,
)

from yenibot.experiment.ensembles import (
    _best_profile_blend,
    _profile_blend_entries,
    _profile_blend_frame,
    _profile_blend_leaders,
    _profile_blend_review_frame,
    _profile_delta_vs_control,
    _seed_audit_coverage_frame,
    _seed_audit_entries_to_frames,
    _seed_audit_scope,
    _seed_ensemble_entries,
    _seed_ensemble_frame,
    _write_profile_blend_files,
    _write_profile_delta,
    _write_profile_diagnostic_summaries,
    _write_seed_audit_files,
    _write_seed_ensemble_files,
)

from yenibot.experiment.execution import (
    diagnostics_status_path,
    traced_workflow,
    training_status_path,
    workflow_checkpoint,
)

from yenibot.experiment.folds import (
    _fold_stability_forensics_frame,
    _fold_stability_summary_frame,
)

from yenibot.experiment.holdout import (
    _aggregate_holdout_predictions,
    _attach_holdout_cv_threshold_metrics,
    _attach_holdout_policy_consistency,
    _attach_holdout_policy_metrics,
    _attach_holdout_soft_pass,
    _experiment_policy_guard_frame,
    _frozen_policy_monitoring_plan_frame,
    _future_oos_candidate_plan_frame,
    _holdout_boundary_audit_frame,
    _holdout_policy_decision_frame,
    _performance_gap_analysis_frame,
    _predict_holdout_for_profile,
    _read_holdout_context,
    _recommendation_with_policy_guard,
    _resolve_holdout_settings,
    _selection_frame_before_holdout,
    _write_experiment_policy_guard,
    _write_frozen_policy_monitoring_plan,
    _write_future_oos_candidate_plan,
    _write_holdout_boundary_audit,
    _write_holdout_files,
    _write_holdout_reservation,
    _write_performance_gap_analysis,
)

from yenibot.experiment.payoff import (
    _frozen_policy_robustness_frame,
    _payoff_alignment_frame,
    _payoff_alignment_summary_frame,
    _payoff_policy_robustness_frame,
    _payoff_policy_robustness_summary_frame,
    _write_frozen_policy_robustness,
    _write_payoff_alignment,
    _write_payoff_policy_robustness,
)

from yenibot.experiment.rank_ic import (
    _rank_ic_stability_evidence_frames,
    _rank_ic_uncertainty_frames,
    _write_rank_ic_stability_evidence,
    _write_rank_ic_uncertainty,
)

from yenibot.experiment.root_cause import (
    _bad_fold_mechanism_summary_frame,
    _historical_experiment_memory_audit_frame,
    _phase1_blocker_action_plan_frame,
    _phase1_blocker_root_cause_frame,
    _phase1_decision_ladder_payload,
    _prediction_error_audit_frame,
    _threshold_oracle_gap_frame,
    _write_phase1_blocker_action_plan,
    _write_root_cause_reports,
)

from yenibot.experiment.separation import (
    _bad_fold_signature_frame,
    _score_separation_forensics_frame,
)

from yenibot.experiment.thresholds import (
    _regime_stability_frames,
    _regime_threshold_policy_frames,
    _threshold_forensics_frame,
    _threshold_policy_review_frame,
    _threshold_transfer_review_frames,
    _write_regime_stability,
    _write_regime_threshold_policy,
    _write_threshold_policy_review,
    _write_threshold_transfer_review,
)

from yenibot.experiment.training import (
    _auto_full_profiles,
    _best_candidate,
    _comparison_frame,
    _decision_rows,
    _test_predictions,
    _write_decision_files,
    run_profile_experiment,
    summarize_profile_predictions,
)

__all__ = [
    '_evaluate_holdout_candidates',
    'run_experiment_matrix',
    '_profile_dirs',
    'write_experiment_diagnostics',
]

def _evaluate_holdout_candidates(
    *,
    profile_entries: list[dict[str, Any]],
    cv_blend_entries: list[dict[str, Any]] | None = None,
    settings: dict[str, Any],
    config: dict[str, Any],
    decision: dict[str, Any],
    holdout_boundary_passed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    holdout_context, holdout_start = _read_holdout_context(settings, config)
    if holdout_context.empty or holdout_start is None:
        holdout_decision = {
            "available": False,
            "reason": "missing_holdout_frame_or_metadata",
            "policy": "holdout result must remain separate from profile selection",
            "holdout_boundary_passed": bool(holdout_boundary_passed),
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
        holdout_diagnostics = dict(diagnostics)
        holdout_diagnostics["row"] = row
        holdout_entries.append(
            {
                **entry,
                "fold_scope": str(entry.get("fold_scope", "")),
                "diagnostics": holdout_diagnostics,
                "summary": row,
            }
        )
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
            "holdout_boundary_passed": bool(holdout_boundary_passed),
        }
        return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries

    available_candidates = set(holdout_evaluation["candidate"].astype(str))
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    configured_frozen = str(policy_review.get("frozen_candidate", "")).strip()
    configured_available = bool(configured_frozen and configured_frozen in available_candidates)
    frozen_selection = str(settings.get("control_profile", ""))
    frozen_selection_source = "control_profile"
    best_blend = decision.get("best_profile_blend") or {}
    best_candidate = decision.get("best_candidate") or {}
    if configured_available:
        frozen_selection = configured_frozen
        frozen_selection_source = "configured_policy_review"
    elif configured_frozen:
        frozen_selection_source = "configured_policy_review_missing_fallback_control_profile"
    elif best_blend:
        frozen_selection = str(best_blend.get("profile") or frozen_selection)
        frozen_selection_source = "best_profile_blend"
    elif best_candidate:
        frozen_selection = str(best_candidate.get("profile") or frozen_selection)
        frozen_selection_source = "best_candidate"
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
        "holdout_boundary_passed": bool(holdout_boundary_passed),
        "holdout_start": str(pd.to_datetime(holdout_start, utc=True)),
        "holdout_rows": int(len(holdout_context.loc[pd.to_datetime(holdout_context["timestamp"], utc=True) >= holdout_start])),
        "candidate_count": int(len(holdout_evaluation)),
        "frozen_selection": frozen_selection,
        "frozen_selection_source": frozen_selection_source,
        "configured_frozen_candidate_available": configured_available,
        "frozen_selection_metrics": _json_ready(frozen_row),
        "frozen_policy_validation": _json_ready(frozen_row),
        "observed_best_holdout_candidate": _json_ready(observed_best),
        "observed_best_holdout_warning": observed_best_warning,
        "observed_best_policy_candidate": _json_ready(observed_best_policy),
        "observed_best_policy_warning": observed_best_policy_warning,
        "score_policy_recommendation": score_policy_recommendation,
    }
    policy_validation = _holdout_policy_decision_frame(holdout_decision, config)
    holdout_decision["policy_validation"] = (
        _json_ready(policy_validation.iloc[0].to_dict())
        if not policy_validation.empty
        else {}
    )
    return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries

@traced_workflow("training_matrix", training_status_path)
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
    settings = _apply_experiment_policy_guard(settings, config)
    frame = _selection_frame_before_holdout(frame, settings)
    settings = _preflight_experiment_profiles(settings, frame, config)
    settings = _apply_experiment_policy_guard(settings, config)
    available_fold_ids = _preflight_fold_plans(frame, settings, config)
    signature = _experiment_signature(config, settings)
    signature_hash = _hash_payload(signature)
    run_id, run_id_source = resolve_experiment_run_id(checkpoint_dir, config, settings, run_id)
    run_dir = experiment_root(checkpoint_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    workflow_checkpoint(
        "write_run_manifest",
        status_path=run_dir / "workflow_status.json",
        run_id=run_id,
        run_id_source=run_id_source,
    )
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
    experiment_policy_guard = _experiment_policy_guard_frame(settings, config)
    _write_experiment_policy_guard(run_dir, experiment_policy_guard)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(settings, config)
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)

    triage_fold_ids = [int(fold_id) for fold_id in settings.get("triage_fold_ids", [])]
    resume_existing = bool(settings.get("resume_existing", True))
    force_retrain = bool(settings.get("force_retrain", False))
    rows: list[dict[str, Any]] = []
    profile_results = []
    workflow_checkpoint(
        "train_triage_profiles",
        profile_count=len(settings["profiles"]),
        fold_count=len(triage_fold_ids),
    )
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
    workflow_checkpoint("train_full_profiles", profiles=list(dict.fromkeys(full_profiles)))
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
    workflow_checkpoint(
        "train_seed_audit",
        enabled=bool(seed_audit_cfg.get("enabled", False)),
        profiles=seed_audit_cfg.get("profiles", []),
        seeds=seed_audit_cfg.get("seeds", []),
    )
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
    executed_results = [result for result in [*profile_results, *seed_results] if not bool(result.get("skipped", False))]
    skipped_results = [result for result in [*profile_results, *seed_results] if bool(result.get("skipped", False))]
    seed_audit, seed_stability = _seed_audit_entries_to_frames(all_results)
    seed_audit_coverage = _seed_audit_coverage_frame(
        all_results,
        settings,
        available_fold_ids=available_fold_ids,
    )
    seed_ensemble = _seed_ensemble_frame(all_results)
    profile_blend = _profile_blend_frame(all_results)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, config, settings["control_profile"])
    workflow_checkpoint("compute_training_diagnostics", result_count=len(all_results))
    performance_gap_analysis = _performance_gap_analysis_frame(
        all_results,
        pd.DataFrame(),
        config,
        settings,
    )
    fold_stability_forensics = _fold_stability_forensics_frame(all_results, config)
    fold_stability_summary = _fold_stability_summary_frame(fold_stability_forensics, config)
    score_separation_forensics = _score_separation_forensics_frame(all_results, config)
    bad_fold_signature = _bad_fold_signature_frame(score_separation_forensics, config)
    feature_drift_forensics = _feature_drift_forensics_frame(all_results, score_separation_forensics, config)
    feature_family_drift_summary = _feature_family_drift_summary_frame(feature_drift_forensics)
    probability_quality_forensics = _probability_quality_forensics_frame(all_results, config)
    probability_quality_summary = _probability_quality_summary_frame(probability_quality_forensics, config)
    score_distribution_shift = _score_distribution_shift_frame(all_results, config)
    score_distribution_shift_summary = _score_distribution_shift_summary_frame(score_distribution_shift, config)
    fold_reliability_gate = _fold_reliability_gate_frame(all_results, config)
    fold_reliability_gate_summary = _fold_reliability_gate_summary_frame(fold_reliability_gate, config)
    regime_threshold_policy_by_fold, regime_threshold_policy_summary = _regime_threshold_policy_frames(
        all_results,
        config,
    )
    regime_stability_forensics, regime_stability_summary = _regime_stability_frames(all_results, config)
    threshold_forensics = _threshold_forensics_frame(all_results, config)
    threshold_policy_review = _threshold_policy_review_frame(all_results, config)
    threshold_transfer_review, threshold_transfer_by_fold = _threshold_transfer_review_frames(all_results, config)
    threshold_score_quantile_review, threshold_score_quantile_by_fold = _threshold_score_quantile_review_frames(
        all_results,
        config,
    )
    rank_ic_variance_decomposition, rank_ic_sampling_uncertainty = _rank_ic_uncertainty_frames(
        all_results,
        config,
    )
    causal_threshold_policy_summary, causal_threshold_policy_by_fold = _causal_threshold_policy_frames(
        all_results,
        config,
    )
    payoff_alignment = _payoff_alignment_frame(all_results, [], config)
    payoff_alignment_summary = _payoff_alignment_summary_frame(payoff_alignment)
    payoff_policy_robustness = _payoff_policy_robustness_frame(all_results, [], config)
    payoff_policy_robustness_summary = _payoff_policy_robustness_summary_frame(payoff_policy_robustness, config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(
        settings,
        config,
        payoff_policy_robustness_summary,
    )
    phase1_blocker_action_plan = _phase1_blocker_action_plan_frame(
        comparison=comparison,
        profile_blend=profile_blend,
        performance_gap_analysis=performance_gap_analysis,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
        payoff_policy_robustness_summary=payoff_policy_robustness_summary,
        future_oos_candidate_plan=future_oos_candidate_plan,
        phase2_readiness={},
        config=config,
        settings=settings,
    )
    threshold_oracle_gap = _threshold_oracle_gap_frame(threshold_forensics, config)
    historical_experiment_memory_audit = _historical_experiment_memory_audit_frame(
        feature_family_drift_summary,
        config,
    )
    score_reversal_context_audit = _score_reversal_context_audit_frame(
        feature_drift_forensics,
        historical_experiment_memory_audit,
        config,
    )
    bad_fold_mechanism_summary = _bad_fold_mechanism_summary_frame(
        bad_fold_signature=bad_fold_signature,
        feature_family_drift_summary=feature_family_drift_summary,
        score_distribution_shift_summary=score_distribution_shift_summary,
        probability_quality_summary=probability_quality_summary,
        historical_memory_audit=historical_experiment_memory_audit,
        config=config,
    )
    prediction_error_audit = _prediction_error_audit_frame(
        all_results,
        score_separation_forensics,
        config,
    )
    phase1_blocker_root_cause = _phase1_blocker_root_cause_frame(
        phase1_blocker_action_plan=phase1_blocker_action_plan,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        historical_experiment_memory_audit=historical_experiment_memory_audit,
        phase2_readiness={},
        settings=settings,
        config=config,
    )
    phase1_decision_ladder = _phase1_decision_ladder_payload(
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        phase2_readiness={},
        settings=settings,
    )
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability, seed_audit_coverage)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_performance_gap_analysis(run_dir, performance_gap_analysis)
    _write_phase1_blocker_action_plan(run_dir, phase1_blocker_action_plan)
    _write_root_cause_reports(
        run_dir,
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        prediction_error_audit=prediction_error_audit,
        historical_experiment_memory_audit=historical_experiment_memory_audit,
        decision_ladder=phase1_decision_ladder,
    )
    _write_forensics_reports(
        run_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(run_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(run_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_score_reversal_context_audit(run_dir, score_reversal_context_audit)
    _write_probability_quality_forensics(run_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(run_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(run_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(run_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(run_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(run_dir, threshold_policy_review)
    _write_threshold_transfer_review(run_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_threshold_score_quantile_review(run_dir, threshold_score_quantile_review, threshold_score_quantile_by_fold)
    _write_rank_ic_uncertainty(run_dir, rank_ic_variance_decomposition, rank_ic_sampling_uncertainty)
    _write_causal_threshold_policy(run_dir, causal_threshold_policy_summary, causal_threshold_policy_by_fold)
    _write_payoff_alignment(run_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(run_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    profile_delta = _profile_delta_vs_control(profile_results, settings["control_profile"])
    best = _best_candidate(comparison, settings["control_profile"])
    blend_leaders = _profile_blend_leaders(profile_blend)
    best_blend = _best_profile_blend(profile_blend)
    missing_selected = _missing_selected_profiles(experiment_selection, comparison)
    _write_missing_selected_profiles(run_dir, missing_selected)
    workflow_checkpoint(
        "write_training_artifacts",
        missing_selected_profiles=len(missing_selected),
    )
    training_execution = _training_execution_summary(
        run_id=run_id,
        run_id_source=run_id_source,
        executed_results=executed_results,
        skipped_results=skipped_results,
        profile_results=profile_results,
        seed_results=seed_results,
    )
    _write_json(_training_execution_summary_path(run_dir), training_execution)
    recommendation = "fix_missing_selected_profiles" if not missing_selected.empty else (
        "promote_best_candidate" if best else ("review_profile_blend" if best_blend else "keep_control_profile")
    )
    decision = {
        "run_id": run_id,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "full_profiles": full_profiles,
        "seed_audit_profiles": [str(profile) for profile in seed_audit_cfg.get("profiles", [])] if seed_audit_cfg else [],
        "seed_audit_seeds": [int(seed) for seed in seed_audit_cfg.get("seeds", [])] if seed_audit_cfg else [],
        "seed_audit_coverage": seed_audit_coverage.to_dict(orient="records"),
        "skipped_profiles": settings.get("skipped_profiles", []) or [],
        **{key: training_execution[key] for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution["executed_training_scopes"],
        "training_execution_metadata_source": "run_experiment_matrix",
        "training_execution_metadata_available": True,
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout": settings.get("holdout", {}) or {},
        "experiment_policy_guard": settings.get("experiment_policy_guard", {}) or {},
        "future_oos_candidate_plan": future_oos_candidate_plan.to_dict(orient="records"),
        "performance_gap_analysis": performance_gap_analysis.to_dict(orient="records"),
        "phase1_blocker_action_plan": phase1_blocker_action_plan.to_dict(orient="records"),
        "phase1_blocker_root_cause": phase1_blocker_root_cause.to_dict(orient="records"),
        "threshold_oracle_gap": threshold_oracle_gap.to_dict(orient="records"),
        "bad_fold_mechanism_summary": bad_fold_mechanism_summary.to_dict(orient="records"),
        "prediction_error_audit": prediction_error_audit.to_dict(orient="records"),
        "historical_experiment_memory_audit": historical_experiment_memory_audit.to_dict(orient="records"),
        "score_reversal_context_audit": score_reversal_context_audit.to_dict(orient="records"),
        "phase1_decision_ladder": phase1_decision_ladder,
        "fold_stability_summary": fold_stability_summary.to_dict(orient="records"),
        "score_separation_forensics": score_separation_forensics.to_dict(orient="records"),
        "bad_fold_signature": bad_fold_signature.to_dict(orient="records"),
        "feature_family_drift_summary": feature_family_drift_summary.to_dict(orient="records"),
        "probability_quality_summary": probability_quality_summary.to_dict(orient="records"),
        "score_distribution_shift_summary": score_distribution_shift_summary.to_dict(orient="records"),
        "fold_reliability_gate_summary": fold_reliability_gate_summary.to_dict(orient="records"),
        "regime_threshold_policy_summary": regime_threshold_policy_summary.to_dict(orient="records"),
        "regime_stability_summary": regime_stability_summary.to_dict(orient="records"),
        "threshold_forensics_summary": (
            threshold_forensics["primary_issue"].value_counts().to_dict()
            if not threshold_forensics.empty and "primary_issue" in threshold_forensics.columns
            else {}
        ),
        "threshold_policy_review": threshold_policy_review.to_dict(orient="records"),
        "threshold_transfer_review": threshold_transfer_review.to_dict(orient="records"),
        "threshold_score_quantile_review": threshold_score_quantile_review.to_dict(orient="records"),
        "rank_ic_variance_decomposition": rank_ic_variance_decomposition.to_dict(orient="records"),
        "causal_threshold_policy_summary": causal_threshold_policy_summary.to_dict(orient="records"),
        "payoff_alignment_summary": payoff_alignment_summary.to_dict(orient="records"),
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary.to_dict(orient="records"),
        "recommendation": _recommendation_with_policy_guard(recommendation, settings),
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
        "seed_audit_coverage": seed_audit_coverage,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "performance_gap_analysis": performance_gap_analysis,
        "phase1_blocker_action_plan": phase1_blocker_action_plan,
        "phase1_blocker_root_cause": phase1_blocker_root_cause,
        "threshold_oracle_gap": threshold_oracle_gap,
        "bad_fold_mechanism_summary": bad_fold_mechanism_summary,
        "prediction_error_audit": prediction_error_audit,
        "historical_experiment_memory_audit": historical_experiment_memory_audit,
        "phase1_decision_ladder": phase1_decision_ladder,
        "fold_stability_forensics": fold_stability_forensics,
        "fold_stability_summary": fold_stability_summary,
        "score_separation_forensics": score_separation_forensics,
        "bad_fold_signature": bad_fold_signature,
        "feature_drift_forensics": feature_drift_forensics,
        "feature_family_drift_summary": feature_family_drift_summary,
        "score_reversal_context_audit": score_reversal_context_audit,
        "probability_quality_forensics": probability_quality_forensics,
        "probability_quality_summary": probability_quality_summary,
        "score_distribution_shift": score_distribution_shift,
        "score_distribution_shift_summary": score_distribution_shift_summary,
        "fold_reliability_gate": fold_reliability_gate,
        "fold_reliability_gate_summary": fold_reliability_gate_summary,
        "regime_threshold_policy_by_fold": regime_threshold_policy_by_fold,
        "regime_threshold_policy_summary": regime_threshold_policy_summary,
        "regime_stability_forensics": regime_stability_forensics,
        "regime_stability_summary": regime_stability_summary,
        "threshold_forensics": threshold_forensics,
        "threshold_policy_review": threshold_policy_review,
        "threshold_transfer_review": threshold_transfer_review,
        "threshold_transfer_by_fold": threshold_transfer_by_fold,
        "threshold_score_quantile_review": threshold_score_quantile_review,
        "threshold_score_quantile_by_fold": threshold_score_quantile_by_fold,
        "rank_ic_variance_decomposition": rank_ic_variance_decomposition,
        "rank_ic_sampling_uncertainty": rank_ic_sampling_uncertainty,
        "causal_threshold_policy_summary": causal_threshold_policy_summary,
        "causal_threshold_policy_by_fold": causal_threshold_policy_by_fold,
        "payoff_alignment": payoff_alignment,
        "payoff_alignment_summary": payoff_alignment_summary,
        "payoff_policy_robustness": payoff_policy_robustness,
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "experiment_policy_guard": experiment_policy_guard,
        "future_oos_candidate_plan": future_oos_candidate_plan,
        **{key: training_execution[key] for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution["executed_training_scopes"],
        "training_execution_metadata_source": "run_experiment_matrix",
        "training_execution_metadata_available": True,
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

@traced_workflow("diagnostics", diagnostics_status_path)
def write_experiment_diagnostics(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    run_id: str | None = None,
    write_full_bundles: bool | None = None,
) -> dict[str, Any]:
    if write_full_bundles is None:
        write_full_bundles = bool(
            _cfg(config, ["experiments", "diagnostics", "write_full_bundles"], default=False)
        )
    run_dir = experiment_root(checkpoint_dir) / run_id if run_id else latest_experiment_run(checkpoint_dir)
    workflow_checkpoint(
        "load_training_outputs",
        status_path=run_dir / "diagnostics_workflow_status.json",
        run_id=run_dir.name,
    )
    run_manifest_path = run_dir / "experiment_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    training_execution = _load_training_execution_summary(run_dir, run_manifest)
    settings = copy.deepcopy(run_manifest.get("settings") or experiment_settings(config))
    settings = _resolve_holdout_settings(settings, config)
    diagnostic_config = copy.deepcopy(config)
    current_experiment_cfg = copy.deepcopy(_cfg(diagnostic_config, ["experiments"], default={}) or {})
    experiment_cfg = copy.deepcopy(current_experiment_cfg)
    experiment_cfg.update(settings)
    # Training run settings are historical metadata, but policy review is a
    # diagnostics-time decision. Keep it sourced from the current config so a
    # failed/retired frozen policy cannot be resurrected by an old run manifest.
    if "policy_review" in current_experiment_cfg:
        experiment_cfg["policy_review"] = copy.deepcopy(current_experiment_cfg["policy_review"])
    _set_cfg(diagnostic_config, ["experiments"], experiment_cfg)
    settings = _apply_experiment_policy_guard(settings, diagnostic_config)
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
    workflow_checkpoint("build_profile_summaries", profile_scope_count=len(profile_entries))
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
    seed_audit_coverage = _seed_audit_coverage_frame(entries, settings)
    seed_ensemble = _seed_ensemble_frame(entries)
    profile_blend = _profile_blend_frame(entries)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, diagnostic_config, settings["control_profile"])
    holdout_boundary_audit = _holdout_boundary_audit_frame(entries, settings)
    holdout_boundary_passed = bool(holdout_boundary_audit.empty or holdout_boundary_audit["passed"].astype(bool).all())
    frozen_policy_robustness = _frozen_policy_robustness_frame(entries, diagnostic_config)
    frozen_policy_monitoring_plan = _frozen_policy_monitoring_plan_frame(diagnostic_config, settings)
    experiment_policy_guard = _experiment_policy_guard_frame(settings, diagnostic_config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(settings, diagnostic_config)
    decision_lookup = {
        (str(row["profile"]), str(row["fold_scope"])): row
        for row in [*triage_rows, *full_rows]
    }

    zip_paths = []
    if write_full_bundles:
        workflow_checkpoint("write_full_profile_bundles", profile_scope_count=len(entries))
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
    if not missing_selected.empty:
        recommendation = "fix_missing_selected_profiles"
    elif not holdout_boundary_passed:
        recommendation = "rerun_training_with_holdout_split"
    elif best:
        recommendation = "promote_best_candidate"
    elif best_blend:
        recommendation = "review_profile_blend"
    else:
        recommendation = "keep_control_profile"
    guarded_recommendation = _recommendation_with_policy_guard(recommendation, settings)
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
        **{key: training_execution.get(key) for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution.get("executed_training_scopes", []),
        "training_execution_metadata_source": training_execution.get("training_execution_metadata_source"),
        "training_execution_metadata_available": bool(training_execution.get("training_execution_metadata_available", False)),
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout_boundary_passed": holdout_boundary_passed,
        "holdout_boundary_audit": holdout_boundary_audit.to_dict(orient="records"),
        "frozen_policy_robustness": frozen_policy_robustness.to_dict(orient="records"),
        "frozen_policy_monitoring_plan": frozen_policy_monitoring_plan.to_dict(orient="records"),
        "experiment_policy_guard": settings.get("experiment_policy_guard", {}) or {},
        "future_oos_candidate_plan": future_oos_candidate_plan.to_dict(orient="records"),
        "holdout": settings.get("holdout", {}) or {},
        "recommendation": guarded_recommendation,
        "diagnostic_zips": zip_paths,
    }
    holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries = (
        _evaluate_holdout_candidates(
            profile_entries=profile_entries,
            cv_blend_entries=profile_blend_entries,
            settings=settings,
            config=diagnostic_config,
            decision=decision,
            holdout_boundary_passed=holdout_boundary_passed,
        )
    )
    workflow_checkpoint(
        "evaluate_holdout_policy",
        holdout_available=bool(holdout_decision.get("available", False)),
    )
    decision["holdout_evaluation"] = holdout_decision
    decision["holdout_evaluation_available"] = bool(holdout_decision.get("available", False))
    performance_gap_analysis = _performance_gap_analysis_frame(
        entries,
        holdout_evaluation,
        diagnostic_config,
        settings,
    )
    fold_stability_forensics = _fold_stability_forensics_frame(entries, diagnostic_config)
    fold_stability_summary = _fold_stability_summary_frame(fold_stability_forensics, diagnostic_config)
    score_separation_forensics = _score_separation_forensics_frame(entries, diagnostic_config)
    bad_fold_signature = _bad_fold_signature_frame(score_separation_forensics, diagnostic_config)
    feature_drift_forensics = _feature_drift_forensics_frame(entries, score_separation_forensics, diagnostic_config)
    feature_family_drift_summary = _feature_family_drift_summary_frame(feature_drift_forensics)
    probability_quality_forensics = _probability_quality_forensics_frame(entries, diagnostic_config)
    probability_quality_summary = _probability_quality_summary_frame(probability_quality_forensics, diagnostic_config)
    score_distribution_shift = _score_distribution_shift_frame(entries, diagnostic_config)
    score_distribution_shift_summary = _score_distribution_shift_summary_frame(score_distribution_shift, diagnostic_config)
    fold_reliability_gate = _fold_reliability_gate_frame(entries, diagnostic_config)
    fold_reliability_gate_summary = _fold_reliability_gate_summary_frame(fold_reliability_gate, diagnostic_config)
    regime_threshold_policy_by_fold, regime_threshold_policy_summary = _regime_threshold_policy_frames(
        entries,
        diagnostic_config,
    )
    regime_stability_forensics, regime_stability_summary = _regime_stability_frames(entries, diagnostic_config)
    threshold_forensics = _threshold_forensics_frame(entries, diagnostic_config)
    threshold_policy_review = _threshold_policy_review_frame(entries, diagnostic_config)
    threshold_transfer_review, threshold_transfer_by_fold = _threshold_transfer_review_frames(entries, diagnostic_config)
    threshold_score_quantile_review, threshold_score_quantile_by_fold = _threshold_score_quantile_review_frames(
        entries,
        diagnostic_config,
    )
    rank_ic_variance_decomposition, rank_ic_sampling_uncertainty = _rank_ic_uncertainty_frames(
        entries,
        diagnostic_config,
    )
    rank_ic_aggregate_evidence, rank_ic_block_sensitivity = _rank_ic_stability_evidence_frames(
        entries,
        diagnostic_config,
    )
    causal_threshold_policy_summary, causal_threshold_policy_by_fold = _causal_threshold_policy_frames(
        entries,
        diagnostic_config,
    )
    classification_skill_summary, classification_skill_by_fold = _classification_skill_frames(
        entries,
        causal_threshold_policy_by_fold,
        diagnostic_config,
    )
    workflow_checkpoint("compute_root_cause_diagnostics")
    validation_charter_review = _validation_charter_review_frame(
        control_profile=settings["control_profile"],
        rank_ic_evidence=rank_ic_aggregate_evidence,
        classification_skill_summary=classification_skill_summary,
        config=diagnostic_config,
    )
    validation_charter_proposal = _validation_charter_proposal_frame(
        control_profile=settings["control_profile"],
        rank_ic_evidence=rank_ic_aggregate_evidence,
        classification_skill_summary=classification_skill_summary,
        config=diagnostic_config,
    )
    payoff_alignment = _payoff_alignment_frame(entries, holdout_entries, diagnostic_config)
    payoff_alignment_summary = _payoff_alignment_summary_frame(payoff_alignment)
    payoff_policy_robustness = _payoff_policy_robustness_frame(entries, holdout_entries, diagnostic_config)
    payoff_policy_robustness_summary = _payoff_policy_robustness_summary_frame(payoff_policy_robustness, diagnostic_config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(
        settings,
        diagnostic_config,
        payoff_policy_robustness_summary,
    )
    decision["future_oos_candidate_plan"] = future_oos_candidate_plan.to_dict(orient="records")
    decision["performance_gap_analysis"] = performance_gap_analysis.to_dict(orient="records")
    decision["fold_stability_summary"] = fold_stability_summary.to_dict(orient="records")
    decision["bad_fold_signature"] = bad_fold_signature.to_dict(orient="records")
    decision["feature_family_drift_summary"] = feature_family_drift_summary.to_dict(orient="records")
    decision["probability_quality_summary"] = probability_quality_summary.to_dict(orient="records")
    decision["score_distribution_shift_summary"] = score_distribution_shift_summary.to_dict(orient="records")
    decision["fold_reliability_gate_summary"] = fold_reliability_gate_summary.to_dict(orient="records")
    decision["regime_threshold_policy_summary"] = regime_threshold_policy_summary.to_dict(orient="records")
    decision["regime_stability_summary"] = regime_stability_summary.to_dict(orient="records")
    decision["threshold_forensics_summary"] = (
        threshold_forensics["primary_issue"].value_counts().to_dict()
        if not threshold_forensics.empty and "primary_issue" in threshold_forensics.columns
        else {}
    )
    decision["threshold_policy_review"] = threshold_policy_review.to_dict(orient="records")
    decision["threshold_transfer_review"] = threshold_transfer_review.to_dict(orient="records")
    decision["threshold_score_quantile_review"] = threshold_score_quantile_review.to_dict(orient="records")
    decision["rank_ic_variance_decomposition"] = rank_ic_variance_decomposition.to_dict(orient="records")
    decision["rank_ic_aggregate_evidence"] = rank_ic_aggregate_evidence.to_dict(orient="records")
    decision["causal_threshold_policy_summary"] = causal_threshold_policy_summary.to_dict(orient="records")
    decision["classification_skill_summary"] = classification_skill_summary.to_dict(orient="records")
    decision["validation_charter_review"] = validation_charter_review.to_dict(orient="records")
    decision["validation_charter_proposal"] = validation_charter_proposal.to_dict(orient="records")
    decision["seed_audit_coverage"] = seed_audit_coverage.to_dict(orient="records")
    decision["payoff_alignment_summary"] = payoff_alignment_summary.to_dict(orient="records")
    decision["payoff_policy_robustness_summary"] = payoff_policy_robustness_summary.to_dict(orient="records")
    bundle_path = Path(output_dir) / f"phase1_experiment_bundle_{run_dir.name}.zip" if write_full_bundles else None
    latest_bundle_path = Path(output_dir) / "phase1_latest_experiment_bundle.zip" if write_full_bundles else None
    slim_bundle_path = Path(output_dir) / f"phase1_experiment_slim_bundle_{run_dir.name}.zip"
    latest_slim_bundle_path = Path(output_dir) / "phase1_latest_experiment_slim_bundle.zip"
    decision["write_full_bundles"] = bool(write_full_bundles)
    decision["bundle_zip"] = str(bundle_path) if bundle_path is not None else None
    decision["latest_bundle_zip"] = str(latest_bundle_path) if latest_bundle_path is not None else None
    decision["slim_bundle_zip"] = str(slim_bundle_path)
    decision["latest_slim_bundle_zip"] = str(latest_slim_bundle_path)
    _write_decision_files(report_dir, comparison, decision)
    _write_json(report_dir / "training_execution_summary.json", training_execution)
    _write_profile_delta(report_dir, profile_delta)
    _write_seed_audit_files(report_dir, seed_audit, seed_stability, seed_audit_coverage)
    _write_seed_ensemble_files(report_dir, seed_ensemble)
    _write_profile_blend_files(report_dir, profile_blend)
    _write_performance_gap_analysis(report_dir, performance_gap_analysis)
    _write_forensics_reports(
        report_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(report_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(report_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_probability_quality_forensics(report_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(report_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(report_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(report_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(report_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(report_dir, threshold_policy_review)
    _write_threshold_transfer_review(report_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_threshold_score_quantile_review(
        report_dir,
        threshold_score_quantile_review,
        threshold_score_quantile_by_fold,
    )
    _write_rank_ic_uncertainty(
        report_dir,
        rank_ic_variance_decomposition,
        rank_ic_sampling_uncertainty,
    )
    _write_rank_ic_stability_evidence(
        report_dir,
        rank_ic_aggregate_evidence,
        rank_ic_block_sensitivity,
    )
    _write_causal_threshold_policy(
        report_dir,
        causal_threshold_policy_summary,
        causal_threshold_policy_by_fold,
    )
    _write_classification_skill(
        report_dir,
        classification_skill_summary,
        classification_skill_by_fold,
    )
    _write_validation_charter_review(report_dir, validation_charter_review)
    _write_validation_charter_proposal(report_dir, validation_charter_proposal)
    _write_payoff_alignment(report_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(report_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    _write_profile_diagnostic_summaries(report_dir, entries)
    _write_holdout_boundary_audit(report_dir, holdout_boundary_audit)
    _write_frozen_policy_robustness(report_dir, frozen_policy_robustness)
    _write_frozen_policy_monitoring_plan(report_dir, frozen_policy_monitoring_plan)
    _write_experiment_policy_guard(report_dir, experiment_policy_guard)
    _write_future_oos_candidate_plan(report_dir, future_oos_candidate_plan)
    _write_holdout_files(
        report_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
        config=diagnostic_config,
    )
    # Prewrite root-cause diagnostics so the auto-review completeness audit can
    # require them. They are overwritten below after auto-review adds Phase 2 blockers.
    pre_phase1_blocker_action_plan = _phase1_blocker_action_plan_frame(
        comparison=comparison,
        profile_blend=profile_blend,
        performance_gap_analysis=performance_gap_analysis,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
        payoff_policy_robustness_summary=payoff_policy_robustness_summary,
        future_oos_candidate_plan=future_oos_candidate_plan,
        phase2_readiness={},
        config=diagnostic_config,
        settings=settings,
    )
    pre_threshold_oracle_gap = _threshold_oracle_gap_frame(threshold_forensics, diagnostic_config)
    pre_historical_experiment_memory_audit = _historical_experiment_memory_audit_frame(
        feature_family_drift_summary,
        diagnostic_config,
    )
    score_reversal_context_audit = _score_reversal_context_audit_frame(
        feature_drift_forensics,
        pre_historical_experiment_memory_audit,
        diagnostic_config,
    )
    decision["score_reversal_context_audit"] = score_reversal_context_audit.to_dict(orient="records")
    _write_score_reversal_context_audit(report_dir, score_reversal_context_audit)
    pre_bad_fold_mechanism_summary = _bad_fold_mechanism_summary_frame(
        bad_fold_signature=bad_fold_signature,
        feature_family_drift_summary=feature_family_drift_summary,
        score_distribution_shift_summary=score_distribution_shift_summary,
        probability_quality_summary=probability_quality_summary,
        historical_memory_audit=pre_historical_experiment_memory_audit,
        config=diagnostic_config,
    )
    pre_prediction_error_audit = _prediction_error_audit_frame(
        entries,
        score_separation_forensics,
        diagnostic_config,
    )
    pre_phase1_blocker_root_cause = _phase1_blocker_root_cause_frame(
        phase1_blocker_action_plan=pre_phase1_blocker_action_plan,
        threshold_oracle_gap=pre_threshold_oracle_gap,
        bad_fold_mechanism_summary=pre_bad_fold_mechanism_summary,
        historical_experiment_memory_audit=pre_historical_experiment_memory_audit,
        phase2_readiness={},
        settings=settings,
        config=diagnostic_config,
    )
    pre_phase1_decision_ladder = _phase1_decision_ladder_payload(
        phase1_blocker_root_cause=pre_phase1_blocker_root_cause,
        threshold_oracle_gap=pre_threshold_oracle_gap,
        bad_fold_mechanism_summary=pre_bad_fold_mechanism_summary,
        phase2_readiness={},
        settings=settings,
    )
    _write_phase1_blocker_action_plan(report_dir, pre_phase1_blocker_action_plan)
    _write_root_cause_reports(
        report_dir,
        phase1_blocker_root_cause=pre_phase1_blocker_root_cause,
        threshold_oracle_gap=pre_threshold_oracle_gap,
        bad_fold_mechanism_summary=pre_bad_fold_mechanism_summary,
        prediction_error_audit=pre_prediction_error_audit,
        historical_experiment_memory_audit=pre_historical_experiment_memory_audit,
        decision_ladder=pre_phase1_decision_ladder,
    )
    from yenibot.automation import write_auto_review

    workflow_checkpoint("run_auto_review", report_dir=report_dir)
    auto_review = write_auto_review(report_dir)
    auto_review_path = Path(auto_review["auto_review_path"])
    auto_review_json_path = Path(auto_review["auto_review_json_path"])
    next_actions_path = Path(auto_review["next_actions_path"])
    phase2_readiness_path = Path(auto_review["phase2_readiness_path"])
    phase2_readiness_md_path = Path(auto_review["phase2_readiness_md_path"])
    phase1_transition_plan_path = Path(auto_review["phase1_transition_plan_path"])
    phase1_transition_plan_md_path = Path(auto_review["phase1_transition_plan_md_path"])
    decision["auto_review_path"] = str(auto_review_path)
    decision["auto_review_json_path"] = str(auto_review_json_path)
    decision["next_actions_path"] = str(next_actions_path)
    decision["phase2_readiness_path"] = str(phase2_readiness_path)
    decision["phase2_readiness_md_path"] = str(phase2_readiness_md_path)
    decision["phase2_readiness"] = auto_review["review"].get("phase2_readiness", {})
    decision["phase1_transition_plan_path"] = str(phase1_transition_plan_path)
    decision["phase1_transition_plan_md_path"] = str(phase1_transition_plan_md_path)
    decision["phase1_transition_plan"] = auto_review["review"].get("phase1_transition_plan", {})
    phase1_blocker_action_plan = _phase1_blocker_action_plan_frame(
        comparison=comparison,
        profile_blend=profile_blend,
        performance_gap_analysis=performance_gap_analysis,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
        payoff_policy_robustness_summary=payoff_policy_robustness_summary,
        future_oos_candidate_plan=future_oos_candidate_plan,
        phase2_readiness=decision["phase2_readiness"],
        config=diagnostic_config,
        settings=settings,
    )
    threshold_oracle_gap = _threshold_oracle_gap_frame(threshold_forensics, diagnostic_config)
    historical_experiment_memory_audit = _historical_experiment_memory_audit_frame(
        feature_family_drift_summary,
        diagnostic_config,
    )
    bad_fold_mechanism_summary = _bad_fold_mechanism_summary_frame(
        bad_fold_signature=bad_fold_signature,
        feature_family_drift_summary=feature_family_drift_summary,
        score_distribution_shift_summary=score_distribution_shift_summary,
        probability_quality_summary=probability_quality_summary,
        historical_memory_audit=historical_experiment_memory_audit,
        config=diagnostic_config,
    )
    prediction_error_audit = _prediction_error_audit_frame(
        entries,
        score_separation_forensics,
        diagnostic_config,
    )
    phase1_blocker_root_cause = _phase1_blocker_root_cause_frame(
        phase1_blocker_action_plan=phase1_blocker_action_plan,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        historical_experiment_memory_audit=historical_experiment_memory_audit,
        phase2_readiness=decision["phase2_readiness"],
        settings=settings,
        config=diagnostic_config,
    )
    phase1_decision_ladder = _phase1_decision_ladder_payload(
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        phase2_readiness=decision["phase2_readiness"],
        settings=settings,
    )
    decision["phase1_blocker_action_plan"] = phase1_blocker_action_plan.to_dict(orient="records")
    decision["phase1_blocker_root_cause"] = phase1_blocker_root_cause.to_dict(orient="records")
    decision["threshold_oracle_gap"] = threshold_oracle_gap.to_dict(orient="records")
    decision["bad_fold_mechanism_summary"] = bad_fold_mechanism_summary.to_dict(orient="records")
    decision["prediction_error_audit"] = prediction_error_audit.to_dict(orient="records")
    decision["historical_experiment_memory_audit"] = historical_experiment_memory_audit.to_dict(orient="records")
    decision["phase1_decision_ladder"] = phase1_decision_ladder
    _write_phase1_blocker_action_plan(report_dir, phase1_blocker_action_plan)
    _write_root_cause_reports(
        report_dir,
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        prediction_error_audit=prediction_error_audit,
        historical_experiment_memory_audit=historical_experiment_memory_audit,
        decision_ladder=phase1_decision_ladder,
    )
    _write_decision_files(report_dir, comparison, decision)
    _write_decision_files(run_dir, comparison, decision)
    _write_json(_training_execution_summary_path(run_dir), training_execution)
    _write_profile_delta(run_dir, profile_delta)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability, seed_audit_coverage)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_performance_gap_analysis(run_dir, performance_gap_analysis)
    _write_phase1_blocker_action_plan(run_dir, phase1_blocker_action_plan)
    _write_root_cause_reports(
        run_dir,
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        prediction_error_audit=prediction_error_audit,
        historical_experiment_memory_audit=historical_experiment_memory_audit,
        decision_ladder=phase1_decision_ladder,
    )
    _write_forensics_reports(
        run_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(run_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(run_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_score_reversal_context_audit(run_dir, score_reversal_context_audit)
    _write_probability_quality_forensics(run_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(run_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(run_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(run_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(run_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(run_dir, threshold_policy_review)
    _write_threshold_transfer_review(run_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_threshold_score_quantile_review(run_dir, threshold_score_quantile_review, threshold_score_quantile_by_fold)
    _write_rank_ic_uncertainty(run_dir, rank_ic_variance_decomposition, rank_ic_sampling_uncertainty)
    _write_rank_ic_stability_evidence(run_dir, rank_ic_aggregate_evidence, rank_ic_block_sensitivity)
    _write_causal_threshold_policy(run_dir, causal_threshold_policy_summary, causal_threshold_policy_by_fold)
    _write_classification_skill(run_dir, classification_skill_summary, classification_skill_by_fold)
    _write_validation_charter_review(run_dir, validation_charter_review)
    _write_validation_charter_proposal(run_dir, validation_charter_proposal)
    _write_payoff_alignment(run_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(run_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    _write_profile_diagnostic_summaries(run_dir, entries)
    _write_experiment_selection(run_dir, settings)
    _write_holdout_reservation(run_dir, settings)
    _write_missing_selected_profiles(run_dir, missing_selected)
    _write_holdout_boundary_audit(run_dir, holdout_boundary_audit)
    _write_frozen_policy_robustness(run_dir, frozen_policy_robustness)
    _write_frozen_policy_monitoring_plan(run_dir, frozen_policy_monitoring_plan)
    _write_experiment_policy_guard(run_dir, experiment_policy_guard)
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)
    _write_holdout_files(
        run_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
        config=diagnostic_config,
    )
    if write_full_bundles:
        workflow_checkpoint("package_full_bundle")
        bundle_path, latest_bundle_path = _write_experiment_bundle(
            output_dir=Path(output_dir),
            run_id=run_dir.name,
            report_dir=report_dir,
            zip_paths=zip_paths,
        )
    else:
        bundle_path = None
        latest_bundle_path = None
    slim_bundle_path, latest_slim_bundle_path = _write_experiment_slim_bundle(
        output_dir=Path(output_dir),
        run_id=run_dir.name,
        report_dir=report_dir,
    )
    workflow_checkpoint("package_slim_bundle", slim_bundle=slim_bundle_path)
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "comparison": comparison,
        "profile_delta": profile_delta,
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_audit_coverage": seed_audit_coverage,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "performance_gap_analysis": performance_gap_analysis,
        "phase1_blocker_action_plan": phase1_blocker_action_plan,
        "phase1_blocker_root_cause": phase1_blocker_root_cause,
        "threshold_oracle_gap": threshold_oracle_gap,
        "bad_fold_mechanism_summary": bad_fold_mechanism_summary,
        "prediction_error_audit": prediction_error_audit,
        "historical_experiment_memory_audit": historical_experiment_memory_audit,
        "phase1_decision_ladder": phase1_decision_ladder,
        "fold_stability_forensics": fold_stability_forensics,
        "fold_stability_summary": fold_stability_summary,
        "score_separation_forensics": score_separation_forensics,
        "bad_fold_signature": bad_fold_signature,
        "feature_drift_forensics": feature_drift_forensics,
        "feature_family_drift_summary": feature_family_drift_summary,
        "score_reversal_context_audit": score_reversal_context_audit,
        "probability_quality_forensics": probability_quality_forensics,
        "probability_quality_summary": probability_quality_summary,
        "score_distribution_shift": score_distribution_shift,
        "score_distribution_shift_summary": score_distribution_shift_summary,
        "fold_reliability_gate": fold_reliability_gate,
        "fold_reliability_gate_summary": fold_reliability_gate_summary,
        "regime_threshold_policy_by_fold": regime_threshold_policy_by_fold,
        "regime_threshold_policy_summary": regime_threshold_policy_summary,
        "regime_stability_forensics": regime_stability_forensics,
        "regime_stability_summary": regime_stability_summary,
        "threshold_forensics": threshold_forensics,
        "threshold_policy_review": threshold_policy_review,
        "threshold_transfer_review": threshold_transfer_review,
        "threshold_transfer_by_fold": threshold_transfer_by_fold,
        "threshold_score_quantile_review": threshold_score_quantile_review,
        "threshold_score_quantile_by_fold": threshold_score_quantile_by_fold,
        "rank_ic_variance_decomposition": rank_ic_variance_decomposition,
        "rank_ic_sampling_uncertainty": rank_ic_sampling_uncertainty,
        "rank_ic_aggregate_evidence": rank_ic_aggregate_evidence,
        "rank_ic_block_sensitivity": rank_ic_block_sensitivity,
        "causal_threshold_policy_summary": causal_threshold_policy_summary,
        "causal_threshold_policy_by_fold": causal_threshold_policy_by_fold,
        "classification_skill_summary": classification_skill_summary,
        "classification_skill_by_fold": classification_skill_by_fold,
        "validation_charter_review": validation_charter_review,
        "validation_charter_proposal": validation_charter_proposal,
        "payoff_alignment": payoff_alignment,
        "payoff_alignment_summary": payoff_alignment_summary,
        "payoff_policy_robustness": payoff_policy_robustness,
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "holdout_boundary_audit": holdout_boundary_audit,
        "frozen_policy_robustness": frozen_policy_robustness,
        "frozen_policy_monitoring_plan": frozen_policy_monitoring_plan,
        "experiment_policy_guard": experiment_policy_guard,
        "future_oos_candidate_plan": future_oos_candidate_plan,
        "holdout_evaluation": holdout_evaluation,
        "holdout_score_bands": holdout_score_bands,
        "holdout_thresholds": holdout_thresholds,
        "holdout_entries": holdout_entries,
        "missing_selected_profiles": missing_selected,
        "decision": decision,
        "zip_paths": zip_paths,
        "write_full_bundles": bool(write_full_bundles),
        "bundle_zip": str(bundle_path) if bundle_path is not None else None,
        "latest_bundle_zip": str(latest_bundle_path) if latest_bundle_path is not None else None,
        "slim_bundle_zip": str(slim_bundle_path),
        "latest_slim_bundle_zip": str(latest_slim_bundle_path),
        "auto_review": auto_review["review"],
        "auto_review_path": str(auto_review_path),
        "auto_review_json_path": str(auto_review_json_path),
        "next_actions_path": str(next_actions_path),
        "phase2_readiness_path": str(phase2_readiness_path),
        "phase2_readiness_md_path": str(phase2_readiness_md_path),
        "phase1_transition_plan_path": str(phase1_transition_plan_path),
        "phase1_transition_plan_md_path": str(phase1_transition_plan_md_path),
    }
