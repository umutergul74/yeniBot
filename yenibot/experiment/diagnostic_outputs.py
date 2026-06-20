"""Small helpers for diagnostics output staging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.dashboard import write_model_performance_dashboard
from yenibot.experiment.drift import _score_reversal_context_audit_frame, _write_score_reversal_context_audit
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


def profile_dirs(run_dir: Path) -> list[Path]:
    """Return completed profile/scope directories for a run."""

    paths = []
    for profile_dir in run_dir.iterdir():
        if not profile_dir.is_dir():
            continue
        for scope_dir in profile_dir.iterdir():
            if (scope_dir / "training_manifest.json").exists() and (scope_dir / "predictions_all.parquet").exists():
                paths.append(scope_dir)
    return sorted(paths)


def prewrite_auto_review_inputs(
    report_dir: Path,
    *,
    entries: list[dict[str, Any]],
    comparison: pd.DataFrame,
    profile_blend: pd.DataFrame,
    performance_gap_analysis: pd.DataFrame,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
    payoff_policy_robustness_summary: pd.DataFrame,
    future_oos_candidate_plan: pd.DataFrame,
    diagnostic_config: dict[str, Any],
    settings: dict[str, Any],
    feature_family_drift_summary: pd.DataFrame,
    feature_drift_forensics: pd.DataFrame,
    bad_fold_signature: pd.DataFrame,
    score_distribution_shift_summary: pd.DataFrame,
    probability_quality_summary: pd.DataFrame,
    score_separation_forensics: pd.DataFrame,
    recency_policy_decision: dict[str, Any],
    replacement_candidate_fit: dict[str, Any],
    rank_ic_aggregate_evidence: pd.DataFrame,
    classification_skill_summary: pd.DataFrame,
    model_evidence_uncertainty: pd.DataFrame,
    probability_calibration_comparison: pd.DataFrame,
    payoff_alignment: pd.DataFrame,
    seed_stability: pd.DataFrame,
    future_oos_readiness: dict[str, Any],
) -> pd.DataFrame:
    """Write provisional files required by the auto-review completeness audit."""

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
        config=diagnostic_config,
        settings=settings,
    )
    threshold_oracle_gap = _threshold_oracle_gap_frame(threshold_forensics, diagnostic_config)
    historical_memory_audit = _historical_experiment_memory_audit_frame(
        feature_family_drift_summary,
        diagnostic_config,
    )
    score_reversal_context_audit = _score_reversal_context_audit_frame(
        feature_drift_forensics,
        historical_memory_audit,
        diagnostic_config,
    )
    _write_score_reversal_context_audit(report_dir, score_reversal_context_audit)
    bad_fold_mechanism_summary = _bad_fold_mechanism_summary_frame(
        bad_fold_signature=bad_fold_signature,
        feature_family_drift_summary=feature_family_drift_summary,
        score_distribution_shift_summary=score_distribution_shift_summary,
        probability_quality_summary=probability_quality_summary,
        historical_memory_audit=historical_memory_audit,
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
        historical_experiment_memory_audit=historical_memory_audit,
        phase2_readiness={},
        settings=settings,
        config=diagnostic_config,
    )
    phase1_decision_ladder = _phase1_decision_ladder_payload(
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        phase2_readiness={},
        settings=settings,
        recency_policy_decision=recency_policy_decision,
        replacement_candidate_fit=replacement_candidate_fit,
    )
    _write_phase1_blocker_action_plan(report_dir, phase1_blocker_action_plan)
    _write_root_cause_reports(
        report_dir,
        phase1_blocker_root_cause=phase1_blocker_root_cause,
        threshold_oracle_gap=threshold_oracle_gap,
        bad_fold_mechanism_summary=bad_fold_mechanism_summary,
        prediction_error_audit=prediction_error_audit,
        historical_experiment_memory_audit=historical_memory_audit,
        decision_ladder=phase1_decision_ladder,
    )
    write_model_performance_dashboard(
        report_dir,
        entries=entries,
        comparison=comparison,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        rank_ic_aggregate_evidence=rank_ic_aggregate_evidence,
        classification_skill_summary=classification_skill_summary,
        probability_quality_summary=probability_quality_summary,
        model_evidence_uncertainty=model_evidence_uncertainty,
        probability_calibration_comparison=probability_calibration_comparison,
        payoff_alignment=payoff_alignment,
        seed_stability=seed_stability,
        phase2_readiness={},
        future_oos_readiness=future_oos_readiness,
        control_profile=settings["control_profile"],
    )
    return score_reversal_context_audit
