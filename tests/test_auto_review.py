from __future__ import annotations

import json

import pandas as pd

from yenibot.automation import review_experiment_report, write_auto_review


def _write_minimal_report(path, *, missing_selected: bool = False, future_oos_ready: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    control = "control_profile"
    challenger = "candidate_profile"
    pd.DataFrame(
        [
            {
                "profile": control,
                "fold_scope": "full",
                "mean_rank_ic": 0.05,
                "std_rank_ic": 0.07,
                "positive_ic_fraction": 0.80,
                "mean_long_f1": 0.31,
                "test_f1_at_selected_threshold": 0.47,
                "test_f1_at_constrained_threshold": 0.46,
                "calibration_separation": 0.01,
                "top_10_lift_global": 1.10,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
            {
                "profile": challenger,
                "fold_scope": "full",
                "mean_rank_ic": 0.052,
                "std_rank_ic": 0.09,
                "positive_ic_fraction": 0.70,
                "mean_long_f1": 0.30,
                "test_f1_at_selected_threshold": 0.45,
                "test_f1_at_constrained_threshold": 0.44,
                "calibration_separation": 0.01,
                "top_10_lift_global": 1.14,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
        ]
    ).to_csv(path / "profile_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": "blend_prob_mean",
                "fold_scope": "blend_full",
                "mean_rank_ic": 0.055,
                "std_rank_ic": 0.075,
                "positive_ic_fraction": 0.78,
                "top_10_lift_global": 1.15,
            }
        ]
    ).to_csv(path / "profile_blend.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": challenger,
                "fold_scope": "holdout_profile",
                "mean_rank_ic": 0.06,
                "top_10_lift_global": 1.12,
                "top_10_forward_return_global": 0.002,
                "holdout_signal_pass": True,
            }
        ]
    ).to_csv(path / "holdout_evaluation.csv", index=False)
    pd.DataFrame(
        [
            {
                "plan_rank": 1,
                "candidate": challenger,
                "candidate_label": f"{challenger} [top_10]",
                "candidate_type": "profile",
                "stage": "future_oos_score_band_policy",
                "policy_name": "top_10",
                "future_oos_priority_score": 0.7,
                "cv_mean_rank_ic": 0.052,
                "holdout_mean_rank_ic": 0.06,
            }
        ]
    ).to_csv(path / "future_oos_candidate_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "failed_clean_holdout_review",
                "action": "wait_for_new_unseen_bars_keep_control_profile",
                "future_oos_ready": future_oos_ready,
                "future_oos_preferred_ready": False,
                "new_bars_since_anchor": 250,
                "min_new_bars_remaining": 470,
                "preferred_new_bars_remaining": 1900,
                "min_ready_at": "2026-06-12 08:00:00+00:00",
                "preferred_ready_at": "2026-08-11 08:00:00+00:00",
                "holdout_roll_forward_locked": True,
            }
        ]
    ).to_csv(path / "experiment_policy_guard.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected": True,
                "profile": control,
                "fold_scope": "full",
            }
        ]
    ).to_csv(path / "experiment_selection.csv", index=False)
    missing = pd.DataFrame(
        [{"profile": "missing_profile", "fold_scope": "full"}]
        if missing_selected
        else [],
        columns=["profile", "fold_scope"],
    )
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "decision_report.json").write_text(
        json.dumps(
            {
                "run_id": "test_run",
                "control_profile": control,
                "holdout_boundary_passed": True,
                "recommendation": "keep_control_profile",
            }
        ),
        encoding="utf-8",
    )
    (path / "training_execution_summary.json").write_text(
        json.dumps(
            {
                "training_executed_count": 2,
                "training_skipped_count": 0,
                "all_training_scopes_reused": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "blocker": "rank_ic_std_above_phase1_target",
                "root_cause": "fold_score_reversal_or_compression",
                "evidence": "synthetic fixture",
                "decision": "diagnostic_only",
            }
        ]
    ).to_csv(path / "phase1_blocker_root_cause.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "selected_f1_mean": 0.47,
                "constrained_f1_mean": 0.46,
                "official_f1_mean": 0.46,
                "oracle_f1_mean": 0.48,
                "oracle_minus_official_f1": 0.02,
            }
        ]
    ).to_csv(path / "threshold_oracle_gap.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "policy_name": "fixed_top_50",
                "test_f1_mean": 0.46,
                "official_f1_mean": 0.43,
                "f1_delta_vs_official": 0.03,
                "diagnostic_outcome": "score_scale_transfer_candidate",
                "selection_guard": "uses_current_test_score_distribution_no_labels_diagnostic_only",
            }
        ]
    ).to_csv(path / "threshold_score_quantile_review.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "observed_std_rank_ic": 0.07,
                "block_bootstrap_noise_floor_std": 0.04,
                "estimated_between_fold_std": 0.057,
                "diagnostic_conclusion": "material_between_fold_instability_remains_after_noise_adjustment",
            }
        ]
    ).to_csv(path / "rank_ic_variance_decomposition.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "fold": 0,
                "observed_rank_ic": 0.05,
                "bootstrap_rank_ic_std": 0.04,
            }
        ]
    ).to_csv(path / "rank_ic_sampling_uncertainty.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "candidate_type": "profile",
                "fold_scope": "full",
                "observed_mean_rank_ic": 0.05,
                "observed_std_rank_ic": 0.07,
                "positive_fold_fraction": 0.80,
                "min_noise_floor_std": 0.04,
                "max_noise_floor_std": 0.09,
                "target_rank_ic_std": 0.03,
                "target_below_noise_floor_all_blocks": True,
                "random_effects_positive_all_blocks": True,
                "evidence_conclusion": "positive_aggregate_signal_with_unrealistic_absolute_std_target",
            }
        ]
    ).to_csv(path / "rank_ic_aggregate_evidence.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "block_length": 24,
                "rms_bootstrap_noise_std": 0.06,
                "random_effects_ci_low": 0.02,
            }
        ]
    ).to_csv(path / "rank_ic_block_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "policy_name": "causal_fixed_top_60",
                "test_f1_mean": 0.46,
                "f1_delta_vs_official": 0.03,
                "causal_policy_passed_cv": True,
                "diagnostic_outcome": "causal_threshold_transfer_candidate",
            }
        ]
    ).to_csv(path / "causal_threshold_policy_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "policy_name": "causal_fixed_top_60",
                "fold": 0,
                "test_f1": 0.46,
                "selection_guard": "causal_past_scores_only_no_test_labels",
            }
        ]
    ).to_csv(path / "causal_threshold_policy_by_fold.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "candidate_type": "profile",
                "fold_scope": "full",
                "policy_name": "official_threshold",
                "policy_type": "official",
                "f1_mean": 0.46,
                "pred_long_rate_mean": 0.64,
                "always_long_f1_mean": 0.48,
                "rate_matched_random_f1_mean": 0.42,
                "f1_skill_vs_rate_matched_random_mean": 0.04,
                "prauc_lift_vs_prevalence_mean": 1.10,
                "precision_lift_vs_prevalence_mean": 1.04,
                "f1_target_exceeds_always_long_baseline": False,
                "f1_target_exceeds_max_rate_random_baseline": True,
                "skill_evidence_passed": False,
                "classification_conclusion": "standalone_f1_target_below_always_long_no_skill_baseline",
            }
        ]
    ).to_csv(path / "classification_skill_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_scope": "full",
                "policy_name": "official_threshold",
                "fold": 0,
                "f1": 0.46,
                "always_long_f1": 0.48,
            }
        ]
    ).to_csv(path / "classification_skill_by_fold.csv", index=False)
    pd.DataFrame(
        [
            {
                "enabled": True,
                "profile": control,
                "seed": 42,
                "available_fold_count": 8,
                "configured_fold_count": 8,
                "configured_fold_ids": "0,1,2,3,4,5,6,7",
                "valid_configured_fold_ids": "0,1,2,3,4,5,6,7",
                "invalid_configured_fold_ids": "",
                "observed_fold_count": 8,
                "observed_fold_ids": "0,1,2,3,4,5,6,7",
                "missing_valid_fold_ids": "",
                "temporal_span_fraction": 1.0,
                "minimum_temporal_span_fraction": 0.8,
                "first_available_fold_covered": True,
                "last_available_fold_covered": True,
                "coverage_passed": True,
                "status": "passed",
            }
        ]
    ).to_csv(path / "seed_audit_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "criterion": "rank_ic_std",
                "control_profile": control,
                "charter_review_recommended": True,
                "automatic_gate_change_allowed": False,
            },
            {
                "criterion": "long_f1",
                "control_profile": control,
                "charter_review_recommended": True,
                "automatic_gate_change_allowed": False,
            },
        ]
    ).to_csv(path / "validation_charter_review.csv", index=False)
    pd.DataFrame(
        [
            {
                "proposal_version": "v4_draft",
                "proposal_status": "proposed_not_active",
                "active_for_phase1_readiness": False,
                "criterion": "mean_rank_ic",
                "criterion_role": "gate",
                "comparison": ">=",
                "proposed_target": 0.03,
                "observed_value": 0.06,
                "evidence_passed": True,
                "evidence_source": "rank_ic_aggregate_evidence.csv",
                "rationale": "Aggregate signal.",
                "official_gate_unchanged": True,
            }
        ]
    ).to_csv(path / "validation_charter_proposal.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "mechanism": "score_separation_instability",
                "fold_count": 1,
                "recommendation": "diagnose_before_retrain",
            }
        ]
    ).to_csv(path / "bad_fold_mechanism_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": control,
                "fold_id": 0,
                "sample_type": "bad_fold_top_score_false_positive",
                "prob_long": 0.8,
                "label": 0,
            }
        ]
    ).to_csv(path / "prediction_error_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_family": "4h_large_trade_ratio",
                "memory_status": "previously_rejected",
                "recommendation": "do_not_repeat_same_ablation",
            }
        ]
    ).to_csv(path / "historical_experiment_memory_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": "baseline_stable_plus_4h_taker_mean12_ltr_guarded_tanh",
                "suspect_feature": "4h_taker_imbalance_mean_12",
                "context_feature": "4h_large_trade_ratio",
                "mechanism": "stable_tanh_source_guarded_by_stable_large_trade_context",
                "suspect_score": 2.0,
                "requires_02_03": True,
                "requires_04": True,
                "promotion_allowed_now": False,
            }
        ]
    ).to_csv(path / "score_reversal_context_audit.csv", index=False)
    (path / "phase1_decision_ladder.json").write_text(
        json.dumps(
            {
                "run_05_first": True,
                "run_04_required_now": False,
                "full_zip_required_now": False,
                "decision": "diagnostic_only",
            }
        ),
        encoding="utf-8",
    )
    (path / "validation_charter_status.json").write_text(
        json.dumps(
            {
                "active_version": "v3_legacy",
                "official_gate_unchanged": True,
                "automatic_activation_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (path / "frozen_candidate_manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": "control_fold_ensemble_v1",
                "available": True,
                "manifest_hash": "fixture",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "candidate_id": "control_fold_ensemble_v1",
                "candidate_type": "profile",
                "available": True,
            }
        ]
    ).to_csv(path / "frozen_candidate_index.csv", index=False)
    (path / "future_oos_readiness.json").write_text(
        json.dumps(
            {
                "ready_for_evaluation": future_oos_ready,
                "evaluation_completed": False,
                "primary_candidate_passed": False,
                "new_labeled_rows": 250,
                "min_rows_remaining": 470,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        columns=["candidate_id", "rank_ic", "evidence_passed"]
    ).to_csv(path / "future_oos_evaluation.csv", index=False)
    (path / "experiment_registry_snapshot.jsonl").write_text(
        json.dumps({"event_id": "fixture", "run_id": "test_run"}) + "\n",
        encoding="utf-8",
    )


def test_auto_review_waits_for_future_oos_when_no_cv_candidate(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is True
    assert review["next_action"]["action"] == "wait_for_new_unseen_bars_keep_control"
    assert review["next_action"]["do_not_promote_from_current_holdout"] is True
    assert review["cv"]["control"]["profile"] == "control_profile"
    assert review["future_oos"]["best_candidate_plan_row"]["candidate_label"] == "candidate_profile [top_10]"
    assert review["threshold_score_quantile"]["policy_count"] == 1
    assert review["threshold_score_quantile"]["best_test_f1_policy"]["policy_name"] == "fixed_top_50"
    assert review["rank_ic_uncertainty"]["control"]["estimated_between_fold_std"] == 0.057
    assert review["rank_ic_stability_evidence"]["control"]["random_effects_positive_all_blocks"] is True
    assert review["causal_threshold_policy"]["passed_policy_count"] == 1
    assert review["causal_threshold_policy"]["best_test_f1_policy"]["policy_name"] == "causal_fixed_top_60"
    assert review["classification_skill"]["control_official"]["always_long_f1_mean"] == 0.48
    assert review["seed_audit_coverage"]["coverage_passed"] is True
    assert review["validation_charter_review"]["formal_revision_recommended"] is True
    assert review["validation_charter_proposal"]["active_for_phase1_readiness"] is False
    assert review["score_reversal_context"]["hypothesis_count"] == 1
    assert review["phase2_readiness"]["ready_for_phase2"] is False
    assert "rank_ic_std_above_phase1_target" in review["phase2_readiness"]["blockers"]
    assert "long_f1_below_phase1_target" not in review["phase2_readiness"]["blockers"]
    assert "future_unseen_oos_not_ready" in review["phase2_readiness"]["blockers"]
    assert "future_unseen_oos_not_evaluated" not in review["phase2_readiness"]["blockers"]
    assert "future_unseen_oos_candidate_failed" not in review["phase2_readiness"]["blockers"]
    future_checks = {
        row["check"]: row for row in review["phase2_readiness"]["checks"]
    }
    assert future_checks["future_unseen_oos_ready"]["status"] == "failed"
    assert future_checks["future_unseen_oos_evaluated"]["status"] == "pending"
    assert future_checks["future_unseen_oos_passed"]["status"] == "pending"
    assert review["phase2_readiness"]["long_f1_source"] == "validation_selected_threshold"
    assert "fixed_0_50_f1_below_target_calibration_issue" in review["phase2_readiness"]["advisories"]
    assert "rank_ic_std_legacy_gate_requires_governance_review" in review["phase2_readiness"]["advisories"]
    assert "raw_f1_target_below_always_long_no_skill_baseline" in review["phase2_readiness"]["advisories"]
    assert review["phase1_transition_plan"]["decision"] == "PHASE1_RESEARCH_READY_PHASE2_BLOCKED"
    assert "do_not_start_phase2_backtest" in review["phase1_transition_plan"]["blocked_actions"]


def test_auto_review_uses_guarded_f1_when_selected_threshold_is_too_broad(tmp_path) -> None:
    _write_minimal_report(tmp_path)
    comparison = pd.read_csv(tmp_path / "profile_comparison.csv")
    comparison.loc[comparison["profile"] == "control_profile", "test_pred_long_rate_at_selected_threshold"] = 0.86
    comparison.loc[comparison["profile"] == "control_profile", "test_pred_long_rate_at_constrained_threshold"] = 0.64
    comparison.loc[comparison["profile"] == "control_profile", "test_f1_at_constrained_threshold"] = 0.43
    comparison.to_csv(tmp_path / "profile_comparison.csv", index=False)

    review = review_experiment_report(tmp_path)

    assert review["phase2_readiness"]["long_f1_source"] == "validation_constrained_threshold"
    assert "long_f1_below_phase1_target" in review["phase2_readiness"]["blockers"]


def test_auto_review_uses_official_calibrated_threshold_when_available(tmp_path) -> None:
    _write_minimal_report(tmp_path)
    comparison = pd.read_csv(tmp_path / "profile_comparison.csv")
    mask = comparison["profile"] == "control_profile"
    comparison.loc[mask, "test_f1_at_guarded_threshold"] = 0.43
    comparison.loc[mask, "test_pred_long_rate_at_guarded_threshold"] = 0.64
    comparison.loc[mask, "guarded_threshold_source"] = "validation_constrained_threshold"
    comparison.loc[mask, "test_f1_at_official_threshold"] = 0.455
    comparison.loc[mask, "test_pred_long_rate_at_official_threshold"] = 0.63
    comparison.loc[mask, "official_threshold_source"] = "calibrated_validation_constrained_threshold"
    comparison.loc[mask, "official_threshold_uses_calibration"] = True
    comparison.to_csv(tmp_path / "profile_comparison.csv", index=False)

    review = review_experiment_report(tmp_path)

    assert review["phase2_readiness"]["long_f1_source"] == "calibrated_validation_constrained_threshold"
    assert "long_f1_below_phase1_target" not in review["phase2_readiness"]["blockers"]


def test_auto_review_flags_missing_selected_profiles(tmp_path) -> None:
    _write_minimal_report(tmp_path, missing_selected=True)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is False
    assert review["next_action"]["action"] == "fix_missing_selected_profiles"


def test_write_auto_review_outputs_files(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    result = write_auto_review(tmp_path)

    assert (tmp_path / "auto_review.md").exists()
    assert (tmp_path / "auto_review.json").exists()
    assert (tmp_path / "next_actions.json").exists()
    assert (tmp_path / "phase2_readiness.json").exists()
    assert (tmp_path / "phase2_readiness.md").exists()
    assert (tmp_path / "phase1_transition_plan.json").exists()
    assert (tmp_path / "phase1_transition_plan.md").exists()
    next_actions = json.loads((tmp_path / "next_actions.json").read_text(encoding="utf-8"))
    assert next_actions["action"] == "wait_for_new_unseen_bars_keep_control"
    phase2 = json.loads((tmp_path / "phase2_readiness.json").read_text(encoding="utf-8"))
    assert phase2["decision"] == "DO_NOT_PROCEED_TO_PHASE2"
    assert phase2["long_f1_source"] == "validation_selected_threshold"
    transition = json.loads((tmp_path / "phase1_transition_plan.json").read_text(encoding="utf-8"))
    assert transition["decision"] == "PHASE1_RESEARCH_READY_PHASE2_BLOCKED"
    assert transition["long_f1_source"] == "validation_selected_threshold"
    assert result["auto_review_path"].endswith("auto_review.md")
