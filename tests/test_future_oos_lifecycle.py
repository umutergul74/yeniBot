from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from yenibot.config import load_config
from yenibot.experiment.configuration import _experiment_policy_guard
from yenibot.experiment.current_status import build_phase1_current_status
from yenibot.experiment.lifecycle import (
    PIN_REPLACEMENT_MANIFEST_ACTION,
    RETIRE_FAILED_FUTURE_OOS_ACTION,
)
from yenibot.experiment.lifecycle_reporting import (
    reconcile_future_oos_lifecycle_reports,
)
from yenibot.experiment.oos_preflight import reconcile_preflight_with_evaluation
from yenibot.experiment.rolling_research import (
    reconcile_recency_manifest_lifecycle,
    research_protocol_payload,
)
from yenibot.experiment.root_cause import _phase1_decision_ladder_payload


def _seed_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison_role": "same_seed_reproduction",
                "reproducibility_status": "same_seed_reproduced",
            }
        ]
    )


def _failed_readiness() -> dict:
    return {
        "ready_for_evaluation": True,
        "evaluation_completed": True,
        "evaluation_state": "evaluated_failed",
        "primary_candidate_id": "control_recent3_equal_v2",
        "primary_candidate_passed": False,
    }


def test_failed_evaluation_overrides_stale_waiting_protocol_config() -> None:
    protocol = research_protocol_payload(
        {
            "experiments": {
                "next_research_cycle": {
                    "status": "replacement_candidate_manifest_pinned_awaiting_future_oos",
                    "next_action": "wait_for_new_future_oos_rows",
                    "new_future_oos_anchor_required": False,
                    "replacement_candidate": {
                        "candidate_id": "control_recent3_equal_v2",
                        "enabled": False,
                        "status": "manifest_pinned_awaiting_future_oos",
                    },
                }
            }
        },
        phase2_readiness={
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_candidate_failed"],
            "next_action": RETIRE_FAILED_FUTURE_OOS_ACTION,
        },
        future_oos_preflight={"state": "ready_prediction_only"},
        future_oos_readiness=_failed_readiness(),
        frozen_candidate_index=pd.DataFrame(
            [{"candidate_id": "control_recent3_equal_v2", "available": True}]
        ),
        seed_reproducibility_audit=_seed_audit(),
        replacement_candidate_fit={"status": "not_run_no_preregistered_replacement"},
    )

    assert protocol["status"] == "failed_future_oos_new_research_cycle_required"
    assert protocol["next_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION
    assert protocol["failed_candidate_status"] == "retired_after_failed_future_oos"
    assert protocol["replacement_candidate_id"] is None
    assert protocol["replacement_candidate"]["status"] == "retired_after_failed_future_oos"
    assert protocol["new_future_oos_anchor_required"] is True
    assert protocol["run_04_required_now"] is False
    assert protocol["run_05_required_now"] is False


def test_failed_evaluation_disables_stale_notebook_routing() -> None:
    status = build_phase1_current_status(
        run_id="run",
        control_profile="control",
        phase2_readiness={
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_candidate_failed"],
            "next_action": RETIRE_FAILED_FUTURE_OOS_ACTION,
        },
        model_performance_summary={
            "historical_walk_forward_evidence_passed": True,
            "model_evidence_passed": False,
            "frozen_future_oos_evidence_passed": False,
        },
        phase1_decision_ladder={
            "recommended_next_action": "run_05_only_and_review_root_cause_reports",
            "run_04_required_now": True,
            "run_05_first": True,
            "next_notebook": "05",
        },
        next_research_protocol={"next_action": "wait_for_new_future_oos_rows"},
        future_oos_preflight={"state": "ready_prediction_only"},
        future_oos_readiness=_failed_readiness(),
        seed_reproducibility_audit=_seed_audit(),
        training_execution={"training_executed_count": 0},
    )

    assert status["current_status"] == "failed_future_oos_new_research_cycle_required"
    assert status["next_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION
    assert status["run_04_required_now"] is False
    assert status["run_05_first"] is False
    assert status["next_notebook"] == "none_until_new_research_cycle_is_preregistered"


def test_preregistered_adaptive_cycle_routes_only_to_notebook_04a() -> None:
    config = load_config("config.yaml")
    protocol = research_protocol_payload(
        config,
        phase2_readiness={
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_candidate_failed"],
            "next_action": RETIRE_FAILED_FUTURE_OOS_ACTION,
        },
        future_oos_preflight={"state": "ready_prediction_only"},
        future_oos_readiness=_failed_readiness(),
        frozen_candidate_index=pd.DataFrame(
            [{"candidate_id": "control_recent3_equal_v2", "available": True}]
        ),
        seed_reproducibility_audit=_seed_audit(),
        replacement_candidate_fit={"status": "not_run_no_preregistered_replacement"},
    )

    assert protocol["status"] == (
        "historical_validation_adaptive_ensemble_preregistered"
    )
    assert protocol["next_action"] == (
        "run_notebook_04a_historical_policy_research_only"
    )
    assert protocol["run_04a_required_now"] is True
    assert protocol["run_05_required_now"] is False
    assert protocol["next_notebook"] == "04a"
    assert protocol["new_research_cycle_required"] is False

    status = build_phase1_current_status(
        run_id="run",
        control_profile="control",
        phase2_readiness={
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_candidate_failed"],
            "next_action": RETIRE_FAILED_FUTURE_OOS_ACTION,
        },
        model_performance_summary={},
        phase1_decision_ladder={},
        next_research_protocol=protocol,
        future_oos_preflight={"state": "ready_prediction_only"},
        future_oos_readiness=_failed_readiness(),
        seed_reproducibility_audit=_seed_audit(),
        training_execution={"training_executed_count": 0},
    )
    assert status["current_status"] == (
        "failed_future_oos_historical_research_preregistered"
    )
    assert status["next_notebook"] == "04a"
    assert status["run_04_required_now"] is False
    assert status["run_05_first"] is False


def test_failed_evaluation_keeps_research_hint_separate_from_lifecycle_action() -> None:
    ladder = _phase1_decision_ladder_payload(
        phase1_blocker_root_cause=pd.DataFrame(),
        threshold_oracle_gap=pd.DataFrame(),
        bad_fold_mechanism_summary=pd.DataFrame(),
        phase2_readiness={
            "ready_for_phase2": False,
            "blockers": ["future_unseen_oos_candidate_failed"],
        },
        settings={"control_profile": "control"},
        recency_policy_decision={
            "candidate_ready_for_preregistration": True,
            "recommended_policy": "recent_3_equal",
        },
    )

    assert ladder["recommended_next_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION
    assert ladder["research_cycle_followup_action"] == (
        "explicitly_review_and_preregister_historical_recency_winner"
    )
    assert ladder["run_04_required_now"] is False
    assert ladder["run_05_first"] is False
    assert ladder["next_notebook"] == "none_until_new_research_cycle_is_preregistered"


def test_completed_evaluation_supersedes_but_preserves_preflight_action() -> None:
    reconciled = reconcile_preflight_with_evaluation(
        {
            "state": "ready_prediction_only",
            "next_action": "run_notebook_05_prediction_only",
        },
        _failed_readiness(),
    )

    assert reconciled["next_action"] == "run_notebook_05_prediction_only"
    assert reconciled["next_action_scope"] == "pre_evaluation_preflight_only"
    assert reconciled["lifecycle_superseded"] is True
    assert reconciled["current_lifecycle_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION


def test_configured_failed_outcome_overrides_monitor_wait_action() -> None:
    config = load_config("config.yaml")
    guard = _experiment_policy_guard(
        {
            "control_profile": config["experiments"]["control_profile"],
            "candidate_profiles": [],
            "full_cv_profiles": [],
            "seed_audit": {},
            "holdout": {},
        },
        config,
    )

    assert guard["primary_candidate_outcome"] == "failed_future_oos_retired"
    assert guard["action"] == RETIRE_FAILED_FUTURE_OOS_ACTION
    assert guard["next_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION
    assert guard["monitor_next_action"] != guard["next_action"]


def test_completed_evaluation_scopes_copied_recency_manifest_action(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recency_ensemble_manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "next_action": "wait_for_new_future_oos_rows",
            }
        ),
        encoding="utf-8",
    )

    payload = reconcile_recency_manifest_lifecycle(
        tmp_path,
        future_oos_readiness=_failed_readiness(),
        current_lifecycle_action=RETIRE_FAILED_FUTURE_OOS_ACTION,
    )

    assert payload["artifact_role"] == "historical_recency_research_snapshot"
    assert payload["next_action"] == "wait_for_new_future_oos_rows"
    assert payload["next_action_scope"] == "historical_research_generation_time_only"
    assert payload["lifecycle_superseded"] is True
    assert payload["current_lifecycle_action"] == RETIRE_FAILED_FUTURE_OOS_ACTION


def test_lifecycle_report_reconciliation_routes_completed_replacement_to_pin(
    tmp_path: Path,
) -> None:
    result = reconcile_future_oos_lifecycle_reports(
        tmp_path,
        preflight={
            "state": "awaiting_replacement_preregistration",
            "next_action": "select_and_preregister_replacement_before_new_oos_anchor",
            "warnings": [],
        },
        readiness={"evaluation_completed": False},
        replacement_candidate_fit={
            "status": "fit_complete_manifest_pin_required",
        },
    )

    assert result["next_action"] == PIN_REPLACEMENT_MANIFEST_ACTION
    assert result["current_lifecycle_action"] == PIN_REPLACEMENT_MANIFEST_ACTION
    assert (tmp_path / "future_oos_preflight.json").exists()
    assert (tmp_path / "future_oos_preflight.md").exists()
