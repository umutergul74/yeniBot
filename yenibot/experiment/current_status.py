"""Single-source current Phase 1 status report for diagnostics bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _json_ready, _write_json
from yenibot.experiment.lifecycle import (
    PIN_REPLACEMENT_MANIFEST_ACTION,
    RETIRE_FAILED_FUTURE_OOS_ACTION,
    future_oos_failed,
)

__all__ = [
    "build_phase1_current_status",
    "write_phase1_current_status",
]


def _same_seed_status(seed_reproducibility_audit: pd.DataFrame) -> str:
    if seed_reproducibility_audit.empty:
        return "missing"
    if "comparison_role" not in seed_reproducibility_audit.columns:
        return "missing_comparison_role"
    same_seed = seed_reproducibility_audit.loc[
        seed_reproducibility_audit["comparison_role"].astype(str).eq(
            "same_seed_reproduction"
        )
    ]
    if same_seed.empty:
        return "missing_same_seed_reproduction"
    if "reproducibility_status" not in same_seed.columns:
        return "missing_reproducibility_status"
    return str(same_seed.iloc[0]["reproducibility_status"])


def _same_seed_passed(seed_reproducibility_audit: pd.DataFrame) -> bool:
    return _same_seed_status(seed_reproducibility_audit) in {
        "same_seed_reproduced",
        "same_seed_ranking_reproduced_with_numeric_drift",
    }


def build_phase1_current_status(
    *,
    run_id: str,
    control_profile: str,
    phase2_readiness: dict[str, Any],
    model_performance_summary: dict[str, Any],
    phase1_decision_ladder: dict[str, Any],
    next_research_protocol: dict[str, Any],
    future_oos_preflight: dict[str, Any],
    future_oos_readiness: dict[str, Any],
    seed_reproducibility_audit: pd.DataFrame,
    training_execution: dict[str, Any],
    research_focus: dict[str, Any] | None = None,
    configured_candidate_profiles: list[str] | None = None,
    run_best_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the clearest current state, preferring generated artifacts over config text."""

    blockers = [str(item) for item in phase2_readiness.get("blockers", []) or []]
    seed_status = _same_seed_status(seed_reproducibility_audit)
    seed_passed = _same_seed_passed(seed_reproducibility_audit)
    frozen_missing = "frozen_candidate_manifest_unavailable" in blockers
    frozen_failed = future_oos_failed(future_oos_readiness, blockers)
    phase2_ready = bool(phase2_readiness.get("ready_for_phase2", False))
    model_evidence_passed = bool(
        model_performance_summary.get("model_evidence_passed", False)
    )
    historical_evidence_passed = bool(
        model_performance_summary.get("historical_walk_forward_evidence_passed", False)
    )
    frozen_future_oos_passed = bool(
        model_performance_summary.get("frozen_future_oos_evidence_passed", False)
    )
    ladder_next_action = str(
        phase1_decision_ladder.get("recommended_next_action") or ""
    )
    protocol_hypothesis = str(
        next_research_protocol.get("primary_hypothesis") or ""
    ).strip()
    protocol_action = str(next_research_protocol.get("next_action") or "").strip()
    research_preregistered = bool(
        protocol_hypothesis
        and protocol_hypothesis != "not_yet_preregistered"
        and protocol_action.startswith("run_notebook_04a_")
    )

    if phase2_ready:
        status = "phase2_ready_review_required"
        next_action = "review_phase2_boundary_before_writing_phase2_code"
    elif not seed_passed:
        status = "seed_reproducibility_review_required"
        next_action = "complete_seed_reproducibility_review_before_replacement_preregistration"
    elif frozen_failed:
        if research_preregistered:
            next_action = protocol_action
            status = "failed_future_oos_historical_research_preregistered"
        else:
            next_action = str(
                phase2_readiness.get("next_action")
                or next_research_protocol.get("next_action")
                or RETIRE_FAILED_FUTURE_OOS_ACTION
            )
            status = (
                "failed_future_oos_replacement_manifest_pin_required"
                if next_action == PIN_REPLACEMENT_MANIFEST_ACTION
                else "failed_future_oos_new_research_cycle_required"
            )
    elif frozen_missing:
        if ladder_next_action.startswith("pin_replacement_candidate_manifest"):
            status = "historical_model_evidence_passed_awaiting_replacement_manifest_pin"
            next_action = ladder_next_action
        else:
            status = "historical_model_evidence_passed_awaiting_replacement_preregistration"
            next_action = "select_and_preregister_replacement_candidate_from_historical_cv_only"
    elif bool(future_oos_readiness.get("ready_for_evaluation", False)) and not bool(
        future_oos_readiness.get("evaluation_completed", False)
    ):
        status = "future_oos_ready_prediction_only"
        next_action = "run_no_refit_future_oos_evaluator"
    elif (
        "future_unseen_oos_not_ready" in blockers
        and future_oos_preflight.get("state") == "waiting_for_mature_labeled_rows"
    ):
        status = "replacement_manifest_pinned_waiting_for_future_oos_rows"
        next_action = "wait_for_new_future_oos_rows"
    else:
        status = "phase1_blocked_review_required"
        next_action = str(
            phase1_decision_ladder.get("recommended_next_action")
            or next_research_protocol.get("next_action")
            or phase2_readiness.get("next_action")
            or "review_phase1_reports"
        )

    research_focus = research_focus or {}
    configured_candidate_profiles = [
        str(profile) for profile in configured_candidate_profiles or []
    ]
    run_best_candidate = run_best_candidate or {}
    if configured_candidate_profiles and run_best_candidate:
        historical_research_result = "candidate_passed_historical_promotion_gates"
        historical_research_next_action = (
            "review_historical_candidate_without_using_seen_holdout"
        )
    elif configured_candidate_profiles:
        historical_research_result = "candidates_evaluated_no_promotion"
        historical_research_next_action = (
            "review_candidate_audits_and_update_experiment_memory"
        )
    else:
        historical_research_result = "no_active_candidate"
        historical_research_next_action = "preregister_distinct_historical_mechanism"
    if research_preregistered:
        historical_research_result = "preregistered_policy_pending_historical_evaluation"
        historical_research_next_action = protocol_action

    if frozen_failed and research_preregistered:
        run_04_required_now = False
        run_05_first = False
        next_notebook = "04a"
    elif frozen_failed:
        run_04_required_now = False
        run_05_first = False
        next_notebook = (
            "none_until_replacement_manifest_is_pinned"
            if next_action == PIN_REPLACEMENT_MANIFEST_ACTION
            else "none_until_new_research_cycle_is_preregistered"
        )
    else:
        run_04_required_now = bool(
            phase1_decision_ladder.get("run_04_required_now", False)
        )
        run_05_first = bool(phase1_decision_ladder.get("run_05_first", True))
        next_notebook = str(
            phase1_decision_ladder.get("next_notebook")
            or ("04" if run_04_required_now else "05")
        )

    return {
        "run_id": run_id,
        "control_profile": control_profile,
        "current_status": status,
        "phase2_ready": phase2_ready,
        "phase2_blockers": blockers,
        "active_blocker": blockers[0] if blockers else "",
        "next_action": next_action,
        "phase2_track_next_action": (
            RETIRE_FAILED_FUTURE_OOS_ACTION if frozen_failed else next_action
        ),
        "research_track_next_action": next_action,
        "next_action_source": "phase1_current_status_artifact_state",
        "historical_research_mode": str(research_focus.get("mode") or ""),
        "historical_research_status": str(research_focus.get("status") or ""),
        "historical_research_result": historical_research_result,
        "historical_research_next_action": historical_research_next_action,
        "historical_candidate_profiles": configured_candidate_profiles,
        "historical_best_candidate": run_best_candidate,
        "historical_walk_forward_evidence_passed": historical_evidence_passed,
        "model_evidence_passed": model_evidence_passed,
        "frozen_future_oos_evidence_passed": frozen_future_oos_passed,
        "seed_reproducibility_status": seed_status,
        "seed_reproducibility_passed": seed_passed,
        "future_oos_preflight_state": future_oos_preflight.get("state"),
        "future_oos_ready_for_evaluation": bool(
            future_oos_readiness.get("ready_for_evaluation", False)
        ),
        "future_oos_evaluation_completed": bool(
            future_oos_readiness.get("evaluation_completed", False)
        ),
        "future_oos_primary_candidate_passed": future_oos_readiness.get(
            "primary_candidate_passed"
        ),
        "future_oos_failed": frozen_failed,
        "new_research_cycle_required": bool(
            frozen_failed
            and not research_preregistered
            and next_action != PIN_REPLACEMENT_MANIFEST_ACTION
        ),
        "new_future_oos_anchor_required": frozen_failed,
        "replacement_preregistration_required": frozen_missing,
        "run_04_required_now": run_04_required_now,
        "run_05_first": run_05_first,
        "next_notebook": next_notebook,
        "promotion_allowed_now": phase2_ready,
        "phase2_code_allowed": phase2_ready,
        "training_executed_count": training_execution.get("training_executed_count"),
        "all_training_scopes_reused": training_execution.get("all_training_scopes_reused"),
        "guardrails": [
            "do_not_promote_from_seen_holdout",
            "do_not_write_phase2_code_until_phase2_ready",
            "do_not_repeat_rejected_direct_ablation_without_new_mechanism",
            "do_not_run_notebook_05_before_adaptive_policy_review",
        ],
    }


def _current_status_markdown(status: dict[str, Any]) -> str:
    blockers = status.get("phase2_blockers") or []
    lines = [
        "# Phase 1 Current Status",
        "",
        f"- Run id: `{status.get('run_id')}`",
        f"- Current status: `{status.get('current_status')}`",
        f"- Phase 2 ready: `{status.get('phase2_ready')}`",
        f"- Active blocker: `{status.get('active_blocker') or 'none'}`",
        f"- Blockers: `{', '.join(blockers) if blockers else 'none'}`",
        f"- Phase 2 track next action: `{status.get('phase2_track_next_action')}`",
        f"- Historical research status: `{status.get('historical_research_status')}`",
        f"- Historical research result: `{status.get('historical_research_result')}`",
        f"- Historical research next action: `{status.get('historical_research_next_action')}`",
        "",
        "## Evidence",
        "",
        f"- Historical walk-forward evidence passed: `{status.get('historical_walk_forward_evidence_passed')}`",
        f"- Model evidence passed: `{status.get('model_evidence_passed')}`",
        f"- Frozen future-OOS evidence passed: `{status.get('frozen_future_oos_evidence_passed')}`",
        f"- Seed reproducibility: `{status.get('seed_reproducibility_status')}`",
        f"- Future-OOS preflight state: `{status.get('future_oos_preflight_state')}`",
        f"- Future-OOS evaluation completed: `{status.get('future_oos_evaluation_completed')}`",
        f"- Future-OOS candidate passed: `{status.get('future_oos_primary_candidate_passed')}`",
        f"- Next notebook: `{status.get('next_notebook')}`",
        "",
        "## Guardrails",
        "",
    ]
    for item in status.get("guardrails") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def write_phase1_current_status(path: Path, status: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "phase1_current_status.json", _json_ready(status))
    pd.DataFrame([status]).to_csv(path / "phase1_current_status.csv", index=False)
    (path / "phase1_current_status.md").write_text(
        _current_status_markdown(status),
        encoding="utf-8",
    )
