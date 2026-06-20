"""Single-source current Phase 1 status report for diagnostics bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _json_ready, _write_json

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
) -> dict[str, Any]:
    """Return the clearest current state, preferring generated artifacts over config text."""

    blockers = [str(item) for item in phase2_readiness.get("blockers", []) or []]
    seed_status = _same_seed_status(seed_reproducibility_audit)
    seed_passed = _same_seed_passed(seed_reproducibility_audit)
    frozen_missing = "frozen_candidate_manifest_unavailable" in blockers
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

    if phase2_ready:
        status = "phase2_ready_review_required"
        next_action = "review_phase2_boundary_before_writing_phase2_code"
    elif not seed_passed:
        status = "seed_reproducibility_review_required"
        next_action = "complete_seed_reproducibility_review_before_replacement_preregistration"
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
    else:
        status = "phase1_blocked_review_required"
        next_action = str(
            phase1_decision_ladder.get("recommended_next_action")
            or next_research_protocol.get("next_action")
            or phase2_readiness.get("next_action")
            or "review_phase1_reports"
        )

    return {
        "run_id": run_id,
        "control_profile": control_profile,
        "current_status": status,
        "phase2_ready": phase2_ready,
        "phase2_blockers": blockers,
        "active_blocker": blockers[0] if blockers else "",
        "next_action": next_action,
        "next_action_source": "phase1_current_status_artifact_state",
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
        "replacement_preregistration_required": frozen_missing,
        "run_04_required_now": bool(phase1_decision_ladder.get("run_04_required_now", False)),
        "run_05_first": bool(phase1_decision_ladder.get("run_05_first", True)),
        "promotion_allowed_now": phase2_ready,
        "phase2_code_allowed": phase2_ready,
        "training_executed_count": training_execution.get("training_executed_count"),
        "all_training_scopes_reused": training_execution.get("all_training_scopes_reused"),
        "guardrails": [
            "do_not_promote_from_seen_holdout",
            "do_not_write_phase2_code_until_phase2_ready",
            "do_not_repeat_rejected_direct_ablation_without_new_mechanism",
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
        f"- Next action: `{status.get('next_action')}`",
        "",
        "## Evidence",
        "",
        f"- Historical walk-forward evidence passed: `{status.get('historical_walk_forward_evidence_passed')}`",
        f"- Model evidence passed: `{status.get('model_evidence_passed')}`",
        f"- Frozen future-OOS evidence passed: `{status.get('frozen_future_oos_evidence_passed')}`",
        f"- Seed reproducibility: `{status.get('seed_reproducibility_status')}`",
        f"- Future-OOS preflight state: `{status.get('future_oos_preflight_state')}`",
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
