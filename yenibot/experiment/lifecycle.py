"""Shared Future-OOS lifecycle states and operator actions.

The configuration file is an audit record of what was preregistered.  Generated
evaluation artifacts are the source of truth once a frozen candidate has been
scored.  Keeping the outcome predicates and action names here prevents reports
from independently falling back to stale configuration text.
"""

from __future__ import annotations

from typing import Any, Iterable

PIN_REPLACEMENT_MANIFEST_ACTION = (
    "pin_replacement_candidate_manifest_and_activate_new_oos_anchor"
)
RETIRE_FAILED_FUTURE_OOS_ACTION = (
    "retire_failed_frozen_candidate_and_open_new_research_anchor"
)
REVIEW_PASSED_FUTURE_OOS_ACTION = (
    "future_oos_candidate_passed_review_phase2_readiness"
)
RUN_FUTURE_OOS_EVALUATOR_ACTION = "run_no_refit_future_oos_evaluator"
WAIT_FOR_FUTURE_OOS_ACTION = "wait_for_new_future_oos_rows"


def future_oos_failed(
    readiness: dict[str, Any] | None,
    blockers: Iterable[str] | None = None,
) -> bool:
    """Return true only for a completed failed evaluation or its explicit blocker."""

    state = readiness or {}
    blocker_set = {str(item) for item in blockers or []}
    return bool(
        "future_unseen_oos_candidate_failed" in blocker_set
        or (
            bool(state.get("evaluation_completed", False))
            and state.get("primary_candidate_passed") is False
        )
        or str(state.get("evaluation_state", "")) == "evaluated_failed"
    )


def future_oos_passed(readiness: dict[str, Any] | None) -> bool:
    """Return true only when the primary frozen candidate completed and passed."""

    state = readiness or {}
    return bool(
        bool(state.get("evaluation_completed", False))
        and state.get("primary_candidate_passed") is True
    )
