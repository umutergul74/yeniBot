"""Reconcile stage-scoped reports with the current Future-OOS lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yenibot.experiment.common import _write_json
from yenibot.experiment.lifecycle import (
    PIN_REPLACEMENT_MANIFEST_ACTION,
    RETIRE_FAILED_FUTURE_OOS_ACTION,
    REVIEW_PASSED_FUTURE_OOS_ACTION,
    future_oos_failed,
    future_oos_passed,
)
from yenibot.experiment.oos_preflight import (
    future_oos_preflight_markdown,
    reconcile_preflight_with_evaluation,
)
from yenibot.experiment.rolling_research import reconcile_recency_manifest_lifecycle


def reconcile_future_oos_lifecycle_reports(
    report_dir: str | Path,
    *,
    preflight: dict[str, Any],
    readiness: dict[str, Any],
    replacement_candidate_fit: dict[str, Any] | None = None,
    current_lifecycle_action: str | None = None,
) -> dict[str, Any]:
    """Write preflight/recency lifecycle metadata from one authoritative action."""

    replacement = replacement_candidate_fit or {}
    if current_lifecycle_action is None:
        if future_oos_failed(readiness):
            current_lifecycle_action = (
                PIN_REPLACEMENT_MANIFEST_ACTION
                if replacement.get("status") == "fit_complete_manifest_pin_required"
                else RETIRE_FAILED_FUTURE_OOS_ACTION
            )
        elif future_oos_passed(readiness):
            current_lifecycle_action = REVIEW_PASSED_FUTURE_OOS_ACTION
    result = reconcile_preflight_with_evaluation(
        preflight,
        readiness,
        current_lifecycle_action=current_lifecycle_action,
    )
    if (
        replacement.get("status") == "fit_complete_manifest_pin_required"
        and result.get("state") == "awaiting_replacement_preregistration"
    ):
        result["current_lifecycle_action"] = PIN_REPLACEMENT_MANIFEST_ACTION
        if not bool(result.get("lifecycle_superseded", False)):
            result["next_action"] = PIN_REPLACEMENT_MANIFEST_ACTION
        warning = (
            "Replacement candidate fit is complete; pin its manifest hash and "
            "activate a new future-OOS anchor before scoring."
        )
        warnings = list(result.get("warnings", []) or [])
        if warning not in warnings:
            warnings.append(warning)
        result["warnings"] = warnings
    path = Path(report_dir)
    _write_json(path / "future_oos_preflight.json", result)
    (path / "future_oos_preflight.md").write_text(
        future_oos_preflight_markdown(result),
        encoding="utf-8",
    )
    reconcile_recency_manifest_lifecycle(
        path,
        future_oos_readiness=readiness,
        current_lifecycle_action=result.get("current_lifecycle_action"),
    )
    return result
