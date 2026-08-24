"""Cross-report consistency checks for Phase 1 diagnostics bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _json_ready, _table_markdown, _write_json
from yenibot.experiment.lifecycle import (
    PIN_REPLACEMENT_MANIFEST_ACTION,
    RETIRE_FAILED_FUTURE_OOS_ACTION,
    future_oos_failed,
)

__all__ = [
    "build_report_consistency_audit",
    "write_report_consistency_audit",
]


PIN_MANIFEST_ACTION = PIN_REPLACEMENT_MANIFEST_ACTION


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _action_from_auto_review(auto_review: dict[str, Any]) -> str:
    action = auto_review.get("next_action")
    if isinstance(action, dict):
        return str(action.get("action") or "")
    return str(action or "")


def _canonical_action(action: str) -> str:
    value = str(action or "")
    if not value:
        return ""
    if value.startswith("pin_replacement_candidate_manifest"):
        return PIN_MANIFEST_ACTION
    if value.startswith("select_and_preregister_replacement_candidate"):
        return "select_and_preregister_replacement_candidate_from_historical_cv_only"
    if value in {
        "refresh_01_02_03_then_recheck_without_running_04",
        "refresh_data_and_run_05_when_future_oos_minimum_is_available",
        "refresh_01_02_03_then_run_05_when_future_oos_rows_are_available",
        "wait_for_new_unseen_bars_keep_control",
        "wait_for_new_future_oos_rows",
        "wait_for_new_future_oos_rows_after_manifest_verification",
    }:
        return "wait_for_new_future_oos_rows"
    if "enough fresh rows mature" in value:
        return "wait_for_new_future_oos_rows"
    if value in {
        "run_no_refit_future_oos_evaluator",
        "run_notebook_05_prediction_only",
    }:
        return "run_no_refit_future_oos_evaluator"
    if value in {
        "repair_preflight_blockers_do_not_evaluate",
        "repair_future_oos_preflight_without_refit",
    }:
        return "repair_future_oos_preflight_without_refit"
    return value


def _row(
    *,
    check: str,
    passed: bool,
    severity: str,
    expected: str,
    observed: str,
    details: str = "",
) -> dict[str, Any]:
    status = "passed" if passed else ("warning" if severity == "warning" else "failed")
    return {
        "check": check,
        "status": status,
        "severity": severity,
        "expected": expected,
        "observed": observed,
        "details": details,
    }


def _core_file_rows(report_dir: Path) -> list[dict[str, Any]]:
    required = [
        "auto_review.json",
        "phase2_readiness.json",
        "phase1_current_status.json",
        "next_research_protocol.json",
        "future_oos_preflight.json",
        "future_oos_readiness.json",
        "replacement_candidate_fit.json",
        "replacement_preregistration_patch.json",
        "frozen_candidate_manifest.json",
        "profile_comparison.csv",
        "model_performance_scorecard.csv",
        "experiment_selection.csv",
        "experiment_memory_registry.csv",
        "decision_report.json",
        "training_execution_summary.json",
    ]
    missing = [name for name in required if not (report_dir / name).exists()]
    empty = [
        name
        for name in required
        if (report_dir / name).exists() and (report_dir / name).stat().st_size == 0
    ]
    return [
        _row(
            check="core_report_files_present",
            passed=not missing and not empty,
            severity="error",
            expected="all core JSON/CSV reports exist and are non-empty",
            observed=f"missing={missing}; empty={empty}",
        )
    ]


def _unique_nonempty(values: dict[str, str]) -> set[str]:
    return {value for value in values.values() if value}


def build_report_consistency_audit(report_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return consistency rows and an operator-facing next-step summary."""

    path = Path(report_dir)
    auto_review = _read_json(path / "auto_review.json")
    phase2 = _read_json(path / "phase2_readiness.json")
    current = _read_json(path / "phase1_current_status.json")
    protocol = _read_json(path / "next_research_protocol.json")
    preflight = _read_json(path / "future_oos_preflight.json")
    readiness = _read_json(path / "future_oos_readiness.json")
    replacement = _read_json(path / "replacement_candidate_fit.json")
    frozen = _read_json(path / "frozen_candidate_manifest.json")
    failure_summary = _read_json(path / "future_oos_failure_summary.json")
    recency_manifest = _read_json(path / "recency_ensemble_manifest.json")
    candidate_plan = _read_csv(path / "future_oos_candidate_plan.csv")
    scorecard = _read_csv(path / "model_performance_scorecard.csv")
    missing_selected = _read_csv(path / "missing_selected_profiles.csv")
    selection = _read_csv(path / "experiment_selection.csv")
    memory = _read_csv(path / "experiment_memory_registry.csv")

    rows = _core_file_rows(path)
    evaluation_completed = bool(readiness.get("evaluation_completed", False))
    preflight_superseded = bool(
        evaluation_completed or preflight.get("lifecycle_superseded", False)
    )
    raw_actions = {
        "auto_review": _action_from_auto_review(auto_review),
        "phase2_readiness": str(phase2.get("next_action") or ""),
        "phase1_current_status": str(current.get("next_action") or ""),
        "next_research_protocol": str(protocol.get("next_action") or ""),
    }
    ignored_stage_actions: dict[str, str] = {}
    if preflight_superseded:
        ignored_stage_actions["future_oos_preflight"] = str(
            preflight.get("next_action") or ""
        )
        if preflight.get("current_lifecycle_action"):
            raw_actions["future_oos_preflight_current_lifecycle"] = str(
                preflight.get("current_lifecycle_action") or ""
            )
    else:
        raw_actions["future_oos_preflight"] = str(preflight.get("next_action") or "")
    actions = {key: _canonical_action(value) for key, value in raw_actions.items()}
    unique_actions = _unique_nonempty(actions)
    rows.append(
        _row(
            check="next_action_consistency",
            passed=len(unique_actions) <= 1,
            severity="error",
            expected="all current operator-facing next_action fields match",
            observed=json.dumps(
                {
                    "canonical": actions,
                    "raw": raw_actions,
                    "ignored_stage_actions": ignored_stage_actions,
                },
                sort_keys=True,
            ),
        )
    )

    phase2_blockers = [str(item) for item in phase2.get("blockers", []) or []]
    auto_blockers = [
        str(item)
        for item in (auto_review.get("phase2_readiness", {}) or {}).get("blockers", [])
        or []
    ]
    current_blockers = [str(item) for item in current.get("phase2_blockers", []) or []]
    blocker_sets = {
        "auto_review": sorted(auto_blockers),
        "phase2_readiness": sorted(phase2_blockers),
        "phase1_current_status": sorted(current_blockers),
    }
    rows.append(
        _row(
            check="phase2_blocker_consistency",
            passed=len({tuple(value) for value in blocker_sets.values()}) <= 1,
            severity="error",
            expected="phase2 blocker lists agree across reports",
            observed=json.dumps(blocker_sets, sort_keys=True),
        )
    )

    frozen_available = bool(frozen.get("available", False))
    replacement_fit_complete = (
        replacement.get("status") == "fit_complete_manifest_pin_required"
    )
    replacement_pin_pending = replacement_fit_complete and not frozen_available
    if replacement_pin_pending:
        rows.append(
            _row(
                check="replacement_fit_routes_to_manifest_pin",
                passed=unique_actions == {PIN_MANIFEST_ACTION}
                and protocol.get("status")
                == "replacement_fit_complete_manifest_pin_required"
                and bool(protocol.get("replacement_manifest_pin_required", False)),
                severity="error",
                expected=(
                    "completed replacement fit routes every report to manifest pin "
                    "before new OOS scoring"
                ),
                observed=(
                    f"actions={json.dumps(actions, sort_keys=True)}; "
                    f"protocol_status={protocol.get('status')}; "
                    f"pin_required={protocol.get('replacement_manifest_pin_required')}; "
                    f"frozen_available={frozen_available}"
                ),
            )
        )

    failed_future_oos = future_oos_failed(readiness, phase2_blockers)
    if failed_future_oos:
        expected_failure_action = (
            PIN_MANIFEST_ACTION
            if replacement_fit_complete
            else RETIRE_FAILED_FUTURE_OOS_ACTION
        )
        expected_protocol_status = (
            "failed_future_oos_replacement_manifest_pin_required"
            if replacement_fit_complete
            else "failed_future_oos_new_research_cycle_required"
        )
        expected_current_status = (
            "failed_future_oos_replacement_manifest_pin_required"
            if replacement_fit_complete
            else "failed_future_oos_new_research_cycle_required"
        )
        rows.append(
            _row(
                check="failed_future_oos_lifecycle_transition",
                passed=(
                    unique_actions == {expected_failure_action}
                    and protocol.get("status") == expected_protocol_status
                    and current.get("current_status") == expected_current_status
                    and bool(protocol.get("new_future_oos_anchor_required", False))
                    and not bool(current.get("run_04_required_now", False))
                    and not bool(current.get("run_05_first", True))
                ),
                severity="error",
                expected=(
                    "failed candidate is retired, same-window notebook routing is disabled, "
                    f"and current action is {expected_failure_action}"
                ),
                observed=(
                    f"actions={json.dumps(actions, sort_keys=True)}; "
                    f"protocol_status={protocol.get('status')}; "
                    f"current_status={current.get('current_status')}; "
                    f"new_anchor={protocol.get('new_future_oos_anchor_required')}; "
                    f"run04={current.get('run_04_required_now')}; "
                    f"run05_first={current.get('run_05_first')}"
                ),
            )
        )
        failed_candidate_id = str(readiness.get("primary_candidate_id") or "")
        protocol_replacement_id = str(protocol.get("replacement_candidate_id") or "")
        candidate_retired = bool(
            replacement_fit_complete
            or (
                protocol.get("failed_candidate_status")
                == "retired_after_failed_future_oos"
                and str(protocol.get("failed_candidate_id") or "")
                == failed_candidate_id
                and protocol_replacement_id != failed_candidate_id
            )
        )
        rows.append(
            _row(
                check="failed_candidate_not_reused_as_replacement",
                passed=candidate_retired,
                severity="error",
                expected="failed frozen candidate is retired and not exposed as the active replacement",
                observed=(
                    f"failed_candidate_id={failed_candidate_id}; "
                    f"failed_candidate_status={protocol.get('failed_candidate_status')}; "
                    f"replacement_candidate_id={protocol_replacement_id or 'none'}"
                ),
            )
        )

    if evaluation_completed:
        expected_current_action = (
            PIN_MANIFEST_ACTION
            if failed_future_oos and replacement_fit_complete
            else RETIRE_FAILED_FUTURE_OOS_ACTION
            if failed_future_oos
            else str(phase2.get("next_action") or "")
        )
        rows.append(
            _row(
                check="completed_evaluation_supersedes_preflight_action",
                passed=(
                    preflight.get("next_action_scope")
                    == "pre_evaluation_preflight_only"
                    and bool(preflight.get("lifecycle_superseded", False))
                    and str(preflight.get("current_lifecycle_action") or "")
                    == expected_current_action
                ),
                severity="error",
                expected=(
                    "preflight action is audit-only and current_lifecycle_action reflects "
                    "the completed evaluation"
                ),
                observed=(
                    f"scope={preflight.get('next_action_scope')}; "
                    f"superseded={preflight.get('lifecycle_superseded')}; "
                    f"current={preflight.get('current_lifecycle_action')}; "
                    f"expected_current={expected_current_action}"
                ),
            )
        )
        if recency_manifest:
            rows.append(
                _row(
                    check="completed_evaluation_supersedes_recency_manifest_action",
                    passed=(
                        recency_manifest.get("artifact_role")
                        == "historical_recency_research_snapshot"
                        and recency_manifest.get("next_action_scope")
                        == "historical_research_generation_time_only"
                        and bool(
                            recency_manifest.get("lifecycle_superseded", False)
                        )
                        and str(
                            recency_manifest.get("current_lifecycle_action") or ""
                        )
                        == expected_current_action
                    ),
                    severity="error",
                    expected=(
                        "copied recency manifest action is historical-only after "
                        "Future-OOS completes"
                    ),
                    observed=(
                        f"role={recency_manifest.get('artifact_role')}; "
                        f"scope={recency_manifest.get('next_action_scope')}; "
                        f"superseded={recency_manifest.get('lifecycle_superseded')}; "
                        f"current={recency_manifest.get('current_lifecycle_action')}"
                    ),
                )
            )

    rows.append(
        _row(
            check="phase2_ready_matches_blockers",
            passed=bool(phase2.get("ready_for_phase2", False)) is (not phase2_blockers),
            severity="error",
            expected="ready_for_phase2 is true iff blocker list is empty",
            observed=(
                f"ready_for_phase2={phase2.get('ready_for_phase2')}; "
                f"blockers={phase2_blockers}"
            ),
        )
    )

    frozen_missing = "frozen_candidate_manifest_unavailable" in phase2_blockers
    rows.append(
        _row(
            check="frozen_manifest_blocker_state",
            passed=(
                (frozen_missing and not bool(frozen.get("available", False)))
                or (not frozen_missing)
            ),
            severity="error",
            expected="frozen manifest blocker is paired with unavailable manifest",
            observed=f"blocker={frozen_missing}; manifest_available={frozen.get('available')}",
        )
    )

    anchor_values = {
        "frozen_manifest": str(frozen.get("anchor_data_end") or ""),
        "preflight": str(
            (preflight.get("primary_candidate", {}) or {}).get("anchor_data_end")
            or ""
        ),
        "readiness": str(readiness.get("anchor_data_end") or ""),
    }
    unique_anchors = _unique_nonempty(anchor_values)
    if len(unique_anchors) >= 1 and sum(bool(value) for value in anchor_values.values()) >= 2:
        rows.append(
            _row(
                check="future_oos_anchor_consistency",
                passed=len(unique_anchors) == 1,
                severity="error",
                expected="manifest, preflight, and readiness use one active anchor",
                observed=json.dumps(anchor_values, sort_keys=True),
            )
        )

    if {"new_labeled_rows", "min_rows", "min_rows_remaining"}.issubset(readiness):
        observed_rows = int(readiness.get("new_labeled_rows", 0) or 0)
        min_rows = int(readiness.get("min_rows", 0) or 0)
        remaining = int(readiness.get("min_rows_remaining", 0) or 0)
        expected_remaining = max(0, min_rows - observed_rows)
        rows.append(
            _row(
                check="future_oos_remaining_rows_arithmetic",
                passed=remaining == expected_remaining,
                severity="error",
                expected=f"min_rows_remaining={expected_remaining}",
                observed=(
                    f"new_labeled_rows={observed_rows}; min_rows={min_rows}; "
                    f"min_rows_remaining={remaining}"
                ),
            )
        )

    if (
        not candidate_plan.empty
        and "min_new_bars_remaining" in candidate_plan.columns
        and "min_rows_remaining" in readiness
    ):
        plan_remaining = set(
            pd.to_numeric(
                candidate_plan["min_new_bars_remaining"],
                errors="coerce",
            ).dropna().astype(int)
        )
        expected_remaining = int(readiness.get("min_rows_remaining", 0) or 0)
        rows.append(
            _row(
                check="candidate_plan_matches_future_oos_readiness",
                passed=plan_remaining == {expected_remaining},
                severity="error",
                expected=f"all plan rows show {expected_remaining} rows remaining",
                observed=str(sorted(plan_remaining)),
            )
        )

    if (
        not candidate_plan.empty
        and readiness.get("anchor_data_end")
        and readiness.get("min_rows") is not None
    ):
        expected_anchor = pd.to_datetime(
            readiness["anchor_data_end"],
            utc=True,
        )
        expected_plan_ready = expected_anchor + pd.Timedelta(
            hours=int(readiness.get("min_rows", 0) or 0)
        )
        if "anchor_data_end" in candidate_plan.columns:
            plan_anchors = pd.to_datetime(
                candidate_plan["anchor_data_end"],
                utc=True,
                errors="coerce",
            ).dropna()
            anchor_passed = bool(
                len(plan_anchors) == len(candidate_plan)
                and plan_anchors.eq(expected_anchor).all()
            )
            observed_plan_anchors = sorted(
                {value.isoformat() for value in plan_anchors}
            )
        else:
            anchor_passed = False
            observed_plan_anchors = ["missing_anchor_data_end_column"]
        rows.append(
            _row(
                check="candidate_plan_active_anchor_consistency",
                passed=anchor_passed,
                severity="error",
                expected=expected_anchor.isoformat(),
                observed=str(observed_plan_anchors),
            )
        )

        if "min_ready_at" in candidate_plan.columns:
            plan_ready_dates = pd.to_datetime(
                candidate_plan["min_ready_at"],
                utc=True,
                errors="coerce",
            ).dropna()
            ready_date_passed = bool(
                len(plan_ready_dates) == len(candidate_plan)
                and plan_ready_dates.eq(expected_plan_ready).all()
            )
            observed_plan_ready = sorted(
                {value.isoformat() for value in plan_ready_dates}
            )
        else:
            ready_date_passed = False
            observed_plan_ready = ["missing_min_ready_at_column"]
        rows.append(
            _row(
                check="candidate_plan_min_ready_date_consistency",
                passed=ready_date_passed,
                severity="error",
                expected=expected_plan_ready.isoformat(),
                observed=str(observed_plan_ready),
            )
        )

    if readiness.get("anchor_data_end") and readiness.get("min_ready_at"):
        try:
            anchor_ts = pd.to_datetime(readiness["anchor_data_end"], utc=True)
            expected_min_ready = anchor_ts + pd.Timedelta(
                hours=int(readiness.get("min_rows", 0) or 0)
            )
            reported_min_ready = pd.to_datetime(
                readiness["min_ready_at"],
                utc=True,
            )
            date_passed = expected_min_ready == reported_min_ready
        except (TypeError, ValueError):
            expected_min_ready = "valid timestamp"
            reported_min_ready = readiness.get("min_ready_at")
            date_passed = False
        rows.append(
            _row(
                check="future_oos_min_ready_date_arithmetic",
                passed=date_passed,
                severity="error",
                expected=str(expected_min_ready),
                observed=str(reported_min_ready),
            )
        )

    if readiness.get("anchor_data_end") and readiness.get("min_raw_data_ready_at"):
        try:
            anchor_ts = pd.to_datetime(readiness["anchor_data_end"], utc=True)
            expected_raw_ready = anchor_ts + pd.Timedelta(
                hours=(
                    int(readiness.get("min_rows", 0) or 0)
                    + int(readiness.get("label_maturity_horizon_bars", 0) or 0)
                )
            )
            reported_raw_ready = pd.to_datetime(
                readiness["min_raw_data_ready_at"],
                utc=True,
            )
            raw_date_passed = expected_raw_ready == reported_raw_ready
        except (TypeError, ValueError):
            expected_raw_ready = "valid timestamp"
            reported_raw_ready = readiness.get("min_raw_data_ready_at")
            raw_date_passed = False
        rows.append(
            _row(
                check="future_oos_raw_data_ready_date_arithmetic",
                passed=raw_date_passed,
                severity="error",
                expected=str(expected_raw_ready),
                observed=str(reported_raw_ready),
            )
        )

    activation = readiness.get("primary_candidate_activation", {}) or {}
    placeholder_note = str(failure_summary.get("note") or "").lower()
    activation_valid = bool(activation.get("activated", False)) or bool(
        frozen_available and preflight.get("invariants_passed", False)
    )
    if activation_valid and not bool(
        readiness.get("evaluation_completed", False)
    ):
        rows.append(
            _row(
                check="active_candidate_placeholder_wording",
                passed="no active hash-pinned" not in placeholder_note,
                severity="error",
                expected="placeholder acknowledges active candidate waiting state",
                observed=str(failure_summary.get("note") or ""),
            )
        )

    rows.append(
        _row(
            check="missing_selected_profiles_empty",
            passed=missing_selected.empty,
            severity="error",
            expected="no selected profile is missing diagnostics outputs",
            observed=f"missing_selected_rows={len(missing_selected)}",
        )
    )

    if not selection.empty and not memory.empty:
        rejected = set(
            memory.loc[
                memory["memory_status"].astype(str).eq("rejected"),
                "profile",
            ].astype(str)
        )
        selected = set(
            selection.loc[
                selection["selected"].astype(str).str.lower().isin({"true", "1", "yes"}),
                "profile",
            ].astype(str)
        )
        overlap = sorted(selected & rejected)
    else:
        overlap = []
    rows.append(
        _row(
            check="selected_profiles_not_rejected",
            passed=not overlap,
            severity="error",
            expected="automatic selected profiles are not in rejected experiment memory",
            observed=",".join(overlap),
        )
    )

    report_complete_rows = pd.DataFrame()
    if not scorecard.empty and {"metric", "value"}.issubset(scorecard.columns):
        report_complete_rows = scorecard.loc[
            scorecard["metric"].astype(str).eq("report_complete")
        ]
    scorecard_report_complete = (
        str(report_complete_rows.iloc[0]["value"]).lower() in {"true", "1", "yes"}
        if not report_complete_rows.empty
        else False
    )
    auto_complete = bool(
        (auto_review.get("report_completeness", {}) or {}).get("complete", False)
    )
    rows.append(
        _row(
            check="report_complete_consistency",
            passed=scorecard_report_complete == auto_complete,
            severity="warning",
            expected="scorecard report_complete agrees with auto_review completeness",
            observed=(
                f"scorecard={scorecard_report_complete}; "
                f"auto_review={auto_complete}"
            ),
        )
    )

    report_files = [item for item in path.iterdir() if item.is_file()]
    zero_byte = sorted(item.name for item in report_files if item.stat().st_size == 0)
    rows.append(
        _row(
            check="no_zero_byte_report_files",
            passed=not zero_byte,
            severity="error",
            expected="report directory has no zero-byte files",
            observed=",".join(zero_byte),
        )
    )

    frame = pd.DataFrame(rows)
    failed = frame.loc[frame["status"].eq("failed")]
    warnings = frame.loc[frame["status"].eq("warning")]
    action = next(iter(unique_actions), "") if len(unique_actions) == 1 else ""
    operator = {
        "run_id": current.get("run_id") or auto_review.get("run_id") or "",
        "consistency_status": "failed" if not failed.empty else "passed",
        "failed_checks": failed["check"].tolist(),
        "warning_checks": warnings["check"].tolist(),
        "phase2_ready": bool(phase2.get("ready_for_phase2", False)),
        "active_blocker": current.get("active_blocker") or (
            phase2_blockers[0] if phase2_blockers else ""
        ),
        "next_action": action or current.get("next_action") or phase2.get("next_action"),
        "phase2_track_next_action": current.get("phase2_track_next_action")
        or action
        or current.get("next_action")
        or phase2.get("next_action"),
        "historical_research_status": current.get("historical_research_status"),
        "historical_research_result": current.get("historical_research_result"),
        "historical_research_next_action": current.get(
            "historical_research_next_action"
        ),
        "current_status": current.get("current_status"),
        "run_04_required_now": bool(current.get("run_04_required_now", False)),
        "run_05_first": bool(current.get("run_05_first", True)),
        "next_notebook": current.get("next_notebook"),
        "replacement_candidate_id": replacement.get("candidate_id")
        or protocol.get("replacement_candidate_id"),
        "replacement_candidate_fit_status": replacement.get("status")
        or protocol.get("replacement_candidate_fit_status"),
        "replacement_manifest_pin_required": bool(
            (
                replacement.get("manifest_pin_required", False)
                or protocol.get("replacement_manifest_pin_required", False)
            )
            and not frozen_available
        ),
        "future_oos_preflight_state": preflight.get("state"),
        "future_oos_ready_for_evaluation": bool(
            readiness.get("ready_for_evaluation", False)
        ),
        "future_oos_evaluation_completed": evaluation_completed,
        "future_oos_evaluation_state": readiness.get("evaluation_state"),
        "future_oos_primary_candidate_passed": readiness.get(
            "primary_candidate_passed"
        ),
        "preflight_action_superseded": preflight_superseded,
    }
    return frame, operator


def _operator_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Operator Next Step",
        "",
        f"- Run id: `{payload.get('run_id')}`",
        f"- Consistency status: `{payload.get('consistency_status')}`",
        f"- Current status: `{payload.get('current_status')}`",
        f"- Phase 2 ready: `{payload.get('phase2_ready')}`",
        f"- Active blocker: `{payload.get('active_blocker') or 'none'}`",
        f"- Phase 2 track next action: `{payload.get('phase2_track_next_action')}`",
        f"- Historical research status: `{payload.get('historical_research_status')}`",
        f"- Historical research result: `{payload.get('historical_research_result')}`",
        f"- Historical research next action: `{payload.get('historical_research_next_action')}`",
        f"- Run 04 required now: `{payload.get('run_04_required_now')}`",
        f"- Run 05 first: `{payload.get('run_05_first')}`",
        f"- Next notebook: `{payload.get('next_notebook')}`",
        f"- Replacement candidate: `{payload.get('replacement_candidate_id') or 'none'}`",
        f"- Replacement fit status: `{payload.get('replacement_candidate_fit_status') or 'none'}`",
        f"- Replacement manifest pin required: `{payload.get('replacement_manifest_pin_required')}`",
        f"- Future-OOS preflight state: `{payload.get('future_oos_preflight_state')}`",
        f"- Future-OOS evaluation state: `{payload.get('future_oos_evaluation_state')}`",
        f"- Preflight action superseded: `{payload.get('preflight_action_superseded')}`",
        "",
    ]
    failed = payload.get("failed_checks") or []
    warnings = payload.get("warning_checks") or []
    if failed:
        lines.extend(["## Failed Consistency Checks", ""])
        lines.extend(f"- `{item}`" for item in failed)
        lines.append("")
    if warnings:
        lines.extend(["## Warning Checks", ""])
        lines.extend(f"- `{item}`" for item in warnings)
        lines.append("")
    return "\n".join(lines)


def write_report_consistency_audit(report_dir: str | Path) -> dict[str, Any]:
    """Write bundle integrity/consistency artifacts and return operator payload."""

    path = Path(report_dir)
    frame, operator = build_report_consistency_audit(path)
    frame.to_csv(path / "report_consistency_audit.csv", index=False)
    (path / "report_consistency_audit.md").write_text(
        _table_markdown("Report Consistency Audit", frame),
        encoding="utf-8",
    )
    _write_json(
        path / "report_consistency_audit.json",
        {"rows": frame.to_dict(orient="records"), "operator_next_step": operator},
    )
    _write_json(path / "operator_next_step.json", _json_ready(operator))
    (path / "operator_next_step.md").write_text(
        _operator_markdown(operator),
        encoding="utf-8",
    )
    return operator
