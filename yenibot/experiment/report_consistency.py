"""Cross-report consistency checks for Phase 1 diagnostics bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _json_ready, _table_markdown, _write_json

__all__ = [
    "build_report_consistency_audit",
    "write_report_consistency_audit",
]


PIN_MANIFEST_ACTION = "pin_replacement_candidate_manifest_and_activate_new_oos_anchor"


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
    scorecard = _read_csv(path / "model_performance_scorecard.csv")
    missing_selected = _read_csv(path / "missing_selected_profiles.csv")
    selection = _read_csv(path / "experiment_selection.csv")
    memory = _read_csv(path / "experiment_memory_registry.csv")

    rows = _core_file_rows(path)
    actions = {
        "auto_review": _action_from_auto_review(auto_review),
        "phase2_readiness": str(phase2.get("next_action") or ""),
        "phase1_current_status": str(current.get("next_action") or ""),
        "next_research_protocol": str(protocol.get("next_action") or ""),
        "future_oos_preflight": str(preflight.get("next_action") or ""),
    }
    unique_actions = _unique_nonempty(actions)
    rows.append(
        _row(
            check="next_action_consistency",
            passed=len(unique_actions) <= 1,
            severity="error",
            expected="all operator-facing next_action fields match",
            observed=json.dumps(actions, sort_keys=True),
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

    replacement_fit_complete = (
        replacement.get("status") == "fit_complete_manifest_pin_required"
    )
    if replacement_fit_complete:
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
                    f"pin_required={protocol.get('replacement_manifest_pin_required')}"
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
        "current_status": current.get("current_status"),
        "run_04_required_now": bool(current.get("run_04_required_now", False)),
        "run_05_first": bool(current.get("run_05_first", True)),
        "replacement_candidate_id": replacement.get("candidate_id")
        or protocol.get("replacement_candidate_id"),
        "replacement_candidate_fit_status": replacement.get("status")
        or protocol.get("replacement_candidate_fit_status"),
        "replacement_manifest_pin_required": bool(
            replacement.get("manifest_pin_required", False)
            or protocol.get("replacement_manifest_pin_required", False)
        ),
        "future_oos_preflight_state": preflight.get("state"),
        "future_oos_ready_for_evaluation": bool(
            readiness.get("ready_for_evaluation", False)
        ),
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
        f"- Next action: `{payload.get('next_action')}`",
        f"- Run 04 required now: `{payload.get('run_04_required_now')}`",
        f"- Run 05 first: `{payload.get('run_05_first')}`",
        f"- Replacement candidate: `{payload.get('replacement_candidate_id') or 'none'}`",
        f"- Replacement fit status: `{payload.get('replacement_candidate_fit_status') or 'none'}`",
        f"- Replacement manifest pin required: `{payload.get('replacement_manifest_pin_required')}`",
        f"- Future-OOS preflight state: `{payload.get('future_oos_preflight_state')}`",
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
