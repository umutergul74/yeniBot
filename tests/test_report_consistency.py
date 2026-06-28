from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from yenibot.experiment.report_consistency import (
    PIN_MANIFEST_ACTION,
    build_report_consistency_audit,
    write_report_consistency_audit,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_base_report(path: Path, *, protocol_action: str = PIN_MANIFEST_ACTION) -> None:
    path.mkdir(parents=True, exist_ok=True)
    blockers = ["frozen_candidate_manifest_unavailable"]
    _write_json(
        path / "auto_review.json",
        {
            "run_id": "run",
            "next_action": {"action": PIN_MANIFEST_ACTION},
            "phase2_readiness": {"blockers": blockers},
            "report_completeness": {"complete": True},
        },
    )
    _write_json(
        path / "phase2_readiness.json",
        {
            "ready_for_phase2": False,
            "blockers": blockers,
            "next_action": PIN_MANIFEST_ACTION,
        },
    )
    _write_json(
        path / "phase1_current_status.json",
        {
            "run_id": "run",
            "current_status": "historical_model_evidence_passed_awaiting_replacement_manifest_pin",
            "phase2_blockers": blockers,
            "active_blocker": blockers[0],
            "next_action": PIN_MANIFEST_ACTION,
            "run_04_required_now": False,
            "run_05_first": True,
        },
    )
    _write_json(
        path / "next_research_protocol.json",
        {
            "status": "replacement_fit_complete_manifest_pin_required",
            "next_action": protocol_action,
            "replacement_candidate_fit_status": "fit_complete_manifest_pin_required",
            "replacement_candidate_id": "control_recent3_equal_v2",
            "replacement_manifest_pin_required": True,
        },
    )
    _write_json(
        path / "future_oos_preflight.json",
        {
            "state": "awaiting_replacement_preregistration",
            "next_action": PIN_MANIFEST_ACTION,
        },
    )
    _write_json(path / "future_oos_readiness.json", {"ready_for_evaluation": False})
    _write_json(
        path / "replacement_candidate_fit.json",
        {
            "status": "fit_complete_manifest_pin_required",
            "candidate_id": "control_recent3_equal_v2",
            "manifest_pin_required": True,
        },
    )
    _write_json(path / "replacement_preregistration_patch.json", {"status": "ready"})
    _write_json(path / "frozen_candidate_manifest.json", {"available": False})
    _write_json(path / "decision_report.json", {"run_id": "run"})
    _write_json(path / "training_execution_summary.json", {"run_id": "run"})
    pd.DataFrame([{"profile": "control", "fold_scope": "full"}]).to_csv(
        path / "profile_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        [{"metric": "report_complete", "value": True, "status": "passed"}]
    ).to_csv(path / "model_performance_scorecard.csv", index=False)
    pd.DataFrame([{"profile": "control", "selected": True}]).to_csv(
        path / "experiment_selection.csv",
        index=False,
    )
    pd.DataFrame(
        columns=[
            "profile",
            "memory_status",
            "reason",
            "allow_retest",
            "auto_retest_blocked",
            "source",
        ]
    ).to_csv(path / "experiment_memory_registry.csv", index=False)
    pd.DataFrame(columns=["profile", "role", "reason"]).to_csv(
        path / "missing_selected_profiles.csv",
        index=False,
    )


def test_report_consistency_passes_for_manifest_pin_state(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    frame, operator = build_report_consistency_audit(tmp_path)

    assert set(frame["status"]) == {"passed"}
    assert operator["consistency_status"] == "passed"
    assert operator["next_action"] == PIN_MANIFEST_ACTION
    assert operator["replacement_candidate_id"] == "control_recent3_equal_v2"


def test_report_consistency_fails_on_stale_research_protocol_action(tmp_path: Path) -> None:
    _write_base_report(
        tmp_path,
        protocol_action="select_and_preregister_replacement_candidate_from_historical_cv_only",
    )

    frame, operator = build_report_consistency_audit(tmp_path)
    failed = set(frame.loc[frame["status"].eq("failed"), "check"])

    assert "next_action_consistency" in failed
    assert "replacement_fit_routes_to_manifest_pin" in failed
    assert operator["consistency_status"] == "failed"


def test_report_consistency_allows_historical_replacement_fit_after_manifest_pin(
    tmp_path: Path,
) -> None:
    _write_base_report(tmp_path, protocol_action="wait_for_new_future_oos_rows")
    blockers = ["future_unseen_oos_not_ready"]
    for filename in [
        "auto_review.json",
        "phase2_readiness.json",
        "phase1_current_status.json",
        "next_research_protocol.json",
        "future_oos_preflight.json",
    ]:
        payload = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        if filename == "auto_review.json":
            payload["next_action"] = {"action": "wait_for_new_future_oos_rows"}
            payload["phase2_readiness"]["blockers"] = blockers
        elif filename == "phase2_readiness.json":
            payload["blockers"] = blockers
            payload["next_action"] = "wait_for_new_future_oos_rows"
        elif filename == "phase1_current_status.json":
            payload["phase2_blockers"] = blockers
            payload["active_blocker"] = blockers[0]
            payload["next_action"] = "wait_for_new_future_oos_rows"
        elif filename == "next_research_protocol.json":
            payload["status"] = "replacement_candidate_manifest_pinned_awaiting_future_oos"
            payload["next_action"] = "wait_for_new_future_oos_rows"
            payload["replacement_manifest_pin_required"] = False
        elif filename == "future_oos_preflight.json":
            payload["state"] = "waiting_for_min_rows"
            payload["next_action"] = "wait_for_new_future_oos_rows"
        _write_json(tmp_path / filename, payload)
    _write_json(
        tmp_path / "frozen_candidate_manifest.json",
        {"available": True, "manifest_hash": "abc", "expected_manifest_hash": "abc"},
    )

    frame, operator = build_report_consistency_audit(tmp_path)

    assert set(frame["status"]) == {"passed"}
    assert operator["consistency_status"] == "passed"
    assert operator["next_action"] == "wait_for_new_future_oos_rows"
    assert operator["replacement_manifest_pin_required"] is False


def test_report_consistency_canonicalizes_equivalent_wait_actions(tmp_path: Path) -> None:
    _write_base_report(tmp_path, protocol_action="wait_for_new_future_oos_rows")
    blockers = ["future_unseen_oos_not_ready"]
    replacements = {
        "auto_review.json": {"next_action": {"action": "wait_for_new_unseen_bars_keep_control"}},
        "phase2_readiness.json": {"next_action": "wait_for_new_unseen_bars_keep_control"},
        "phase1_current_status.json": {
            "next_action": "refresh_data_and_run_05_when_future_oos_minimum_is_available"
        },
        "next_research_protocol.json": {
            "status": "replacement_candidate_manifest_pinned_awaiting_future_oos",
            "next_action": "Run notebook 05 to verify the pinned manifest. Future-OOS scoring remains blocked until enough fresh rows mature after the 2026-06-13 anchor.",
            "replacement_manifest_pin_required": False,
        },
        "future_oos_preflight.json": {
            "state": "waiting_for_mature_labeled_rows",
            "next_action": "refresh_01_02_03_then_recheck_without_running_04",
        },
    }
    for filename, updates in replacements.items():
        payload = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        payload.update(updates)
        if filename == "auto_review.json":
            payload["phase2_readiness"]["blockers"] = blockers
        if filename == "phase2_readiness.json":
            payload["blockers"] = blockers
        if filename == "phase1_current_status.json":
            payload["phase2_blockers"] = blockers
            payload["active_blocker"] = blockers[0]
        _write_json(tmp_path / filename, payload)
    _write_json(
        tmp_path / "frozen_candidate_manifest.json",
        {"available": True, "manifest_hash": "abc", "expected_manifest_hash": "abc"},
    )

    frame, operator = build_report_consistency_audit(tmp_path)

    assert set(frame["status"]) == {"passed"}
    assert operator["consistency_status"] == "passed"
    assert operator["next_action"] == "wait_for_new_future_oos_rows"


def test_write_report_consistency_outputs_operator_files(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    operator = write_report_consistency_audit(tmp_path)

    assert operator["consistency_status"] == "passed"
    assert (tmp_path / "report_consistency_audit.csv").exists()
    assert (tmp_path / "report_consistency_audit.json").exists()
    assert (tmp_path / "report_consistency_audit.md").exists()
    assert (tmp_path / "operator_next_step.json").exists()
    assert (tmp_path / "operator_next_step.md").exists()


def test_report_consistency_detects_stale_future_oos_state_sources(
    tmp_path: Path,
) -> None:
    _write_base_report(tmp_path)
    _write_json(
        tmp_path / "frozen_candidate_manifest.json",
        {
            "available": True,
            "anchor_data_end": "2026-06-13T01:00:00+00:00",
        },
    )
    _write_json(
        tmp_path / "future_oos_preflight.json",
        {
            "state": "waiting_for_mature_labeled_rows",
            "next_action": PIN_MANIFEST_ACTION,
            "primary_candidate": {
                "anchor_data_end": "2026-05-13T08:00:00+00:00",
            },
        },
    )
    _write_json(
        tmp_path / "future_oos_readiness.json",
        {
            "anchor_data_end": "2026-06-13T01:00:00+00:00",
            "new_labeled_rows": 313,
            "min_rows": 720,
            "min_rows_remaining": 0,
            "min_ready_at": "2026-06-12T08:00:00+00:00",
            "ready_for_evaluation": False,
            "evaluation_completed": False,
            "primary_candidate_activation": {"activated": True},
        },
    )
    _write_json(
        tmp_path / "future_oos_failure_summary.json",
        {
            "note": (
                "Future-OOS prediction reports are placeholders because no active "
                "hash-pinned frozen candidate was available."
            )
        },
    )
    pd.DataFrame([{"min_new_bars_remaining": 0}]).to_csv(
        tmp_path / "future_oos_candidate_plan.csv",
        index=False,
    )

    frame, operator = build_report_consistency_audit(tmp_path)
    failed = set(frame.loc[frame["status"].eq("failed"), "check"])

    assert "future_oos_anchor_consistency" in failed
    assert "future_oos_remaining_rows_arithmetic" in failed
    assert "future_oos_min_ready_date_arithmetic" in failed
    assert "active_candidate_placeholder_wording" in failed
    assert operator["consistency_status"] == "failed"
