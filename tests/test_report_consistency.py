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


def test_write_report_consistency_outputs_operator_files(tmp_path: Path) -> None:
    _write_base_report(tmp_path)

    operator = write_report_consistency_audit(tmp_path)

    assert operator["consistency_status"] == "passed"
    assert (tmp_path / "report_consistency_audit.csv").exists()
    assert (tmp_path / "report_consistency_audit.json").exists()
    assert (tmp_path / "report_consistency_audit.md").exists()
    assert (tmp_path / "operator_next_step.json").exists()
    assert (tmp_path / "operator_next_step.md").exists()
