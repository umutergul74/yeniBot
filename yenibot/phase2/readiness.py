from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yenibot.phase2.contracts import Phase2Mode


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "passed", "ready"}


@dataclass(frozen=True)
class Phase2Gate:
    """Phase 2 authorization state derived from Phase 1 reports."""

    report_dir: Path
    ready_for_phase2: bool
    report_consistency_passed: bool
    future_oos_evaluation_completed: bool
    future_oos_candidate_passed: bool
    promotion_allowed: bool
    blockers: tuple[str, ...]
    advisories: tuple[str, ...] = ()
    next_action: str = ""

    @property
    def official_allowed(self) -> bool:
        return (
            self.ready_for_phase2
            and self.report_consistency_passed
            and self.future_oos_evaluation_completed
            and self.future_oos_candidate_passed
            and self.promotion_allowed
            and not self.blockers
        )

    @property
    def sandbox_allowed(self) -> bool:
        return True

    @property
    def evidence_status(self) -> str:
        if self.official_allowed:
            return "official_phase2_allowed"
        if (
            self.future_oos_evaluation_completed
            and not self.future_oos_candidate_passed
        ):
            return "sandbox_retired_candidate_historical_audit_only"
        return "sandbox_not_promotable_until_future_oos_passes"

    def assert_mode_allowed(self, mode: Phase2Mode) -> None:
        if mode == "sandbox":
            return
        if mode != "official":
            raise ValueError(f"Unsupported Phase 2 mode: {mode}")
        if not self.official_allowed:
            raise RuntimeError(
                "Official Phase 2 is blocked. Use sandbox mode only until "
                "phase2_readiness, future-OOS pass, promotion_allowed, and "
                "report consistency all pass."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_dir": str(self.report_dir),
            "ready_for_phase2": self.ready_for_phase2,
            "report_consistency_passed": self.report_consistency_passed,
            "future_oos_evaluation_completed": self.future_oos_evaluation_completed,
            "future_oos_candidate_passed": self.future_oos_candidate_passed,
            "promotion_allowed": self.promotion_allowed,
            "blockers": list(self.blockers),
            "advisories": list(self.advisories),
            "next_action": self.next_action,
            "official_allowed": self.official_allowed,
            "sandbox_allowed": self.sandbox_allowed,
            "evidence_status": self.evidence_status,
        }


def load_phase2_gate(report_dir: str | Path) -> Phase2Gate:
    """Load the fail-closed Phase 2 gate from a report directory."""

    root = Path(report_dir)
    phase2 = _read_json(root / "phase2_readiness.json")
    future = _read_json(root / "future_oos_readiness.json")
    consistency = _read_json(root / "report_consistency_audit.json")
    operator = consistency.get("operator_next_step", {}) or {}

    blockers = tuple(str(item) for item in phase2.get("blockers", []) or [])
    advisories = tuple(str(item) for item in phase2.get("advisories", []) or [])
    consistency_status = str(operator.get("consistency_status") or "").lower()
    failed_checks = operator.get("failed_checks", []) or []

    return Phase2Gate(
        report_dir=root,
        ready_for_phase2=_to_bool(phase2.get("ready_for_phase2")),
        report_consistency_passed=consistency_status == "passed" and not failed_checks,
        future_oos_evaluation_completed=_to_bool(future.get("evaluation_completed")),
        future_oos_candidate_passed=future.get("primary_candidate_passed") is True,
        promotion_allowed=_to_bool(future.get("promotion_allowed")),
        blockers=blockers,
        advisories=advisories,
        next_action=str(phase2.get("next_action") or operator.get("next_action") or ""),
    )
