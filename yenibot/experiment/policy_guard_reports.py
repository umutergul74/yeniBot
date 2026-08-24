"""Compact report adapters for the experiment policy guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _table_markdown, _write_json
from yenibot.experiment.configuration import _experiment_policy_guard, _future_oos_ready_at_fields


def _experiment_policy_guard_frame(
    settings: dict[str, Any],
    config: dict[str, Any],
    future_oos_readiness: dict[str, Any] | None = None,
) -> pd.DataFrame:
    guard = _experiment_policy_guard(settings, config, future_oos_readiness=future_oos_readiness)
    ready_at = _future_oos_ready_at_fields(guard)
    row = {
        "enabled": bool(guard.get("enabled", False)),
        "status": str(guard.get("status", "")),
        "profile_search_locked": bool(guard.get("profile_search_locked", False)),
        "action": str(guard.get("action", "")),
        "reason": str(guard.get("reason", "")),
        "allowed_benchmark_profiles": ",".join(str(item) for item in guard.get("allowed_benchmark_profiles", []) or []),
        "blocked_candidate_profiles": ",".join(str(item) for item in guard.get("blocked_candidate_profiles", []) or []),
        "blocked_full_profiles": ",".join(str(item) for item in guard.get("blocked_full_profiles", []) or []),
        "blocked_seed_profiles": ",".join(str(item) for item in guard.get("blocked_seed_profiles", []) or []),
        "future_oos_ready": bool(guard.get("future_oos_ready", False)),
        "future_oos_preferred_ready": bool(guard.get("future_oos_preferred_ready", False)),
        "new_bars_since_anchor": int(guard.get("new_bars_since_anchor", 0) or 0),
        "min_new_bars_remaining": int(guard.get("min_new_bars_remaining", 0) or 0),
        "preferred_new_bars_remaining": int(guard.get("preferred_new_bars_remaining", 0) or 0),
        "min_ready_at": ready_at["min_ready_at"],
        "preferred_ready_at": ready_at["preferred_ready_at"],
        "min_raw_data_ready_at": ready_at["min_raw_data_ready_at"],
        "preferred_raw_data_ready_at": ready_at["preferred_raw_data_ready_at"],
        "holdout_roll_forward_locked": bool(guard.get("holdout_roll_forward_locked", False)),
        "next_action": str(guard.get("next_action", "")),
        "monitor_next_action": str(guard.get("monitor_next_action", "")),
        "anchor_run_id": str(guard.get("anchor_run_id", "")),
        "anchor_data_end": str(guard.get("anchor_data_end", "")),
        "latest_available_data_end": str(guard.get("latest_available_data_end", "")),
        "state_source": str(guard.get("state_source", "")),
        "readiness_basis": str(guard.get("readiness_basis", "")),
        "candidate_activation_valid": bool(guard.get("candidate_activation_valid", False)),
        "configured_candidate_status": str(guard.get("configured_candidate_status", "")),
        "evaluation_completed": bool(guard.get("evaluation_completed", False)),
        "primary_candidate_passed": guard.get("primary_candidate_passed"),
    }
    return pd.DataFrame([row])


def _write_experiment_policy_guard(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "experiment_policy_guard.csv", index=False)
    (path / "experiment_policy_guard.md").write_text(
        _table_markdown("Experiment Policy Guard", frame), encoding="utf-8"
    )
    _write_json(path / "experiment_policy_guard.json", {"rows": frame.to_dict(orient="records")})


def _recommendation_with_policy_guard(recommendation: str, settings: dict[str, Any]) -> str:
    guard = settings.get("experiment_policy_guard", {}) or {}
    if bool(guard.get("profile_search_locked", False)) and recommendation not in {
        "fix_missing_selected_profiles",
        "rerun_training_with_holdout_split",
    }:
        return str(guard.get("action") or "wait_for_new_unseen_bars_keep_control_profile")
    return recommendation
