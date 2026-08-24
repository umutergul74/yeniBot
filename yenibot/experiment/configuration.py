"""Experiment policy, profile resolution, preflight validation, and run manifests."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
import pandas as pd
from yenibot.features import filter_feature_columns, resolve_feature_profile, select_feature_columns
from yenibot.training import PurgedWalkForwardCV
from yenibot.reproducibility import training_code_signature_payload

from yenibot.experiment.common import (
    _cfg,
    _deep_update,
    _hash_payload,
    _read_json,
    _set_cfg,
    _slug,
    _write_json,
)

_TRAINING_EXECUTION_KEYS = (
    "run_id_source",
    "training_executed_count",
    "training_skipped_count",
    "all_training_scopes_reused",
    "reused_training_scopes",
)

_CURRENT_DIAGNOSTIC_EXPERIMENT_KEYS = (
    "diagnostics",
    "experiment_memory",
    "frozen_candidate_outcomes",
    "frozen_candidates",
    "future_oos_validation",
    "next_research_cycle",
    "policy_review",
    "profile_blends",
    "research_focus",
    "seed_audit",
)

__all__ = [
    'profile_config',
    '_profile_config_overrides',
    '_profile_rejection_reason',
    '_filter_memory_rejected_profiles',
    '_policy_status_is_retired_or_failed',
    '_future_oos_allowed_benchmark_profiles',
    '_experiment_policy_guard',
    '_apply_experiment_policy_guard',
    'validate_training_research_contract',
    'experiment_settings',
    '_available_walk_forward_fold_ids',
    '_resolve_seed_audit_fold_ids',
    '_resolve_seed_audit_fold_ids_for_seed',
    '_attach_resolved_seed_audit_folds',
    '_diagnostic_seed_audit_settings',
    '_validate_requested_fold_ids',
    '_preflight_fold_plans',
    '_profile_requires_intrahour_features',
    '_profile_requires_futures_context_features',
    '_missing_intrahour_include_patterns',
    '_preflight_experiment_profiles',
    'experiment_root',
    'new_run_id',
    '_training_config_payload',
    '_training_code_signature_payload',
    '_diagnostic_config_for_run',
    '_diagnostics_signature',
    '_experiment_signature',
    '_matching_latest_run',
    'resolve_experiment_run_id',
    'latest_experiment_run',
    'profile_run_dir',
    '_frame_window',
    '_frame_fingerprint',
    '_training_signature',
    '_manifest_path',
    '_training_execution_summary_path',
    '_training_execution_summary',
    '_load_training_execution_summary',
    '_is_complete',
    '_experiment_selection_frame',
    '_experiment_selection_markdown',
    '_write_experiment_selection',
    '_missing_selected_profiles',
    '_missing_selected_markdown',
    '_write_missing_selected_profiles',
    '_holdout_latest_available_data_end',
    '_active_future_oos_state',
    '_future_oos_monitor_state',
    '_future_oos_ready_at_fields',
]

def validate_training_research_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Validate stable notebook-04 safety invariants without pinning lifecycle labels."""

    experiments = _cfg(config, ["experiments"], {}) or {}
    research = experiments.get("next_research_cycle", {}) or {}
    research_focus = experiments.get("research_focus", {}) or {}
    seed_audit = experiments.get("seed_audit", {}) or {}

    errors: list[str] = []
    research_status = str(research.get("status") or "").strip()
    research_focus_status = str(research_focus.get("status") or "").strip()

    if not research_status:
        errors.append("experiments.next_research_cycle.status must be set")
    if str(research_focus.get("mode") or "") != "walk_forward_cv_repair":
        errors.append("experiments.research_focus.mode must be walk_forward_cv_repair")
    if not research_focus_status:
        errors.append("experiments.research_focus.status must be set")
    if not bool(seed_audit.get("enabled", False)):
        errors.append("experiments.seed_audit.enabled must be true")
    if str(seed_audit.get("mode") or "") != "in_run":
        errors.append("experiments.seed_audit.mode must be in_run")
    if seed_audit.get("source_run_id") not in (None, ""):
        errors.append("experiments.seed_audit.source_run_id must be empty for in-run audits")
    if research.get("same_window_selection_allowed") is not False:
        errors.append("experiments.next_research_cycle.same_window_selection_allowed must be false")
    if research.get("new_future_oos_anchor_required") is not True:
        errors.append("experiments.next_research_cycle.new_future_oos_anchor_required must be true")

    adaptive = research.get("adaptive_ensemble", {}) or {}
    if bool(adaptive.get("enabled", False)):
        policy = adaptive.get("policy", {}) or {}
        control = adaptive.get("control", {}) or {}
        replacement = research.get("replacement_candidate", {}) or {}
        recency = research.get("recency_ensemble", {}) or {}
        hypothesis_id = str(adaptive.get("hypothesis_id") or "").strip()
        if adaptive.get("preregistered") is not True:
            errors.append(
                "experiments.next_research_cycle.adaptive_ensemble.preregistered must be true"
            )
        if not hypothesis_id or str(research.get("primary_hypothesis") or "") != hypothesis_id:
            errors.append(
                "adaptive hypothesis_id must equal next_research_cycle.primary_hypothesis"
            )
        if bool(recency.get("enabled", False)):
            errors.append("static recency research must stay disabled during adaptive research")
        if bool(replacement.get("enabled", False)):
            errors.append("replacement fit must stay disabled until adaptive research passes")
        if str(policy.get("policy") or "") != "validation_lcb_top_k":
            errors.append("adaptive policy must be validation_lcb_top_k")
        if str(control.get("policy") or "") != "equal_recent_k":
            errors.append("adaptive control must be equal_recent_k")
        if int(policy.get("pool_recent_k", 0) or 0) < int(
            policy.get("select_top_k", 0) or 0
        ):
            errors.append("adaptive pool_recent_k must be >= select_top_k")
        if int(policy.get("selector_rows", 0) or 0) <= 0:
            errors.append("adaptive selector_rows must be positive")
        if int(policy.get("purge_rows", -1)) < 0:
            errors.append("adaptive purge_rows must be non-negative")
        if int(policy.get("min_calibration_rows", 0) or 0) < 256:
            errors.append("adaptive min_calibration_rows must be at least 256")
        source_signature = str(
            adaptive.get("source_prediction_signature_hash") or ""
        )
        if len(source_signature) != 64 or "<" in source_signature:
            errors.append("adaptive source prediction signature must be pinned")
        try:
            selection_end = pd.to_datetime(
                adaptive.get("historical_selection_end"), utc=True, errors="raise"
            )
            for window in adaptive.get("excluded_evaluation_windows", []) or []:
                if selection_end >= pd.to_datetime(
                    window.get("start"), utc=True, errors="raise"
                ):
                    errors.append(
                        "adaptive historical selection must end before every excluded OOS window"
                    )
                    break
        except (TypeError, ValueError):
            errors.append("adaptive historical/excluded timestamps must be valid")

    if errors:
        details = "\n- ".join(errors)
        raise ValueError(f"Notebook 04 research contract is invalid:\n- {details}")

    return {
        "research_cycle_status": research_status,
        "research_focus_mode": str(research_focus["mode"]),
        "research_focus_status": research_focus_status,
        "seed_audit_mode": str(seed_audit["mode"]),
        "adaptive_ensemble_enabled": bool(adaptive.get("enabled", False)),
        "adaptive_hypothesis_id": str(adaptive.get("hypothesis_id") or ""),
    }

def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return an in-memory config copy with the requested active feature profile."""

    updated = copy.deepcopy(config)
    _set_cfg(updated, ["features", "active_profile"], profile)
    resolve_feature_profile(updated)
    overrides = _profile_config_overrides(updated, profile)
    if overrides:
        _deep_update(updated, overrides)
    return updated

def _profile_config_overrides(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = _cfg(config, ["features", "profiles"], {}) or {}
    if not isinstance(profiles, dict):
        return {}

    def load(name: str, seen: set[str] | None = None) -> dict[str, Any]:
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f"Cyclic feature profile inheritance detected at {name}")
        seen.add(name)
        current = profiles.get(name)
        if not isinstance(current, dict):
            return {}
        parent_name = current.get("inherit")
        overrides = load(str(parent_name), seen) if parent_name else {}
        current_overrides = current.get("config_overrides", current.get("training_overrides", {})) or {}
        if not isinstance(current_overrides, dict):
            raise ValueError(f"Feature profile config_overrides must be a mapping: {name}")
        return _deep_update(overrides, current_overrides)

    return load(str(profile))

def _profile_rejection_reason(profile: str, experiments: dict[str, Any]) -> str:
    memory = experiments.get("experiment_memory", {}) or {}
    if not bool(memory.get("enabled", False)) or not bool(memory.get("reject_retests", True)):
        return ""
    if profile in {str(item) for item in memory.get("allow_retest_profiles", []) or []}:
        return ""

    rejected_profiles = memory.get("rejected_profiles", {}) or {}
    if isinstance(rejected_profiles, dict) and profile in rejected_profiles:
        value = rejected_profiles[profile]
        if isinstance(value, dict):
            return str(value.get("reason") or "historically_rejected_profile")
        return str(value or "historically_rejected_profile")

    for item in memory.get("rejected_profile_patterns", []) or []:
        if isinstance(item, str):
            pattern = item
            reason = "historically_rejected_profile_pattern"
        elif isinstance(item, dict):
            pattern = str(item.get("pattern", ""))
            reason = str(item.get("reason") or "historically_rejected_profile_pattern")
        else:
            continue
        if pattern and fnmatch(profile, pattern):
            return reason
    return ""

def _filter_memory_rejected_profiles(
    profiles: list[str],
    experiments: dict[str, Any],
    *,
    role: str,
    protected_profiles: set[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    protected = set() if protected_profiles is None else {str(profile) for profile in protected_profiles}
    selected: list[str] = []
    skipped: list[dict[str, str]] = []
    for profile in profiles:
        profile = str(profile)
        if profile in selected:
            continue
        reason = "" if profile in protected else _profile_rejection_reason(profile, experiments)
        if reason:
            skipped.append({"profile": profile, "role": role, "skip_reason": reason})
            continue
        selected.append(profile)
    return selected, skipped

def _policy_status_is_retired_or_failed(status: str) -> bool:
    return any(token in str(status).lower() for token in ("failed", "invalidated", "retired"))

def _future_oos_allowed_benchmark_profiles(config: dict[str, Any], control_profile: str) -> list[str]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    future_items = {str(item) for item in policy_review.get("future_oos_candidates", []) or []}
    profiles_cfg = _cfg(config, ["features", "profiles"], {}) or {}
    allowed = [str(control_profile)]

    for item in future_items:
        if item in profiles_cfg and item not in allowed:
            allowed.append(item)

    for blend in (_cfg(config, ["experiments", "profile_blends", "weighted"], []) or []):
        if not isinstance(blend, dict):
            continue
        name = str(blend.get("name", ""))
        candidate_names = {name, f"blend_{name}"}
        if not candidate_names.intersection(future_items):
            continue
        for profile in blend.get("profiles", []) or []:
            profile = str(profile)
            if profile and profile not in allowed:
                allowed.append(profile)
    return allowed

def _experiment_policy_guard(
    settings: dict[str, Any],
    config: dict[str, Any],
    future_oos_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    holdout = settings.get("holdout", {}) or _cfg(config, ["experiments", "holdout"], {}) or {}
    latest_data_end = _holdout_latest_available_data_end(holdout)
    legacy_monitor_state = _future_oos_monitor_state(config, latest_data_end)
    active_monitor_state = _active_future_oos_state(
        config,
        latest_data_end,
        readiness=future_oos_readiness,
    )
    monitor_state = (
        active_monitor_state
        if active_monitor_state["active_frozen_candidate"]
        else legacy_monitor_state
    )
    control = str(settings.get("control_profile") or _cfg(config, ["experiments", "control_profile"], ""))
    status = str(policy_review.get("status", ""))
    enabled = bool(policy_review.get("enabled", False))
    frozen_cfg = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    primary_candidate_id = str(frozen_cfg.get("primary_candidate_id", ""))
    outcome = (
        _cfg(
            config,
            ["experiments", "frozen_candidate_outcomes", primary_candidate_id],
            {},
        )
        or {}
    )
    failed_future_oos = "failed" in str(outcome.get("status", "")).lower()
    active_candidate = bool(active_monitor_state["active_frozen_candidate"])
    evaluation_completed = bool(monitor_state.get("evaluation_completed", False))
    primary_passed = monitor_state.get("primary_candidate_passed")
    locked = bool(
        not failed_future_oos
        and (
            (active_candidate and not evaluation_completed)
            or (
                enabled
                and _policy_status_is_retired_or_failed(status)
                and legacy_monitor_state["holdout_roll_forward_locked"]
                and not legacy_monitor_state["future_oos_ready"]
            )
        )
    )
    allowed = _future_oos_allowed_benchmark_profiles(config, control)
    if failed_future_oos:
        action = "retire_failed_frozen_candidate_and_open_new_research_anchor"
        reason = "primary_frozen_candidate_failed_future_oos"
    elif active_candidate and not bool(monitor_state.get("candidate_activation_valid", False)):
        action = "pin_replacement_candidate_manifest_and_activate_new_oos_anchor"
        reason = "active_frozen_candidate_manifest_not_activated"
    elif active_candidate and evaluation_completed and primary_passed is True:
        action = "future_oos_candidate_passed_review_phase2_readiness"
        reason = "primary_frozen_candidate_passed_future_oos"
    elif active_candidate and evaluation_completed:
        action = "retire_failed_frozen_candidate_and_open_new_research_anchor"
        reason = "primary_frozen_candidate_failed_future_oos"
    elif active_candidate and monitor_state["future_oos_ready"]:
        if (
            future_oos_readiness is not None
            and not bool(monitor_state.get("ready_for_evaluation", False))
        ):
            action = "repair_future_oos_preflight_without_refit"
            reason = "future_oos_window_available_but_preflight_blocked"
        else:
            action = "run_no_refit_future_oos_evaluator"
            reason = "future_oos_minimum_window_available"
    elif active_candidate:
        action = "wait_for_new_future_oos_rows"
        reason = "active_hash_pinned_candidate_waiting_for_mature_labeled_rows"
    elif locked:
        action = "wait_for_new_unseen_bars_keep_control_profile"
        reason = (
            "clean_holdout_policy_failed_and_future_oos_not_ready; "
            "profile search is locked to control/future-OOS benchmark profiles"
        )
    elif enabled and _policy_status_is_retired_or_failed(status) and monitor_state["future_oos_ready"]:
        action = "future_oos_window_available_review_predefined_candidates"
        reason = "future_oos_minimum_window_available"
    else:
        action = "normal_experiment_flow"
        reason = ""
    return {
        "enabled": enabled,
        "status": status,
        "profile_search_locked": locked,
        "action": action,
        "reason": reason,
        "allowed_benchmark_profiles": allowed,
        "primary_candidate_id": primary_candidate_id,
        **monitor_state,
        "primary_candidate_outcome": str(outcome.get("status", "")),
        "monitor_next_action": str(monitor_state.get("next_action", "")),
        "next_action": action,
    }

def _apply_experiment_policy_guard(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    guard = _experiment_policy_guard(updated, config)
    blocked_candidates: list[str] = []
    blocked_full: list[str] = []
    blocked_seed: list[str] = []

    if guard["profile_search_locked"]:
        control = str(updated["control_profile"])
        allowed = set(str(profile) for profile in guard["allowed_benchmark_profiles"])
        candidates = [str(profile) for profile in updated.get("candidate_profiles", []) or []]
        blocked_candidates = [profile for profile in candidates if profile != control]
        if blocked_candidates:
            updated["candidate_profiles"] = []
            updated["profiles"] = [control]

        filtered_full = []
        for profile in [str(item) for item in updated.get("always_full_profiles", []) or []]:
            if profile in allowed:
                filtered_full.append(profile)
            else:
                blocked_full.append(profile)
        if control not in filtered_full:
            filtered_full.insert(0, control)
        updated["always_full_profiles"] = list(dict.fromkeys(filtered_full))

        seed_audit = copy.deepcopy(updated.get("seed_audit", {}) or {})
        if seed_audit:
            filtered_seed = []
            for profile in [str(item) for item in seed_audit.get("profiles", []) or []]:
                if profile in allowed:
                    filtered_seed.append(profile)
                else:
                    blocked_seed.append(profile)
            if control not in filtered_seed:
                filtered_seed.insert(0, control)
            seed_audit["profiles"] = list(dict.fromkeys(filtered_seed))
            updated["seed_audit"] = seed_audit

        skipped = list(updated.get("skipped_profiles", []) or [])
        for role, profiles in (
            ("candidate_profile", blocked_candidates),
            ("always_full_profile", blocked_full),
            ("seed_audit_profile", blocked_seed),
        ):
            for profile in profiles:
                skipped.append(
                    {
                        "profile": profile,
                        "role": role,
                        "skip_reason": "future_oos_not_ready_profile_search_locked",
                    }
                )
        updated["skipped_profiles"] = skipped

    guard["blocked_candidate_profiles"] = blocked_candidates
    guard["blocked_full_profiles"] = blocked_full
    guard["blocked_seed_profiles"] = blocked_seed
    updated["experiment_policy_guard"] = guard
    return updated

def experiment_settings(config: dict[str, Any]) -> dict[str, Any]:
    experiments = copy.deepcopy(_cfg(config, ["experiments"], {}) or {})
    control = str(experiments.get("control_profile") or _cfg(config, ["features", "active_profile"]))
    raw_candidates = [str(profile) for profile in experiments.get("candidate_profiles", [])]
    candidates, skipped_candidates = _filter_memory_rejected_profiles(
        raw_candidates,
        experiments,
        role="candidate_profile",
        protected_profiles={control},
    )
    profiles = []
    for profile in [control, *candidates]:
        if profile not in profiles:
            profiles.append(profile)
    experiments.setdefault("mode", "staged")
    experiments["control_profile"] = control
    experiments["candidate_profiles"] = [profile for profile in profiles if profile != control]
    experiments["profiles"] = profiles
    experiments.setdefault("triage_fold_ids", [])
    experiments.setdefault("full_cv_profiles", "auto")
    experiments.setdefault("always_full_profiles", [control])
    always_full, skipped_always_full = _filter_memory_rejected_profiles(
        [str(profile) for profile in experiments.get("always_full_profiles", []) or []],
        experiments,
        role="always_full_profile",
        protected_profiles={control},
    )
    if control not in always_full:
        always_full.insert(0, control)
    experiments["always_full_profiles"] = always_full
    experiments.setdefault("max_auto_full_candidates", None)
    experiments.setdefault("resume_existing", True)
    experiments.setdefault("force_retrain", False)
    seed_audit = copy.deepcopy(experiments.get("seed_audit", {}) or {})
    seed_audit.setdefault("enabled", False)
    seed_audit.setdefault("profiles", [control])
    seed_profiles, skipped_seed_profiles = _filter_memory_rejected_profiles(
        [str(profile) for profile in seed_audit.get("profiles", []) or [control]],
        experiments,
        role="seed_audit_profile",
        protected_profiles={control},
    )
    seed_audit["profiles"] = seed_profiles or [control]
    seed_audit.setdefault("seeds", [])
    seed_audit.setdefault("fold_ids", experiments.get("triage_fold_ids", []))
    experiments["seed_audit"] = seed_audit
    experiments["skipped_profiles"] = [*skipped_candidates, *skipped_always_full, *skipped_seed_profiles]
    experiments = _apply_experiment_policy_guard(experiments, config)
    return experiments

def _available_walk_forward_fold_ids(n_rows: int, config: dict[str, Any]) -> list[int]:
    cv_cfg = _cfg(config, ["walk_forward"], {}) or {}
    cv = PurgedWalkForwardCV(
        train_bars=int(cv_cfg["train_bars"]),
        val_bars=int(cv_cfg["val_bars"]),
        test_bars=int(cv_cfg["test_bars"]),
        step_bars=int(cv_cfg["step_bars"]),
        purge_bars=int(cv_cfg["purge_bars"]),
        embargo_bars=int(cv_cfg["embargo_bars"]),
    )
    return [int(fold.fold) for fold in cv.split(int(n_rows))]

def _dedupe_ints(values: list[Any] | tuple[Any, ...]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))

def _evenly_spaced_fold_ids(available_fold_ids: list[int], fold_count: int) -> list[int]:
    available = sorted(dict.fromkeys(int(fold_id) for fold_id in available_fold_ids))
    if not available:
        return []
    count = max(1, int(fold_count))
    if len(available) == 1 or count >= len(available):
        return available
    if count == 1:
        return [available[-1]]

    denominator = count - 1
    last_index = len(available) - 1
    selected_indices: list[int] = []
    for position in range(count):
        index = int(round(position * last_index / denominator))
        if index not in selected_indices:
            selected_indices.append(index)

    if len(selected_indices) < count:
        selected_set = set(selected_indices)
        for index in range(len(available)):
            if index not in selected_set:
                selected_indices.append(index)
                selected_set.add(index)
            if len(selected_indices) >= count:
                break

    return [available[index] for index in sorted(selected_indices[:count])]

def _resolve_seed_audit_fold_ids(
    seed_cfg: dict[str, Any],
    available_fold_ids: list[int],
    *,
    fallback_fold_ids: list[int] | None = None,
) -> list[int]:
    """Resolve seed-audit folds against the current walk-forward universe.

    ``fold_ids: auto`` deliberately re-evaluates the audit folds every run so
    the seed audit keeps covering the first and latest available folds as data
    grows. This prevents stale hardcoded fold ids from producing an incomplete
    temporal-coverage audit after new bars are added.
    """

    available = sorted(dict.fromkeys(int(fold_id) for fold_id in available_fold_ids))
    fallback = (
        sorted(dict.fromkeys(int(fold_id) for fold_id in fallback_fold_ids))
        if fallback_fold_ids
        else []
    )
    raw = seed_cfg.get("fold_ids", None)
    auto_values = {"auto", "balanced", "temporal_coverage"}
    if isinstance(raw, str):
        raw_text = raw.strip().lower()
        if raw_text in {"all", "full"}:
            return available
        if raw_text in auto_values:
            fold_count = int(seed_cfg.get("fold_count", len(fallback) or 8))
            if bool(seed_cfg.get("include_boundary_folds", True)) and len(available) >= 2:
                fold_count = max(2, fold_count)
            return _evenly_spaced_fold_ids(available, fold_count)
        requested = [part.strip() for part in raw.split(",") if part.strip()]
        return _dedupe_ints(requested)

    if raw is None:
        if fallback:
            return fallback
        return available

    requested = list(raw)
    if not requested:
        if fallback:
            return fallback
        return available
    return _dedupe_ints(requested)

def _resolve_seed_audit_fold_ids_for_seed(
    seed_cfg: dict[str, Any],
    available_fold_ids: list[int],
    seed: int,
    *,
    fallback_fold_ids: list[int] | None = None,
) -> list[int]:
    """Resolve an optional seed-specific fold plan without mutating shared config."""

    by_seed = seed_cfg.get("fold_ids_by_seed", {}) or {}
    plan = by_seed.get(str(int(seed)), by_seed.get(int(seed), seed_cfg.get("fold_ids")))
    local = copy.deepcopy(seed_cfg)
    local["fold_ids"] = plan
    return _resolve_seed_audit_fold_ids(
        local,
        available_fold_ids,
        fallback_fold_ids=fallback_fold_ids,
    )

def _attach_resolved_seed_audit_folds(
    settings: dict[str, Any],
    available_fold_ids: list[int],
    *,
    fallback_fold_ids: list[int] | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    seed_cfg = updated.get("seed_audit", {}) or {}
    if not bool(seed_cfg.get("enabled", False)):
        return updated
    updated["seed_audit"] = copy.deepcopy(seed_cfg)
    resolved_by_seed = {
        str(int(seed)): _resolve_seed_audit_fold_ids_for_seed(
            seed_cfg,
            available_fold_ids,
            int(seed),
            fallback_fold_ids=fallback_fold_ids,
        )
        for seed in seed_cfg.get("seeds", []) or []
    }
    updated["seed_audit"]["resolved_fold_ids_by_seed"] = resolved_by_seed
    updated["seed_audit"]["resolved_fold_ids"] = sorted(
        {
            int(fold_id)
            for fold_ids in resolved_by_seed.values()
            for fold_id in fold_ids
        }
    )
    return updated

def _diagnostic_seed_audit_settings(
    settings: dict[str, Any],
    diagnostic_config: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    run_seed_cfg = settings.get("seed_audit", {}) or {}
    seed_cfg = copy.deepcopy(
        _cfg(diagnostic_config, ["experiments", "seed_audit"], default={}) or {}
    )
    if run_seed_cfg.get("resolved_fold_ids"):
        seed_cfg["resolved_fold_ids"] = list(
            dict.fromkeys(int(fold_id) for fold_id in run_seed_cfg.get("resolved_fold_ids", []))
        )
    if run_seed_cfg.get("resolved_fold_ids_by_seed"):
        seed_cfg["resolved_fold_ids_by_seed"] = {
            str(seed): list(dict.fromkeys(int(fold_id) for fold_id in fold_ids))
            for seed, fold_ids in (
                run_seed_cfg.get("resolved_fold_ids_by_seed", {}) or {}
            ).items()
        }
    updated["seed_audit"] = seed_cfg
    return updated

def _validate_requested_fold_ids(
    *,
    plan_name: str,
    requested_fold_ids: list[int] | None,
    available_fold_ids: list[int],
) -> None:
    if requested_fold_ids is None:
        return
    requested = list(dict.fromkeys(int(fold_id) for fold_id in requested_fold_ids))
    available = set(int(fold_id) for fold_id in available_fold_ids)
    invalid = [fold_id for fold_id in requested if fold_id not in available]
    if invalid:
        available_range = (
            f"{min(available)}..{max(available)}" if available else "none"
        )
        raise ValueError(
            f"{plan_name} contains unavailable fold ids {invalid}. "
            f"Available walk-forward folds are {available_range} ({len(available)} folds). "
            "Update config.yaml before training; fold ids are never silently skipped."
        )

def _preflight_fold_plans(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    config: dict[str, Any],
) -> list[int]:
    available_fold_ids = _available_walk_forward_fold_ids(len(frame), config)
    triage_fold_ids = [int(fold_id) for fold_id in settings.get("triage_fold_ids", [])]
    _validate_requested_fold_ids(
        plan_name="experiments.triage_fold_ids",
        requested_fold_ids=triage_fold_ids or None,
        available_fold_ids=available_fold_ids,
    )
    seed_cfg = settings.get("seed_audit", {}) or {}
    if bool(seed_cfg.get("enabled", False)):
        seeds = [int(seed) for seed in seed_cfg.get("seeds", []) or []]
        plans = {
            str(seed): _resolve_seed_audit_fold_ids_for_seed(
                seed_cfg,
                available_fold_ids,
                seed,
                fallback_fold_ids=triage_fold_ids or None,
            )
            for seed in seeds
        } or {
            "default": _resolve_seed_audit_fold_ids(
                seed_cfg,
                available_fold_ids,
                fallback_fold_ids=triage_fold_ids or None,
            )
        }
        for seed, seed_fold_ids in plans.items():
            _validate_requested_fold_ids(
                plan_name=f"experiments.seed_audit.fold_ids[{seed}]",
                requested_fold_ids=seed_fold_ids or None,
                available_fold_ids=available_fold_ids,
            )
    return available_fold_ids

def _profile_requires_intrahour_features(config: dict[str, Any], profile: str) -> bool:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    return any("ih15" in str(pattern) for pattern in resolved.get("include_patterns", []) or [])

def _profile_requires_futures_context_features(config: dict[str, Any], profile: str) -> bool:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    return any("fut_" in str(pattern) for pattern in resolved.get("include_patterns", []) or [])

def _missing_intrahour_include_patterns(config: dict[str, Any], profile: str, feature_columns: tuple[str, ...]) -> list[str]:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    patterns = [str(pattern) for pattern in resolved.get("include_patterns", []) or [] if "ih15_" in str(pattern)]
    return [
        pattern
        for pattern in patterns
        if not any(fnmatch(column, pattern) for column in feature_columns)
    ]

def _preflight_experiment_profiles(
    settings: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Skip candidate profiles that cannot change the current feature matrix."""

    updated = copy.deepcopy(settings)
    base_columns = select_feature_columns(frame)
    control = str(updated["control_profile"])
    selected_profiles: list[str] = []
    skipped_profiles = list(updated.get("skipped_profiles", []) or [])
    seen_signatures: dict[tuple[str, ...], str] = {}
    profile_feature_columns: dict[str, tuple[str, ...]] = {}

    for profile in [str(item) for item in updated.get("profiles", [])]:
        cfg = profile_config(config, profile)
        feature_columns = tuple(filter_feature_columns(base_columns, cfg))
        profile_feature_columns[profile] = feature_columns
        has_intrahour = any(column.startswith("ih15_") for column in feature_columns)
        has_futures_context = any(column.startswith("fut_") for column in feature_columns)
        missing_intrahour_patterns = _missing_intrahour_include_patterns(config, profile, feature_columns)
        if profile != control and _profile_requires_intrahour_features(config, profile) and (
            not has_intrahour or missing_intrahour_patterns
        ):
            reason = "missing_intrahour_features_rerun_01_02_03"
            if missing_intrahour_patterns:
                reason = f"{reason}:{','.join(missing_intrahour_patterns[:6])}"
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": reason,
                }
            )
            continue
        if profile != control and _profile_requires_futures_context_features(config, profile) and not has_futures_context:
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": "missing_futures_context_features_rerun_01_02_03",
                }
            )
            continue
        duplicate_of = seen_signatures.get(feature_columns)
        has_config_overrides = bool(_profile_config_overrides(config, profile))
        if profile != control and duplicate_of and not has_config_overrides:
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": f"duplicate_feature_signature:{duplicate_of}",
                }
            )
            continue
        selected_profiles.append(profile)
        seen_signatures[feature_columns] = profile

    if control not in selected_profiles:
        raise ValueError(f"Control profile was removed during experiment preflight: {control}")

    selected_set = set(selected_profiles)
    updated["profiles"] = selected_profiles
    updated["candidate_profiles"] = [profile for profile in selected_profiles if profile != control]

    def profile_is_runnable(profile: str, role: str) -> bool:
        if profile == control or profile in selected_set:
            return True
        if profile not in profile_feature_columns:
            cfg = profile_config(config, profile)
            profile_feature_columns[profile] = tuple(filter_feature_columns(base_columns, cfg))
        feature_columns = profile_feature_columns[profile]
        if _profile_requires_intrahour_features(config, profile) and not any(
            column.startswith("ih15_") for column in feature_columns
        ):
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": role,
                    "skip_reason": "missing_intrahour_features_rerun_01_02_03",
                }
            )
            return False
        if _profile_requires_futures_context_features(config, profile) and not any(
            column.startswith("fut_") for column in feature_columns
        ):
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": role,
                    "skip_reason": "missing_futures_context_features_rerun_01_02_03",
                }
            )
            return False
        return True

    updated["always_full_profiles"] = [
        str(profile)
        for profile in updated.get("always_full_profiles", []) or []
        if profile_is_runnable(str(profile), "always_full_profile")
    ]
    seed_audit = copy.deepcopy(updated.get("seed_audit", {}) or {})
    if seed_audit:
        seed_audit["profiles"] = [
            str(profile)
            for profile in seed_audit.get("profiles", []) or []
            if profile_is_runnable(str(profile), "seed_audit_profile")
        ]
        updated["seed_audit"] = seed_audit
    updated["skipped_profiles"] = skipped_profiles
    return updated

def experiment_root(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "experiments"

def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _training_config_payload(config: dict[str, Any], *, profile: str | None = None) -> dict[str, Any]:
    """Return only settings that can change fitted artifacts or predictions."""

    cfg = profile_config(config, profile) if profile else config
    project = _cfg(cfg, ["project"], {}) or {}
    features = _cfg(cfg, ["features"], {}) or {}
    active_profile = str(profile or features.get("active_profile", ""))
    resolved_profile = resolve_feature_profile(cfg) if active_profile else {}
    return {
        "project": {
            "random_seed": project.get("random_seed"),
            "deterministic": project.get("deterministic"),
        },
        "feature_profile": active_profile,
        "resolved_feature_profile": resolved_profile,
        "feature_generation_policy": {
            key: copy.deepcopy(features.get(key))
            for key in (
                "exclude_columns",
                "exclude_patterns",
                "stationarity",
            )
            if key in features
        },
        "model": copy.deepcopy(_cfg(cfg, ["model"], {}) or {}),
        "training": copy.deepcopy(_cfg(cfg, ["training"], {}) or {}),
        "walk_forward": copy.deepcopy(_cfg(cfg, ["walk_forward"], {}) or {}),
        "hmm": copy.deepcopy(_cfg(cfg, ["hmm"], {}) or {}),
        "labeling": copy.deepcopy(_cfg(cfg, ["labeling"], {}) or {}),
    }

def _training_settings_payload(settings: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: copy.deepcopy(settings.get(key))
        for key in (
            "mode",
            "control_profile",
            "profiles",
            "candidate_profiles",
            "triage_fold_ids",
            "full_cv_profiles",
            "always_full_profiles",
            "max_auto_full_candidates",
            "seed_audit",
        )
        if key in settings
    }
    holdout = settings.get("holdout", {}) or {}
    payload["holdout_training_boundary"] = {
        key: copy.deepcopy(holdout.get(key))
        for key in (
            "enabled",
            "holdout_bars",
            "selection_rows",
            "selection_data_start",
            "selection_data_end",
            "holdout_data_start",
        )
        if key in holdout
    }
    return payload

def _training_code_signature_payload(*, include_files: bool = True) -> dict[str, Any]:
    return training_code_signature_payload(include_files=include_files)

def _experiment_signature(
    config: dict[str, Any],
    settings: dict[str, Any],
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    profiles = list(
        dict.fromkeys(
            [
                str(settings.get("control_profile", "")),
                *[str(item) for item in settings.get("profiles", []) or []],
                *[str(item) for item in settings.get("always_full_profiles", []) or []],
                *[
                    str(item)
                    for item in (settings.get("seed_audit", {}) or {}).get("profiles", []) or []
                ],
            ]
        )
    )
    profiles = [profile for profile in profiles if profile]
    payload = {
        "signature_version": "training_v3",
        "settings": _training_settings_payload(settings),
        "training_code_signature": _training_code_signature_payload(include_files=False),
        "profiles": {
            profile: _training_config_payload(config, profile=profile)
            for profile in profiles
        },
    }
    if frame is not None:
        payload["training_frame"] = {
            "rows": int(len(frame)),
            **_frame_window(frame),
            "fingerprint": _frame_fingerprint(frame),
        }
    return payload

def _diagnostic_config_for_run(
    config: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Combine immutable run settings with current diagnostics policy.

    Profile/fold settings describe the historical fitted artifacts and remain
    sourced from the run manifest. Lifecycle, governance, experiment memory,
    and seed-audit policy are diagnostics-time state and must come from the
    current config so retired candidates cannot be resurrected by old runs.
    """

    diagnostic_config = copy.deepcopy(config)
    current_experiments = copy.deepcopy(
        _cfg(diagnostic_config, ["experiments"], default={}) or {}
    )
    merged_experiments = copy.deepcopy(current_experiments)
    merged_experiments.update(copy.deepcopy(settings))
    for key in _CURRENT_DIAGNOSTIC_EXPERIMENT_KEYS:
        if key in current_experiments:
            merged_experiments[key] = copy.deepcopy(current_experiments[key])
        else:
            merged_experiments.pop(key, None)
    _set_cfg(diagnostic_config, ["experiments"], merged_experiments)
    return diagnostic_config

def _diagnostics_signature(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Track report-policy changes without invalidating fitted artifacts."""

    return {
        "signature_version": "diagnostics_v1",
        "validation": copy.deepcopy(_cfg(config, ["validation"], {}) or {}),
        "policy_review": copy.deepcopy(_cfg(config, ["experiments", "policy_review"], {}) or {}),
        "frozen_candidates": copy.deepcopy(_cfg(config, ["experiments", "frozen_candidates"], {}) or {}),
        "future_oos_validation": copy.deepcopy(
            _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
        ),
        "frozen_candidate_outcomes": copy.deepcopy(
            _cfg(config, ["experiments", "frozen_candidate_outcomes"], {}) or {}
        ),
        "next_research_cycle": copy.deepcopy(
            _cfg(config, ["experiments", "next_research_cycle"], {}) or {}
        ),
        "profile_blends": copy.deepcopy(_cfg(config, ["experiments", "profile_blends"], {}) or {}),
        "control_profile": str(settings.get("control_profile", "")),
    }

def _matching_latest_run(checkpoint_dir: str | Path, signature_hash: str) -> Path | None:
    root = experiment_root(checkpoint_dir)
    if not root.exists():
        return None
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name, reverse=True)
    for run in runs:
        manifest_path = run / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
        except json.JSONDecodeError:
            continue
        if manifest.get("signature_hash") == signature_hash:
            return run
    return None

def resolve_experiment_run_id(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    settings: dict[str, Any] | None = None,
    run_id: str | None = None,
    frame: pd.DataFrame | None = None,
) -> tuple[str, str]:
    settings = experiment_settings(config) if settings is None else settings
    if run_id:
        return str(run_id), "explicit_argument"
    if settings.get("run_id"):
        return str(settings["run_id"]), "config"
    signature_hash = _hash_payload(_experiment_signature(config, settings, frame))
    if bool(settings.get("resume_existing", True)) and not bool(settings.get("force_retrain", False)):
        existing = _matching_latest_run(checkpoint_dir, signature_hash)
        if existing is not None:
            return existing.name, "matching_existing"
    return new_run_id(), "new"

def latest_experiment_run(checkpoint_dir: str | Path) -> Path:
    root = experiment_root(checkpoint_dir)
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name)
    if not runs:
        raise FileNotFoundError(f"No experiment runs found under {root}")
    return runs[-1]

def profile_run_dir(checkpoint_dir: str | Path, run_id: str, profile: str) -> Path:
    return experiment_root(checkpoint_dir) / run_id / _slug(profile)

def _frame_window(frame: pd.DataFrame) -> dict[str, str]:
    if "timestamp" not in frame.columns or frame.empty:
        return {"data_start": "", "data_end": ""}
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return {"data_start": str(timestamps.min()), "data_end": str(timestamps.max())}

def _frame_fingerprint(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty-frame").hexdigest()
    selected = list(frame.columns) if columns is None else [
        column for column in columns if column in frame.columns
    ]
    selected = sorted(dict.fromkeys(selected))
    if not selected:
        return hashlib.sha256(b"no-columns").hexdigest()
    hashed = pd.util.hash_pandas_object(
        frame[selected],
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256()
    digest.update("|".join(selected).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()

def _training_signature(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    profile: str,
    feature_columns: list[str],
    fold_ids: list[int] | None,
    fold_scope: str,
) -> dict[str, Any]:
    return {
        "signature_version": "profile_training_v3",
        "profile": profile,
        "fold_scope": fold_scope,
        "fold_ids": fold_ids,
        "training_code_signature": _training_code_signature_payload(include_files=True),
        "feature_columns": feature_columns,
        "feature_columns_hash": _hash_payload(feature_columns),
        "training_config_hash": _hash_payload(
            _training_config_payload(config, profile=profile)
        ),
        "frame_rows": int(len(frame)),
        "frame_fingerprint": _frame_fingerprint(
            frame,
            columns=[
                "timestamp",
                *feature_columns,
                "label",
                "fwd_return_10h",
                "forward_return",
                "tb_return",
            ],
        ),
        **_frame_window(frame),
    }

def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "training_manifest.json"

def _training_execution_summary_path(run_dir: Path) -> Path:
    return run_dir / "training_execution_summary.json"

def _training_execution_summary(
    *,
    run_id: str,
    run_id_source: str | None,
    executed_results: list[dict[str, Any]],
    skipped_results: list[dict[str, Any]],
    profile_results: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_id_source": run_id_source,
        "training_executed_count": int(len(executed_results)),
        "training_skipped_count": int(len(skipped_results)),
        "all_training_scopes_reused": bool(profile_results or seed_results) and len(executed_results) == 0,
        "reused_training_scopes": [
            {"profile": str(result["profile"]), "fold_scope": str(result["fold_scope"])}
            for result in skipped_results
        ],
        "executed_training_scopes": [
            {"profile": str(result["profile"]), "fold_scope": str(result["fold_scope"])}
            for result in executed_results
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

def _load_training_execution_summary(run_dir: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    summary_path = _training_execution_summary_path(run_dir)
    if summary_path.exists():
        summary = _read_json(summary_path)
        summary["training_execution_metadata_source"] = "training_execution_summary"
        summary["training_execution_metadata_available"] = True
        return summary

    decision_path = run_dir / "decision_report.json"
    if decision_path.exists():
        prior_decision = _read_json(decision_path)
        if any(key in prior_decision for key in _TRAINING_EXECUTION_KEYS):
            summary = {key: prior_decision.get(key) for key in _TRAINING_EXECUTION_KEYS if key in prior_decision}
            summary["run_id"] = str(prior_decision.get("run_id") or run_dir.name)
            summary["training_execution_metadata_source"] = "prior_decision_report"
            summary["training_execution_metadata_available"] = any(
                key in summary for key in ("training_executed_count", "training_skipped_count")
            )
            return summary

    summary = {
        "run_id": str(run_manifest.get("run_id") or run_dir.name),
        "run_id_source": run_manifest.get("run_id_source"),
        "training_executed_count": None,
        "training_skipped_count": None,
        "all_training_scopes_reused": None,
        "reused_training_scopes": [],
        "executed_training_scopes": [],
        "training_execution_metadata_source": "run_manifest_only",
        "training_execution_metadata_available": False,
    }
    return summary

def _is_complete(output_dir: Path, expected_signature_hash: str) -> bool:
    manifest_path = _manifest_path(output_dir)
    predictions_path = output_dir / "predictions_all.parquet"
    if not manifest_path.exists() or not predictions_path.exists():
        return False
    manifest = _read_json(manifest_path)
    return bool(manifest.get("completed")) and manifest.get("signature_hash") == expected_signature_hash

def _experiment_selection_frame(settings: dict[str, Any]) -> pd.DataFrame:
    columns = ["profile", "role", "selected", "expected_fold_scope", "skip_reason"]
    rows: list[dict[str, Any]] = [
        {
            "profile": str(settings["control_profile"]),
            "role": "control_profile",
            "selected": True,
            "expected_fold_scope": "triage",
            "skip_reason": "",
        }
    ]
    for role, key in (
        ("candidate_profile", "candidate_profiles"),
        ("always_full_profile", "always_full_profiles"),
    ):
        values = settings.get(key, [])
        expected_scope = {
            "candidate_profile": "triage",
            "always_full_profile": "full",
        }[role]
        for profile in values or []:
            rows.append(
                {
                    "profile": str(profile),
                    "role": role,
                    "selected": True,
                    "expected_fold_scope": expected_scope,
                    "skip_reason": "",
                }
            )
    seed_audit = settings.get("seed_audit", {}) or {}
    seed_enabled = bool(seed_audit.get("enabled", False))
    for profile in seed_audit.get("profiles", []) or []:
        rows.append(
            {
                "profile": str(profile),
                "role": "seed_audit_profile",
                "selected": seed_enabled,
                "expected_fold_scope": "seed_audit" if seed_enabled else "",
                "skip_reason": "" if seed_enabled else "seed_audit_disabled_not_evaluated",
            }
        )
    for skipped in settings.get("skipped_profiles", []) or []:
        rows.append(
            {
                "profile": str(skipped.get("profile", "")),
                "role": str(skipped.get("role", "skipped_profile")),
                "selected": False,
                "expected_fold_scope": "",
                "skip_reason": str(skipped.get("skip_reason", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)

def _experiment_selection_markdown(selection: pd.DataFrame) -> str:
    lines = ["# Experiment Selection", ""]
    if selection.empty:
        lines.append("No profile selection metadata was produced.")
        return "\n".join(lines)
    lines.append("| profile | role | selected | expected_fold_scope | skip_reason |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in selection.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(bool(row["selected"])),
                    str(row.get("expected_fold_scope", "")),
                    str(row.get("skip_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)

def _write_experiment_selection(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    selection = _experiment_selection_frame(settings)
    selection.to_csv(path / "experiment_selection.csv", index=False)
    (path / "experiment_selection.md").write_text(_experiment_selection_markdown(selection), encoding="utf-8")
    _write_json(
        path / "experiment_selection.json",
        {
            "control_profile": settings.get("control_profile", ""),
            "selected_profiles": selection.loc[selection["selected"].astype(bool), "profile"].drop_duplicates().tolist(),
            "skipped_profiles": settings.get("skipped_profiles", []) or [],
            "rows": selection.to_dict(orient="records"),
        },
    )
    return selection

def _missing_selected_profiles(selection: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    columns = ["profile", "role", "expected_fold_scope", "reason"]
    if selection.empty:
        return pd.DataFrame(columns=columns)
    completed = {
        (str(row["profile"]), str(row["fold_scope"]))
        for _, row in comparison.iterrows()
        if str(row.get("fold_scope", "")) in {"triage", "full"}
    }
    rows = []
    comparable_scopes = {"triage", "full"}
    for _, row in selection.iterrows():
        if not bool(row.get("selected", False)):
            continue
        scope = str(row.get("expected_fold_scope", ""))
        if scope not in comparable_scopes:
            continue
        profile = str(row.get("profile", ""))
        if (profile, scope) in completed:
            continue
        rows.append(
            {
                "profile": profile,
                "role": str(row.get("role", "")),
                "expected_fold_scope": scope,
                "reason": "missing_selected_profile_output",
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _missing_selected_markdown(missing: pd.DataFrame) -> str:
    lines = ["# Missing Selected Profiles", ""]
    if missing.empty:
        lines.append("All selected comparison profiles have completed outputs.")
        return "\n".join(lines)
    lines.append("| profile | role | expected_fold_scope | reason |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in missing.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(row["expected_fold_scope"]),
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)

def _write_missing_selected_profiles(path: Path, missing: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "missing_selected_profiles.md").write_text(_missing_selected_markdown(missing), encoding="utf-8")
    _write_json(path / "missing_selected_profiles.json", {"rows": missing.to_dict(orient="records")})

def _holdout_latest_available_data_end(holdout: dict[str, Any]) -> str:
    """Return the latest labeled-data timestamp, not the frozen holdout end.

    A failed clean holdout can freeze `holdout_data_end` at the anchor while
    fresher rows accumulate outside the frozen window. Future-OOS monitoring
    must count those fresher rows without allowing the holdout to roll forward.
    """

    for key in ("latest_available_data_end", "latest_data_end", "data_end", "holdout_data_end"):
        value = str(holdout.get(key, "") or "")
        if value:
            return value
    return ""

def _active_future_oos_state(
    config: dict[str, Any],
    latest_data_end: Any,
    *,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return live state for the active frozen Future-OOS candidate."""

    frozen = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    future = _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
    primary_id = str(frozen.get("primary_candidate_id", "") or "")
    spec = next(
        (
            item
            for item in frozen.get("candidates", []) or []
            if isinstance(item, dict)
            and str(item.get("candidate_id", "") or "") == primary_id
        ),
        {},
    )
    configured_status = str(
        spec.get("status")
        or spec.get("manifest_candidate_status")
        or ""
    )
    status_lower = configured_status.lower()
    anchor_data_end = str(
        spec.get("anchor_data_end")
        or frozen.get("anchor_data_end")
        or ""
    )
    inactive_status = any(
        token in status_lower
        for token in ("retired", "failed", "rejected", "disabled", "invalidated")
    )
    active_candidate = bool(
        frozen.get("enabled", False)
        and future.get("enabled", False)
        and primary_id
        and spec
        and anchor_data_end
        and not inactive_status
    )
    expected_hash = str(spec.get("expected_manifest_hash", "") or "")
    pin_pending = any(
        token in status_lower
        for token in ("pin_pending", "pin_required", "awaiting_manifest")
    )
    candidate_activation_valid = bool(
        active_candidate
        and not pin_pending
        and (
            not status_lower.startswith("manifest_pinned")
            or bool(expected_hash)
        )
    )
    min_rows = int(future.get("min_rows", 720) or 0)
    preferred_rows = int(future.get("preferred_rows", 2160) or 0)
    label_maturity_horizon = int(
        _cfg(config, ["labeling", "max_holding_bars"], 0) or 0
    )
    latest_text = str(latest_data_end or "")
    actual_readiness = readiness is not None
    if readiness is not None:
        latest_text = str(
            readiness.get("latest_available_data_end")
            or latest_text
            or ""
        )
        observed_rows = int(readiness.get("new_labeled_rows", 0) or 0)
        evaluation_completed = bool(readiness.get("evaluation_completed", False))
        primary_passed = readiness.get("primary_candidate_passed")
        ready_for_evaluation = bool(readiness.get("ready_for_evaluation", False))
    else:
        observed_rows = 0
        if anchor_data_end and latest_text:
            try:
                anchor_ts = pd.to_datetime(anchor_data_end, utc=True)
                latest_ts = pd.to_datetime(latest_text, utc=True)
                if pd.notna(anchor_ts) and pd.notna(latest_ts) and latest_ts > anchor_ts:
                    observed_rows = int(
                        (latest_ts - anchor_ts).total_seconds() // 3600
                    )
            except (TypeError, ValueError):
                observed_rows = 0
        evaluation_completed = False
        primary_passed = None
        ready_for_evaluation = False
    estimated_window_data_available = bool(
        min_rows > 0 and observed_rows >= min_rows
    )
    estimated_preferred_window_data_available = bool(
        preferred_rows > 0 and observed_rows >= preferred_rows
    )
    window_data_available = bool(
        actual_readiness and estimated_window_data_available
    )
    preferred_window_data_available = bool(
        actual_readiness and estimated_preferred_window_data_available
    )
    return {
        "monitor_enabled": active_candidate,
        "active_frozen_candidate": active_candidate,
        "candidate_activation_valid": candidate_activation_valid,
        "configured_candidate_status": configured_status,
        "expected_manifest_hash": expected_hash,
        "state_source": (
            "active_frozen_candidate"
            if active_candidate
            else "no_active_frozen_candidate"
        ),
        "readiness_basis": (
            "actual_mature_labeled_rows"
            if actual_readiness
            else "elapsed_hours_estimate"
        ),
        "anchor_run_id": str(frozen.get("anchor_run_id", "") or ""),
        "anchor_data_end": anchor_data_end,
        "latest_available_data_end": latest_text,
        "new_bars_since_anchor": observed_rows,
        "min_new_bars": min_rows,
        "preferred_new_bars": preferred_rows,
        "label_maturity_horizon_bars": label_maturity_horizon,
        "min_new_bars_remaining": max(0, min_rows - observed_rows),
        "preferred_new_bars_remaining": max(0, preferred_rows - observed_rows),
        "future_oos_ready": bool(active_candidate and window_data_available),
        "future_oos_preferred_ready": bool(
            active_candidate and preferred_window_data_available
        ),
        "window_data_available": window_data_available,
        "preferred_window_data_available": preferred_window_data_available,
        "estimated_window_data_available": estimated_window_data_available,
        "estimated_preferred_window_data_available": (
            estimated_preferred_window_data_available
        ),
        "ready_for_evaluation": ready_for_evaluation,
        "evaluation_completed": evaluation_completed,
        "primary_candidate_passed": primary_passed,
        "allow_holdout_roll_forward": False,
        "holdout_roll_forward_locked": bool(active_candidate),
        "next_action": (
            "run_no_refit_future_oos_evaluator"
            if window_data_available
            else "wait_for_new_future_oos_rows"
        ),
    }


def _future_oos_monitor_state(config: dict[str, Any], latest_data_end: Any) -> dict[str, Any]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    monitor = policy_review.get("future_oos_monitor", {}) or {}
    status = str(policy_review.get("status", "")).lower()
    anchor_data_end = str(monitor.get("anchor_data_end", "") or "")
    latest_text = str(latest_data_end or "")
    new_bars_since_anchor = 0
    if anchor_data_end and latest_text:
        try:
            anchor_ts = pd.to_datetime(anchor_data_end, utc=True)
            latest_ts = pd.to_datetime(latest_text, utc=True)
            if pd.notna(anchor_ts) and pd.notna(latest_ts) and latest_ts > anchor_ts:
                new_bars_since_anchor = int((latest_ts - anchor_ts).total_seconds() // 3600)
        except (TypeError, ValueError):
            new_bars_since_anchor = 0

    min_new_bars = int(monitor.get("min_new_bars", 0) or 0)
    preferred_new_bars = int(monitor.get("preferred_new_bars", 0) or 0)
    monitor_enabled = bool(monitor.get("enabled", False))
    window_data_available = bool(
        min_new_bars > 0 and new_bars_since_anchor >= min_new_bars
    )
    preferred_window_data_available = bool(
        preferred_new_bars > 0 and new_bars_since_anchor >= preferred_new_bars
    )
    future_oos_ready = bool(monitor_enabled and window_data_available)
    future_oos_preferred_ready = bool(
        monitor_enabled and preferred_window_data_available
    )
    allow_roll_forward = bool(monitor.get("allow_holdout_roll_forward", False))
    retired_or_failed = any(token in status for token in ("failed", "invalidated", "retired"))
    lock_active = monitor_enabled and bool(anchor_data_end) and retired_or_failed and not allow_roll_forward
    if not monitor_enabled:
        next_action = "monitor_disabled"
    elif future_oos_ready:
        next_action = "future_oos_window_available"
    else:
        next_action = "wait_for_new_unseen_bars"

    return {
        "monitor_enabled": monitor_enabled,
        "anchor_run_id": str(monitor.get("anchor_run_id", "")),
        "anchor_data_end": anchor_data_end,
        "latest_available_data_end": latest_text,
        "new_bars_since_anchor": new_bars_since_anchor,
        "min_new_bars": min_new_bars,
        "preferred_new_bars": preferred_new_bars,
        "min_new_bars_remaining": max(0, min_new_bars - new_bars_since_anchor),
        "preferred_new_bars_remaining": max(0, preferred_new_bars - new_bars_since_anchor),
        "future_oos_ready": future_oos_ready,
        "future_oos_preferred_ready": future_oos_preferred_ready,
        "window_data_available": window_data_available,
        "preferred_window_data_available": preferred_window_data_available,
        "allow_holdout_roll_forward": allow_roll_forward,
        "holdout_roll_forward_locked": lock_active,
        "next_action": next_action,
    }

def _future_oos_ready_at_fields(monitor_state: dict[str, Any]) -> dict[str, str]:
    anchor = str(monitor_state.get("anchor_data_end", "") or "")
    fields = {
        "min_ready_at": "",
        "preferred_ready_at": "",
        "min_raw_data_ready_at": "",
        "preferred_raw_data_ready_at": "",
    }
    if not anchor:
        return fields
    try:
        anchor_ts = pd.to_datetime(anchor, utc=True)
    except (TypeError, ValueError):
        return fields
    if pd.isna(anchor_ts):
        return fields
    min_new_bars = int(monitor_state.get("min_new_bars", 0) or 0)
    preferred_new_bars = int(monitor_state.get("preferred_new_bars", 0) or 0)
    maturity_horizon = int(
        monitor_state.get("label_maturity_horizon_bars", 0) or 0
    )
    if min_new_bars > 0:
        fields["min_ready_at"] = str(anchor_ts + pd.Timedelta(hours=min_new_bars))
        fields["min_raw_data_ready_at"] = str(
            anchor_ts + pd.Timedelta(hours=min_new_bars + maturity_horizon)
        )
    if preferred_new_bars > 0:
        fields["preferred_ready_at"] = str(anchor_ts + pd.Timedelta(hours=preferred_new_bars))
        fields["preferred_raw_data_ready_at"] = str(
            anchor_ts
            + pd.Timedelta(hours=preferred_new_bars + maturity_horizon)
        )
    return fields
