"""Cohesive experiment services for the Phase 1 research workflow.

The package facade intentionally resolves public objects lazily. Lightweight
Phase 2 tooling imports artifact helpers from this package, and that path must
not eagerly import the full training stack or optional research dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "experiment_settings": ("configuration", "experiment_settings"),
    "latest_experiment_run": ("configuration", "latest_experiment_run"),
    "profile_config": ("configuration", "profile_config"),
    "profile_run_dir": ("configuration", "profile_run_dir"),
    "resolve_experiment_run_id": ("configuration", "resolve_experiment_run_id"),
    "validate_training_research_contract": (
        "configuration",
        "validate_training_research_contract",
    ),
    "prepare_training_holdout_split": ("holdout", "prepare_training_holdout_split"),
    "freeze_candidate_manifests": ("frozen", "freeze_candidate_manifests"),
    "evaluate_future_oos": ("future_oos", "evaluate_future_oos"),
    "future_oos_preflight": ("oos_preflight", "future_oos_preflight"),
    "run_profile_experiment": ("training", "run_profile_experiment"),
    "run_seed_audit_extension": ("seed_audit", "run_seed_audit_extension"),
    "run_experiment_matrix": ("orchestration", "run_experiment_matrix"),
    "write_experiment_diagnostics": ("orchestration", "write_experiment_diagnostics"),
    "run_recency_ensemble_research": (
        "rolling_research",
        "run_recency_ensemble_research",
    ),
    "run_cached_adaptive_ensemble_research": (
        "cached_policy_research",
        "run_cached_adaptive_ensemble_research",
    ),
    "publish_replacement_candidate_reports": (
        "replacement",
        "publish_replacement_candidate_reports",
    ),
    "run_replacement_candidate_fit": ("replacement", "run_replacement_candidate_fit"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'yenibot.experiment' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"yenibot.experiment.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
