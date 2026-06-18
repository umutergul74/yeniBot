"""Append an isolated seed audit to an existing completed experiment run."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _set_cfg, _write_json
from yenibot.experiment.configuration import (
    _preflight_fold_plans,
    _resolve_seed_audit_fold_ids,
    experiment_root,
    experiment_settings,
)
from yenibot.experiment.ensembles import (
    _seed_audit_coverage_frame,
    _seed_audit_entries_to_frames,
    _seed_audit_scope,
    _seed_ensemble_entries,
    _seed_ensemble_frame,
    _write_seed_audit_files,
    _write_seed_ensemble_files,
)
from yenibot.experiment.training import run_profile_experiment

__all__ = ["run_seed_audit_extension"]


def run_seed_audit_extension(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    source_run_id: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Train only configured seed/fold scopes inside a completed source run.

    The source run's full-CV artifacts and training manifest are left
    untouched. Seed scopes have their own fitted-artifact signatures, and the
    extension writes a separate append-only summary at the run root.
    """

    settings = experiment_settings(config)
    seed_cfg = copy.deepcopy(settings.get("seed_audit", {}) or {})
    if not bool(seed_cfg.get("enabled", False)):
        raise ValueError("Seed-audit extension requires experiments.seed_audit.enabled=true")
    if str(seed_cfg.get("mode", "")) != "extend_existing_run":
        raise ValueError(
            "Seed-audit extension requires experiments.seed_audit.mode=extend_existing_run"
        )

    run_dir = experiment_root(checkpoint_dir) / str(source_run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Seed-audit source run does not exist: {run_dir}")

    profiles = [
        str(profile)
        for profile in seed_cfg.get("profiles", []) or [settings["control_profile"]]
    ]
    for profile in profiles:
        full_scope = run_dir / profile / "full"
        if not (full_scope / "training_manifest.json").exists() or not (
            full_scope / "predictions_all.parquet"
        ).exists():
            raise FileNotFoundError(
                "Seed-audit extension requires a completed full-CV source scope: "
                f"{full_scope}"
            )

    available_fold_ids = _preflight_fold_plans(frame, settings, config)
    fold_ids = _resolve_seed_audit_fold_ids(
        seed_cfg,
        available_fold_ids,
        fallback_fold_ids=[int(fold_id) for fold_id in settings.get("triage_fold_ids", [])],
    )
    settings["seed_audit"] = copy.deepcopy(seed_cfg)
    settings["seed_audit"]["resolved_fold_ids"] = fold_ids
    seeds = [int(seed) for seed in seed_cfg.get("seeds", [])]
    if not seeds:
        raise ValueError("Seed-audit extension requires at least one configured seed")

    results: list[dict[str, Any]] = []
    for profile in profiles:
        for seed in seeds:
            seed_config = copy.deepcopy(config)
            _set_cfg(seed_config, ["project", "random_seed"], seed)
            results.append(
                run_profile_experiment(
                    frame,
                    seed_config,
                    profile=profile,
                    checkpoint_dir=checkpoint_dir,
                    run_id=str(source_run_id),
                    fold_scope=_seed_audit_scope(seed),
                    fold_ids=fold_ids or None,
                    resume_existing=bool(settings.get("resume_existing", True)),
                    force_retrain=bool(settings.get("force_retrain", False)),
                    device=device,
                )
            )

    seed_audit, seed_stability = _seed_audit_entries_to_frames(results)
    seed_coverage = _seed_audit_coverage_frame(
        results,
        settings,
        available_fold_ids=available_fold_ids,
    )
    seed_ensemble_entries = _seed_ensemble_entries(results, config)
    seed_ensemble = _seed_ensemble_frame(seed_ensemble_entries)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability, seed_coverage)
    _write_seed_ensemble_files(run_dir, seed_ensemble)

    executed = [result for result in results if not bool(result.get("skipped", False))]
    reused = [result for result in results if bool(result.get("skipped", False))]
    summary = {
        "source_run_id": str(source_run_id),
        "mode": "extend_existing_run",
        "profiles": profiles,
        "seeds": seeds,
        "fold_ids": fold_ids,
        "training_executed_count": len(executed),
        "training_reused_count": len(reused),
        "coverage_passed": bool(
            not seed_coverage.empty
            and "coverage_passed" in seed_coverage.columns
            and seed_coverage["coverage_passed"].astype(bool).all()
        ),
        "source_full_cv_retrained": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "seed_audit_extension_summary.json", summary)
    return {
        "run_id": str(source_run_id),
        "run_dir": run_dir,
        "run_id_source": "seed_audit_extension",
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_audit_coverage": seed_coverage,
        "seed_ensemble": seed_ensemble,
        "comparison": pd.DataFrame(),
        "missing_selected_profiles": pd.DataFrame(),
        "experiment_policy_guard": pd.DataFrame(),
        "future_oos_candidate_plan": pd.DataFrame(),
        "performance_gap_analysis": pd.DataFrame(),
        "payoff_alignment_summary": pd.DataFrame(),
        "payoff_policy_robustness_summary": pd.DataFrame(),
        "training_executed_count": len(executed),
        "training_skipped_count": len(reused),
        "all_training_scopes_reused": bool(results) and not executed,
        "decision": {
            "recommendation": (
                "seed_audit_complete"
                if summary["coverage_passed"]
                else "seed_audit_incomplete"
            ),
            **summary,
        },
    }
