"""Pre-registered multi-seed rank ensemble research for the frozen baseline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.experiment.common import _cfg, _float, _set_cfg, _write_json
from yenibot.experiment.ensembles import (
    _seed_ensemble_entries,
    _seed_ensemble_frame,
    _seed_from_scope,
    _write_seed_ensemble_files,
)
from yenibot.experiment.training import summarize_profile_predictions

__all__ = [
    "build_seed_ensemble_entries",
    "build_seed_ensemble_entries_or_legacy",
    "seed_ensemble_decision_frame",
    "seed_ensemble_report_frames",
    "write_seed_ensemble_outputs",
    "write_seed_ensemble_decision",
]


def _ensemble_config(config: dict[str, Any]) -> dict[str, Any]:
    return _cfg(config, ["experiments", "seed_audit", "ensemble"], {}) or {}


def _midpoint_empirical_cdf(reference: pd.Series, values: pd.Series) -> np.ndarray:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    target = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(ref) < 2:
        raise ValueError("Validation percentile transform requires at least two scores")
    ordered = np.sort(ref)
    left = np.searchsorted(ordered, target, side="left")
    right = np.searchsorted(ordered, target, side="right")
    return (left + right) / (2.0 * len(ordered))


def _validation_percentile_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "split", "prob_long"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Validation-percentile seed ensemble is missing columns: "
            + ", ".join(sorted(missing))
        )

    parts: list[pd.DataFrame] = []
    for fold, fold_part in predictions.groupby("fold", sort=True):
        part = fold_part.copy()
        validation = part.loc[part["split"].eq("val"), "prob_long"]
        if validation.empty:
            raise ValueError(f"Seed ensemble fold {fold} has no validation scores")
        part["prob_long_raw"] = pd.to_numeric(part["prob_long"], errors="coerce")
        part["prob_long"] = _midpoint_empirical_cdf(validation, part["prob_long"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _prediction_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = [column for column in ("split", "fold", "timestamp") if column in frame.columns]
    if "source_row_position" in frame.columns:
        keys.append("source_row_position")
    if not {"fold", "timestamp"}.issubset(keys):
        raise ValueError("Seed ensemble predictions require fold and timestamp keys")
    return keys


def _combine_seed_predictions(
    seed_entries: list[tuple[int, dict[str, Any]]],
    *,
    method: str,
) -> pd.DataFrame:
    if len(seed_entries) < 2:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    seeds: list[int] = []
    for seed, entry in seed_entries:
        prediction = entry["predictions"].copy()
        if method == "validation_percentile_rank_mean":
            prediction = _validation_percentile_predictions(prediction)
        elif method != "raw_probability_mean":
            raise ValueError(f"Unsupported seed ensemble method: {method}")
        prediction["_ensemble_seed"] = int(seed)
        frames.append(prediction)
        seeds.append(int(seed))

    stacked = pd.concat(frames, ignore_index=True)
    key_columns = _prediction_key_columns(stacked)
    seed_count = len(set(seeds))
    grouped = stacked.groupby(key_columns, dropna=False)
    stats = grouped["prob_long"].agg(
        prob_long_ensemble="mean",
        prob_long_seed_std="std",
        prob_long_seed_min="min",
        prob_long_seed_max="max",
        ensemble_seed_count="count",
    ).reset_index()
    stats = stats.loc[stats["ensemble_seed_count"].eq(seed_count)].copy()
    if stats.empty:
        return pd.DataFrame()

    consistency = grouped[["label", "forward_return"]].nunique(dropna=False).reset_index()
    inconsistent = consistency.loc[
        consistency["label"].gt(1) | consistency["forward_return"].gt(1)
    ]
    if not inconsistent.empty:
        raise ValueError("Seed ensemble labels or returns disagree on aligned rows")

    base = grouped.first().reset_index()
    base = base.merge(stats, on=key_columns, how="inner")
    base["prob_long"] = base["prob_long_ensemble"]
    regime_columns = [
        column for column in stacked.columns if column.startswith("regime_prob_")
    ]
    if regime_columns:
        regime_avg = grouped[regime_columns].mean().reset_index()
        base = base.drop(
            columns=[column for column in regime_columns if column in base.columns]
        ).merge(regime_avg, on=key_columns, how="left")
    base = base.drop(
        columns=["_ensemble_seed", "prob_long_ensemble"],
        errors="ignore",
    )
    base["ensemble_method"] = method
    base["ensemble_seeds"] = ",".join(str(seed) for seed in sorted(set(seeds)))
    return base.sort_values(key_columns).reset_index(drop=True)


def _seed_sources(
    profile_results: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    seed_cfg = _cfg(config, ["experiments", "seed_audit"], {}) or {}
    profile = str(
        (seed_cfg.get("profiles", []) or [_cfg(config, ["experiments", "control_profile"])])[0]
    )
    reference_seed = int(seed_cfg.get("reference_full_seed", 42))
    sources: dict[int, dict[str, Any]] = {}
    if bool(seed_cfg.get("include_control_full_seed", False)):
        full = next(
            (
                entry
                for entry in profile_results
                if str(entry.get("profile")) == profile
                and str(entry.get("fold_scope")) == "full"
            ),
            None,
        )
        if full is None:
            raise ValueError("Seed ensemble requires the full control reference scope")
        sources[reference_seed] = full
    for entry in seed_results:
        if str(entry.get("profile")) != profile:
            continue
        seed = _seed_from_scope(str(entry.get("fold_scope", "")))
        if seed is not None and seed not in sources:
            sources[int(seed)] = entry
    return sorted(sources.items())


def _ensemble_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    _set_cfg(updated, ["validation", "calibration", "enabled"], False)
    return updated


def build_seed_ensemble_entries(
    profile_results: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    ensemble_cfg = _ensemble_config(config)
    if not bool(ensemble_cfg.get("enabled", False)):
        return []

    sources = _seed_sources(profile_results, seed_results, config)
    required_seed_count = int(ensemble_cfg.get("required_seed_count", 3))
    if len(sources) != required_seed_count:
        raise ValueError(
            f"Seed ensemble requires {required_seed_count} seeds; found {len(sources)}"
        )
    profile = str(sources[0][1]["profile"])
    feature_columns = list(sources[0][1]["feature_columns"])
    exploratory = {
        int(fold_id) for fold_id in ensemble_cfg.get("exploratory_fold_ids", []) or []
    }
    diagnostic_config = _ensemble_diagnostic_config(config)
    entries: list[dict[str, Any]] = []

    methods = [
        ("raw", "raw_probability_mean"),
        ("rank", "validation_percentile_rank_mean"),
    ]
    for method_name, method in methods:
        combined = _combine_seed_predictions(sources, method=method)
        if combined.empty:
            continue
        available = {
            int(value)
            for value in pd.to_numeric(combined["fold"], errors="coerce").dropna().unique()
        }
        scopes = {
            "all": available,
            "exploratory": available.intersection(exploratory),
            "confirmatory": available.difference(exploratory),
        }
        for evidence_role, fold_ids in scopes.items():
            if not fold_ids:
                continue
            predictions = combined.loc[combined["fold"].isin(sorted(fold_ids))].copy()
            fold_scope = f"seed_ensemble_{method_name}_{evidence_role}"
            diagnostics = summarize_profile_predictions(
                predictions,
                diagnostic_config,
                profile=profile,
                feature_columns=feature_columns,
                fold_scope=fold_scope,
            )
            seed_std = pd.to_numeric(
                predictions["prob_long_seed_std"],
                errors="coerce",
            )
            diagnostics["row"].update(
                {
                    "candidate_id": str(
                        ensemble_cfg.get("candidate_id", "baseline_seed_rank_ensemble_v1")
                    ),
                    "ensemble_method": method,
                    "evidence_role": evidence_role,
                    "score_semantics": (
                        "validation_empirical_percentile_rank"
                        if method_name == "rank"
                        else "raw_sigmoid_mean_comparator_only"
                    ),
                    "automatic_promotion_allowed": False,
                    "seed_count": int(predictions["ensemble_seed_count"].max()),
                    "prob_long_seed_std_mean": float(seed_std.mean()),
                    "prob_long_seed_std_p90": float(seed_std.quantile(0.90)),
                    "ensemble_seeds": str(predictions["ensemble_seeds"].iloc[0]),
                }
            )
            entries.append(
                {
                    "profile": profile,
                    "fold_scope": fold_scope,
                    "feature_columns": feature_columns,
                    "predictions": predictions,
                    "diagnostics": diagnostics,
                    "summary": diagnostics["row"],
                }
            )
    return entries


def build_seed_ensemble_entries_or_legacy(
    profile_results: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use the preregistered ensemble when enabled, otherwise preserve legacy behavior."""

    entries = build_seed_ensemble_entries(profile_results, seed_results, config)
    return entries if entries else _seed_ensemble_entries(seed_results, config)


def _row_for_scope(entries: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    entry = next(
        (item for item in entries if str(item.get("fold_scope")) == scope),
        None,
    )
    return dict(entry["diagnostics"]["row"]) if entry is not None else {}


def seed_ensemble_decision_frame(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    ensemble_cfg = _ensemble_config(config)
    candidate = _row_for_scope(entries, "seed_ensemble_rank_confirmatory")
    raw = _row_for_scope(entries, "seed_ensemble_raw_confirmatory")
    full = next(
        (
            entry
            for entry in entries
            if str(entry.get("profile")) == str(
                _cfg(config, ["experiments", "control_profile"])
            )
            and str(entry.get("fold_scope")) == "full"
        ),
        None,
    )
    if not candidate or full is None:
        return pd.DataFrame()

    confirmatory_folds = {
        int(value)
        for value in pd.to_numeric(
            next(
                item["predictions"]
                for item in entries
                if str(item.get("fold_scope")) == "seed_ensemble_rank_confirmatory"
            )["fold"],
            errors="coerce",
        ).dropna().unique()
    }
    control_predictions = full["predictions"].loc[
        full["predictions"]["fold"].isin(sorted(confirmatory_folds))
    ].copy()
    control_predictions = _validation_percentile_predictions(control_predictions)
    control_diagnostics = summarize_profile_predictions(
        control_predictions,
        _ensemble_diagnostic_config(config),
        profile=str(full["profile"]),
        feature_columns=list(full["feature_columns"]),
        fold_scope="seed_ensemble_control_confirmatory",
    )
    control = dict(control_diagnostics["row"])
    gates = ensemble_cfg.get("confirmatory_gates", {}) or {}

    mean_delta = _float(candidate, "mean_rank_ic") - _float(control, "mean_rank_ic")
    std_delta = _float(candidate, "std_rank_ic") - _float(control, "std_rank_ic")
    positive_delta = (
        _float(candidate, "positive_ic_fraction")
        - _float(control, "positive_ic_fraction")
    )
    worst_delta = (
        _float(candidate, "worst_5_rank_ic_mean")
        - _float(control, "worst_5_rank_ic_mean")
    )
    control_lift = _float(control, "top_10_lift_global")
    lift_ratio = (
        _float(candidate, "top_10_lift_global") / control_lift
        if control_lift > 0
        else np.nan
    )
    f1_delta = (
        _float(candidate, "test_f1_at_official_threshold")
        - _float(control, "test_f1_at_official_threshold")
    )
    reasons: list[str] = []
    if int(candidate.get("fold_count", 0)) < int(gates.get("min_confirmatory_folds", 20)):
        reasons.append("confirmatory_fold_count")
    if mean_delta < float(gates.get("min_mean_rank_ic_delta", -0.003)):
        reasons.append("mean_rank_ic_delta")
    if std_delta > float(gates.get("max_std_rank_ic_delta", -0.005)):
        reasons.append("std_rank_ic_delta")
    if positive_delta < float(gates.get("min_positive_ic_fraction_delta", 0.0)):
        reasons.append("positive_ic_fraction_delta")
    if worst_delta < float(gates.get("min_worst_5_rank_ic_delta", 0.01)):
        reasons.append("worst_5_rank_ic_delta")
    if lift_ratio < float(gates.get("min_top_10_lift_ratio", 0.98)):
        reasons.append("top_10_lift_ratio")
    if _float(candidate, "top_10_forward_return_global") <= float(
        gates.get("min_top_10_forward_return", 0.0)
    ):
        reasons.append("top_10_forward_return")
    if f1_delta < float(gates.get("min_official_f1_delta", -0.005)):
        reasons.append("official_f1_delta")
    if _float(candidate, "test_pred_long_rate_at_official_threshold") > float(
        gates.get("max_prediction_long_rate", 0.70)
    ):
        reasons.append("prediction_long_rate")

    passed = not reasons
    return pd.DataFrame(
        [
            {
                "candidate_id": ensemble_cfg.get(
                    "candidate_id", "baseline_seed_rank_ensemble_v1"
                ),
                "status": (
                    "historical_confirmatory_gates_passed"
                    if passed
                    else "historical_confirmatory_gates_failed"
                ),
                "confirmatory_fold_count": int(candidate.get("fold_count", 0)),
                "ensemble_seeds": candidate.get("ensemble_seeds", ""),
                "ensemble_method": candidate.get("ensemble_method", ""),
                "control_mean_rank_ic": _float(control, "mean_rank_ic"),
                "candidate_mean_rank_ic": _float(candidate, "mean_rank_ic"),
                "mean_rank_ic_delta": mean_delta,
                "control_std_rank_ic": _float(control, "std_rank_ic"),
                "candidate_std_rank_ic": _float(candidate, "std_rank_ic"),
                "std_rank_ic_delta": std_delta,
                "positive_ic_fraction_delta": positive_delta,
                "worst_5_rank_ic_delta": worst_delta,
                "top_10_lift_ratio": lift_ratio,
                "candidate_top_10_forward_return": _float(
                    candidate, "top_10_forward_return_global"
                ),
                "official_f1_delta": f1_delta,
                "candidate_prediction_long_rate": _float(
                    candidate, "test_pred_long_rate_at_official_threshold"
                ),
                "raw_probability_mean_rank_ic": _float(raw, "mean_rank_ic"),
                "historical_cv_gates_passed": passed,
                "candidate_ready_for_future_preregistration": passed,
                "automatic_promotion_allowed": False,
                "reject_reason": ";".join(reasons),
                "next_action": (
                    "review_then_preregister_with_new_future_oos_anchor"
                    if passed
                    else "close_seed_ensemble_hypothesis_without_weight_or_seed_search"
                ),
            }
        ]
    )


def write_seed_ensemble_decision(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "seed_ensemble_decision.csv", index=False)
    _write_json(
        path / "seed_ensemble_decision.json",
        {"rows": frame.to_dict(orient="records")},
    )
    lines = ["# Seed Ensemble Decision", ""]
    if frame.empty:
        lines.append("No confirmatory seed ensemble decision was produced.")
    else:
        row = frame.iloc[0]
        lines.extend(
            [
                f"- Candidate: `{row['candidate_id']}`",
                f"- Status: `{row['status']}`",
                f"- Confirmatory folds: `{row['confirmatory_fold_count']}`",
                f"- Mean Rank IC delta: `{row['mean_rank_ic_delta']}`",
                f"- Rank IC std delta: `{row['std_rank_ic_delta']}`",
                f"- Top-10 lift ratio: `{row['top_10_lift_ratio']}`",
                f"- Official F1 delta: `{row['official_f1_delta']}`",
                f"- Reject reason: `{row['reject_reason']}`",
                "- Automatic promotion allowed: `False`",
            ]
        )
    (path / "seed_ensemble_decision.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def seed_ensemble_report_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _seed_ensemble_frame(entries), seed_ensemble_decision_frame(entries, config)


def write_seed_ensemble_outputs(
    path: Path,
    seed_ensemble: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    _write_seed_ensemble_files(path, seed_ensemble)
    write_seed_ensemble_decision(path, decision)
