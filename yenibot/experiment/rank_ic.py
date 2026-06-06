"""Rank-IC sampling uncertainty and aggregate stability evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, spearmanr

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _is_stability_scope,
    _write_json,
)

from yenibot.experiment.training import (
    _test_predictions,
)

__all__ = [
    '_score_quantile_threshold',
    '_moving_block_rank_ic_values',
    '_rank_ic_uncertainty_frames',
    '_rank_ic_uncertainty_markdown',
    '_write_rank_ic_uncertainty',
    '_random_effects_rank_ic',
    '_rank_ic_stability_evidence_frames',
    '_rank_ic_stability_evidence_markdown',
    '_write_rank_ic_stability_evidence',
]

def _score_quantile_threshold(frame: pd.DataFrame, selection_rate: float) -> float:
    if frame.empty or "prob_long" not in frame.columns:
        return np.nan
    rate = float(selection_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        return np.inf
    if rate >= 1.0:
        return -np.inf
    scores = pd.to_numeric(frame["prob_long"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return np.nan
    return float(scores.quantile(max(0.0, min(1.0, 1.0 - rate))))

def _moving_block_rank_ic_values(
    scores: np.ndarray,
    returns: np.ndarray,
    *,
    block_length: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = int(len(scores))
    effective_block = min(max(1, int(block_length)), count)
    block_count = int(np.ceil(count / effective_block))
    offsets = np.arange(effective_block)
    values = np.empty(int(repeats), dtype=float)
    for repeat in range(int(repeats)):
        starts = rng.integers(0, count, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % count).reshape(-1)[:count]
        values[repeat] = float(spearmanr(scores[indices], returns[indices]).statistic)
    return values[np.isfinite(values)]

def _rank_ic_uncertainty_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "count",
        "observed_rank_ic",
        "independent_sampling_se",
        "bootstrap_rank_ic_mean",
        "bootstrap_rank_ic_std",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_positive_probability",
        "block_length",
        "bootstrap_repeats",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "observed_mean_rank_ic",
        "observed_std_rank_ic",
        "mean_fold_count",
        "independent_noise_floor_std",
        "block_bootstrap_noise_floor_std",
        "estimated_between_fold_std",
        "sampling_variance_fraction",
        "observed_std_to_bootstrap_noise_ratio",
        "target_rank_ic_std",
        "target_below_independent_noise_floor",
        "target_below_block_bootstrap_noise_floor",
        "positive_probability_mean",
        "diagnostic_conclusion",
        "recommended_action",
        "block_length",
        "bootstrap_repeats",
    ]
    cfg = _cfg(config, ["validation", "rank_ic_uncertainty"], {}) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)
    block_length = max(1, int(cfg.get("block_length", 24)))
    repeats = max(20, int(cfg.get("bootstrap_repeats", 120)))
    confidence_level = float(cfg.get("confidence_level", 0.95))
    confidence_level = min(max(confidence_level, 0.50), 0.999)
    alpha = (1.0 - confidence_level) / 2.0
    base_seed = int(cfg.get("random_seed", _cfg(config, ["project", "random_seed"], 42)))
    rows: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(entries):
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test = _test_predictions(predictions)
        if not {"fold", "prob_long", "forward_return"}.issubset(test.columns):
            continue
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold_raw, part in test.groupby("fold"):
            clean = (
                part[["prob_long", "forward_return"]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            count = int(len(clean))
            if count < 8:
                continue
            scores = clean["prob_long"].to_numpy(dtype=float)
            returns = clean["forward_return"].to_numpy(dtype=float)
            observed = float(spearmanr(scores, returns).statistic)
            if not np.isfinite(observed):
                continue
            effective_block = min(block_length, count)
            rng = np.random.default_rng(base_seed + entry_index * 100_003 + int(fold_raw) * 997)
            boot_values = _moving_block_rank_ic_values(
                scores,
                returns,
                block_length=effective_block,
                repeats=repeats,
                rng=rng,
            )
            independent_se = float((1.0 - observed**2) / np.sqrt(max(count - 1, 1)))
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": int(fold_raw),
                    "count": count,
                    "observed_rank_ic": observed,
                    "independent_sampling_se": independent_se,
                    "bootstrap_rank_ic_mean": float(np.mean(boot_values)) if len(boot_values) else np.nan,
                    "bootstrap_rank_ic_std": float(np.std(boot_values, ddof=1)) if len(boot_values) > 1 else np.nan,
                    "bootstrap_ci_low": float(np.quantile(boot_values, alpha)) if len(boot_values) else np.nan,
                    "bootstrap_ci_high": float(np.quantile(boot_values, 1.0 - alpha)) if len(boot_values) else np.nan,
                    "bootstrap_positive_probability": float(np.mean(boot_values > 0.0)) if len(boot_values) else np.nan,
                    "block_length": effective_block,
                    "bootstrap_repeats": int(len(boot_values)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)

    by_fold = (
        pd.DataFrame(rows, columns=by_fold_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "fold"])
        .reset_index(drop=True)
    )
    target_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    summaries: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in by_fold.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        observed = pd.to_numeric(part["observed_rank_ic"], errors="coerce").dropna()
        independent_se = pd.to_numeric(part["independent_sampling_se"], errors="coerce").dropna()
        bootstrap_se = pd.to_numeric(part["bootstrap_rank_ic_std"], errors="coerce").dropna()
        observed_variance = float(observed.var(ddof=0)) if len(observed) else np.nan
        independent_noise = float(np.sqrt(np.mean(np.square(independent_se)))) if len(independent_se) else np.nan
        bootstrap_noise = float(np.sqrt(np.mean(np.square(bootstrap_se)))) if len(bootstrap_se) else np.nan
        bootstrap_variance = bootstrap_noise**2 if np.isfinite(bootstrap_noise) else np.nan
        between_variance = (
            max(observed_variance - bootstrap_variance, 0.0)
            if np.isfinite(observed_variance) and np.isfinite(bootstrap_variance)
            else np.nan
        )
        between_std = float(np.sqrt(between_variance)) if np.isfinite(between_variance) else np.nan
        sampling_fraction = (
            min(max(bootstrap_variance / observed_variance, 0.0), 1.0)
            if np.isfinite(bootstrap_variance) and np.isfinite(observed_variance) and observed_variance > 0
            else np.nan
        )
        observed_std = float(observed.std(ddof=0)) if len(observed) else np.nan
        ratio = (
            observed_std / bootstrap_noise
            if np.isfinite(observed_std) and np.isfinite(bootstrap_noise) and bootstrap_noise > 0
            else np.nan
        )
        target_below_independent = bool(np.isfinite(independent_noise) and target_std < independent_noise)
        target_below_bootstrap = bool(np.isfinite(bootstrap_noise) and target_std < bootstrap_noise)
        if target_below_bootstrap:
            conclusion = "official_std_target_below_estimated_fold_measurement_noise"
            action = "keep_official_gate_but_review_its_statistical_feasibility; prioritize noise_adjusted_stability"
        elif np.isfinite(sampling_fraction) and sampling_fraction >= 0.50:
            conclusion = "observed_std_substantially_explained_by_sampling_noise"
            action = "avoid_profile_churn; confirm with longer non_overlapping_oos_windows"
        elif np.isfinite(between_std) and between_std > target_std:
            conclusion = "material_between_fold_instability_remains_after_noise_adjustment"
            action = "target_score_reversal_mechanism_with_pre_registered_causal_hypothesis"
        else:
            conclusion = "noise_adjusted_fold_stability_near_target"
            action = "do_not_retrain_for_std_alone; focus_on_threshold_transfer_and_future_oos"
        summaries.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "fold_scope": str(fold_scope),
                "fold_count": int(part["fold"].nunique()),
                "observed_mean_rank_ic": float(observed.mean()) if len(observed) else np.nan,
                "observed_std_rank_ic": observed_std,
                "mean_fold_count": float(pd.to_numeric(part["count"], errors="coerce").mean()),
                "independent_noise_floor_std": independent_noise,
                "block_bootstrap_noise_floor_std": bootstrap_noise,
                "estimated_between_fold_std": between_std,
                "sampling_variance_fraction": sampling_fraction,
                "observed_std_to_bootstrap_noise_ratio": ratio,
                "target_rank_ic_std": target_std,
                "target_below_independent_noise_floor": target_below_independent,
                "target_below_block_bootstrap_noise_floor": target_below_bootstrap,
                "positive_probability_mean": float(
                    pd.to_numeric(part["bootstrap_positive_probability"], errors="coerce").mean()
                ),
                "diagnostic_conclusion": conclusion,
                "recommended_action": action,
                "block_length": block_length,
                "bootstrap_repeats": repeats,
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope"])
        .reset_index(drop=True)
    )
    return summary, by_fold

def _rank_ic_uncertainty_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Rank IC Variance Decomposition", ""]
    lines.append(
        "This report separates observed fold-to-fold Rank IC variance from finite-sample uncertainty. "
        "The moving-block bootstrap preserves short-range time dependence. It is a diagnostic and does "
        "not replace the official Phase 1 Rank IC std gate."
    )
    if summary.empty:
        lines.extend(["", "No Rank IC uncertainty rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "observed_std_rank_ic",
        "independent_noise_floor_std",
        "block_bootstrap_noise_floor_std",
        "estimated_between_fold_std",
        "sampling_variance_fraction",
        "target_rank_ic_std",
        "diagnostic_conclusion",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append(f"Fold-level uncertainty rows: {len(by_fold)}. See `rank_ic_sampling_uncertainty.csv`.")
    return "\n".join(lines)

def _write_rank_ic_uncertainty(path: Path, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "rank_ic_variance_decomposition.csv", index=False)
    by_fold.to_csv(path / "rank_ic_sampling_uncertainty.csv", index=False)
    (path / "rank_ic_variance_decomposition.md").write_text(
        _rank_ic_uncertainty_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "rank_ic_variance_decomposition.json",
        {"summary": summary.to_dict(orient="records"), "by_fold": by_fold.to_dict(orient="records")},
    )

def _random_effects_rank_ic(effect: np.ndarray, variance: np.ndarray, confidence_level: float) -> dict[str, float]:
    valid = np.isfinite(effect) & np.isfinite(variance) & (variance > 0)
    values = effect[valid].astype(float)
    variances = variance[valid].astype(float)
    count = int(len(values))
    if count < 2:
        return {
            "fixed_effect_mean_rank_ic": np.nan,
            "random_effects_mean_rank_ic": np.nan,
            "random_effects_se": np.nan,
            "random_effects_ci_low": np.nan,
            "random_effects_ci_high": np.nan,
            "heterogeneity_q": np.nan,
            "heterogeneity_i2": np.nan,
            "between_fold_tau2": np.nan,
            "between_fold_tau": np.nan,
        }
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * values) / np.sum(fixed_weights))
    q_value = float(np.sum(fixed_weights * np.square(values - fixed_mean)))
    degrees = count - 1
    c_value = float(np.sum(fixed_weights) - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights))
    tau2 = float(max((q_value - degrees) / c_value, 0.0)) if c_value > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    random_mean = float(np.sum(random_weights * values) / np.sum(random_weights))
    random_se = float(np.sqrt(1.0 / np.sum(random_weights)))
    z_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    i2 = float(max((q_value - degrees) / q_value, 0.0)) if q_value > 0 else 0.0
    return {
        "fixed_effect_mean_rank_ic": fixed_mean,
        "random_effects_mean_rank_ic": random_mean,
        "random_effects_se": random_se,
        "random_effects_ci_low": random_mean - z_value * random_se,
        "random_effects_ci_high": random_mean + z_value * random_se,
        "heterogeneity_q": q_value,
        "heterogeneity_i2": i2,
        "between_fold_tau2": tau2,
        "between_fold_tau": float(np.sqrt(tau2)),
    }

def _rank_ic_stability_evidence_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "block_length",
        "fold_count",
        "bootstrap_repeats",
        "observed_mean_rank_ic",
        "observed_std_rank_ic",
        "rms_bootstrap_noise_std",
        "target_rank_ic_std",
        "target_below_noise_floor",
        "fixed_effect_mean_rank_ic",
        "random_effects_mean_rank_ic",
        "random_effects_se",
        "random_effects_ci_low",
        "random_effects_ci_high",
        "heterogeneity_q",
        "heterogeneity_i2",
        "between_fold_tau2",
        "between_fold_tau",
        "positive_fold_fraction",
        "positive_fold_sign_test_pvalue",
    ]
    evidence_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "observed_mean_rank_ic",
        "observed_std_rank_ic",
        "positive_fold_fraction",
        "positive_fold_sign_test_pvalue",
        "block_length_count",
        "min_noise_floor_std",
        "max_noise_floor_std",
        "target_rank_ic_std",
        "target_below_noise_floor_all_blocks",
        "random_effects_ci_low_min",
        "random_effects_ci_low_max",
        "random_effects_positive_all_blocks",
        "between_fold_tau_min",
        "between_fold_tau_max",
        "heterogeneity_i2_min",
        "heterogeneity_i2_max",
        "evidence_conclusion",
        "recommended_governance_action",
    ]
    cfg = _cfg(config, ["validation", "rank_ic_uncertainty"], {}) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=evidence_columns), pd.DataFrame(columns=sensitivity_columns)
    block_lengths = cfg.get("block_lengths", [1, 6, 12, 24, 48]) or [1, 6, 12, 24, 48]
    lengths = sorted({max(1, int(value)) for value in block_lengths})
    repeats = max(20, int(cfg.get("sensitivity_bootstrap_repeats", 80)))
    confidence_level = min(max(float(cfg.get("confidence_level", 0.95)), 0.50), 0.999)
    base_seed = int(cfg.get("random_seed", _cfg(config, ["project", "random_seed"], 42)))
    target_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    rows: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(entries):
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test = _test_predictions(predictions)
        if not {"fold", "prob_long", "forward_return"}.issubset(test.columns):
            continue
        fold_data: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for fold_raw, part in test.groupby("fold"):
            clean = (
                part[["prob_long", "forward_return"]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(clean) < 8:
                continue
            scores = clean["prob_long"].to_numpy(dtype=float)
            returns = clean["forward_return"].to_numpy(dtype=float)
            observed = float(spearmanr(scores, returns).statistic)
            if np.isfinite(observed):
                fold_data.append((int(fold_raw), scores, returns, observed))
        if len(fold_data) < 2:
            continue
        effects = np.asarray([item[3] for item in fold_data], dtype=float)
        positive_count = int(np.sum(effects > 0.0))
        sign_pvalue = float(binomtest(positive_count, len(effects), 0.5, alternative="greater").pvalue)
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for block_length in lengths:
            variances: list[float] = []
            for fold, scores, returns, _observed in fold_data:
                rng = np.random.default_rng(
                    base_seed
                    + entry_index * 1_000_003
                    + int(block_length) * 10_007
                    + fold * 997
                )
                values = _moving_block_rank_ic_values(
                    scores,
                    returns,
                    block_length=block_length,
                    repeats=repeats,
                    rng=rng,
                )
                variances.append(float(np.var(values, ddof=1)) if len(values) > 1 else np.nan)
            variance_array = np.asarray(variances, dtype=float)
            meta = _random_effects_rank_ic(effects, variance_array, confidence_level)
            rms_noise = float(np.sqrt(np.nanmean(variance_array))) if np.isfinite(variance_array).any() else np.nan
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "block_length": block_length,
                    "fold_count": len(fold_data),
                    "bootstrap_repeats": repeats,
                    "observed_mean_rank_ic": float(np.mean(effects)),
                    "observed_std_rank_ic": float(np.std(effects, ddof=0)),
                    "rms_bootstrap_noise_std": rms_noise,
                    "target_rank_ic_std": target_std,
                    "target_below_noise_floor": bool(np.isfinite(rms_noise) and target_std < rms_noise),
                    **meta,
                    "positive_fold_fraction": float(positive_count / len(effects)),
                    "positive_fold_sign_test_pvalue": sign_pvalue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=evidence_columns), pd.DataFrame(columns=sensitivity_columns)
    sensitivity = (
        pd.DataFrame(rows, columns=sensitivity_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "block_length"])
        .reset_index(drop=True)
    )
    evidence_rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in sensitivity.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        target_below_all = bool(part["target_below_noise_floor"].astype(bool).all())
        random_positive_all = bool(
            pd.to_numeric(part["random_effects_ci_low"], errors="coerce").gt(0.0).all()
        )
        sign_p = float(pd.to_numeric(part["positive_fold_sign_test_pvalue"], errors="coerce").iloc[0])
        if target_below_all and random_positive_all and sign_p < 0.01:
            conclusion = "positive_aggregate_signal_with_unrealistic_absolute_std_target"
            action = "formal_charter_review_recommended; retain_std_as_monitor_not_absolute_gate"
        elif random_positive_all and sign_p < 0.05:
            conclusion = "positive_aggregate_signal_with_measurable_heterogeneity"
            action = "retain_stability_monitor_and_require_future_unseen_oos"
        else:
            conclusion = "aggregate_rank_ic_evidence_not_robust_across_block_lengths"
            action = "do_not_relax_current_gate"
        evidence_rows.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "fold_scope": str(fold_scope),
                "fold_count": int(part["fold_count"].iloc[0]),
                "observed_mean_rank_ic": float(part["observed_mean_rank_ic"].iloc[0]),
                "observed_std_rank_ic": float(part["observed_std_rank_ic"].iloc[0]),
                "positive_fold_fraction": float(part["positive_fold_fraction"].iloc[0]),
                "positive_fold_sign_test_pvalue": sign_p,
                "block_length_count": int(part["block_length"].nunique()),
                "min_noise_floor_std": float(pd.to_numeric(part["rms_bootstrap_noise_std"], errors="coerce").min()),
                "max_noise_floor_std": float(pd.to_numeric(part["rms_bootstrap_noise_std"], errors="coerce").max()),
                "target_rank_ic_std": target_std,
                "target_below_noise_floor_all_blocks": target_below_all,
                "random_effects_ci_low_min": float(
                    pd.to_numeric(part["random_effects_ci_low"], errors="coerce").min()
                ),
                "random_effects_ci_low_max": float(
                    pd.to_numeric(part["random_effects_ci_low"], errors="coerce").max()
                ),
                "random_effects_positive_all_blocks": random_positive_all,
                "between_fold_tau_min": float(pd.to_numeric(part["between_fold_tau"], errors="coerce").min()),
                "between_fold_tau_max": float(pd.to_numeric(part["between_fold_tau"], errors="coerce").max()),
                "heterogeneity_i2_min": float(pd.to_numeric(part["heterogeneity_i2"], errors="coerce").min()),
                "heterogeneity_i2_max": float(pd.to_numeric(part["heterogeneity_i2"], errors="coerce").max()),
                "evidence_conclusion": conclusion,
                "recommended_governance_action": action,
            }
        )
    evidence = (
        pd.DataFrame(evidence_rows, columns=evidence_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope"])
        .reset_index(drop=True)
    )
    return evidence, sensitivity

def _rank_ic_stability_evidence_markdown(evidence: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    lines = ["# Rank IC Stability Evidence", ""]
    lines.append(
        "This governance diagnostic checks whether the official absolute Rank IC std target is below "
        "the finite-fold uncertainty floor across several moving-block lengths. Random-effects estimates "
        "summarize aggregate IC while explicitly allowing fold heterogeneity. Official Phase 1 gates remain unchanged."
    )
    if evidence.empty:
        lines.extend(["", "No Rank IC stability evidence rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "observed_mean_rank_ic",
        "observed_std_rank_ic",
        "positive_fold_fraction",
        "positive_fold_sign_test_pvalue",
        "min_noise_floor_std",
        "max_noise_floor_std",
        "target_rank_ic_std",
        "random_effects_ci_low_min",
        "random_effects_positive_all_blocks",
        "evidence_conclusion",
        "recommended_governance_action",
    ]
    visible = evidence[[column for column in display_cols if column in evidence.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append(f"Block-sensitivity rows: {len(sensitivity)}. See `rank_ic_block_sensitivity.csv`.")
    return "\n".join(lines)

def _write_rank_ic_stability_evidence(
    path: Path,
    evidence: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(path / "rank_ic_aggregate_evidence.csv", index=False)
    sensitivity.to_csv(path / "rank_ic_block_sensitivity.csv", index=False)
    (path / "rank_ic_stability_evidence.md").write_text(
        _rank_ic_stability_evidence_markdown(evidence, sensitivity),
        encoding="utf-8",
    )
    _write_json(
        path / "rank_ic_stability_evidence.json",
        {"aggregate_evidence": evidence.to_dict(orient="records"), "block_sensitivity": sensitivity.to_dict(orient="records")},
    )
