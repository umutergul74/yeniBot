"""Uncertainty and probability-calibration evidence for Phase 1 diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score

from yenibot.diagnostics.calibration import calibrate_split_probabilities_from_val
from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
    _table_markdown,
    _write_json,
)
from yenibot.experiment.drift import _binary_probability_metrics
from yenibot.experiment.folds import _entry_threshold_policy_frame
from yenibot.experiment.future_oos import _moving_block_sample_indices
from yenibot.experiment.training import _test_predictions

__all__ = [
    "_model_evidence_uncertainty_frame",
    "_probability_calibration_comparison_frames",
    "_write_model_evidence_uncertainty",
    "_write_probability_calibration_comparison",
]


def _interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return np.nan, np.nan
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(clean, alpha)),
        float(np.quantile(clean, 1.0 - alpha)),
    )


def _official_test_frame(entry: dict[str, Any]) -> pd.DataFrame:
    predictions = entry.get("predictions")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return pd.DataFrame()
    test = _test_predictions(predictions)
    if test.empty or not {"fold", "label", "prob_long"}.issubset(test.columns):
        return pd.DataFrame()
    threshold_metrics = _entry_threshold_policy_frame(entry)
    threshold_by_fold = (
        {
            int(row["fold"]): row.to_dict()
            for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()
        }
        if not threshold_metrics.empty and "fold" in threshold_metrics.columns
        else {}
    )
    calibrated = (entry.get("diagnostics", {}) or {}).get("calibrated_predictions")
    parts: list[pd.DataFrame] = []
    for fold_raw, raw_part in test.groupby("fold"):
        fold = int(fold_raw)
        policy = threshold_by_fold.get(fold, {})
        part = raw_part.copy()
        score_column = "prob_long"
        if bool(policy.get("official_threshold_uses_calibration", False)):
            if isinstance(calibrated, pd.DataFrame) and not calibrated.empty:
                candidate = calibrated.loc[
                    pd.to_numeric(calibrated["fold"], errors="coerce").eq(fold)
                    & calibrated["split"].astype(str).eq("test")
                ].copy()
                if not candidate.empty and "prob_long_calibrated" in candidate.columns:
                    part = candidate
                    score_column = "prob_long_calibrated"
        threshold = _float(policy, "official_threshold")
        if not np.isfinite(threshold):
            continue
        keep = ["timestamp", "fold", "label", score_column]
        return_column = (
            "forward_return"
            if "forward_return" in part.columns
            else "fwd_return_10h"
        )
        if return_column in part.columns:
            keep.append(return_column)
        clean = part[keep].copy().replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna(subset=["label", score_column])
        clean = clean.rename(
            columns={score_column: "score", return_column: "forward_return"}
        )
        clean["official_threshold"] = threshold
        clean["selected"] = clean["score"].astype(float) >= threshold
        parts.append(clean)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _evidence_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "prauc_lift_vs_prevalence": np.nan,
            "precision_lift_vs_prevalence": np.nan,
            "f1_skill_vs_rate_matched_random": np.nan,
            "top_10_forward_return": np.nan,
        }
    labels = pd.to_numeric(frame["label"], errors="coerce")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    selected = frame["selected"].astype(bool)
    valid = labels.notna() & scores.notna()
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float)
    selected = selected[valid]
    if labels.empty:
        return _evidence_metrics(pd.DataFrame())
    prevalence = float(labels.mean())
    pred_rate = float(selected.mean())
    ap = (
        float(average_precision_score(labels, scores))
        if labels.nunique() > 1
        else np.nan
    )
    precision = float(precision_score(labels, selected, zero_division=0))
    f1 = float(f1_score(labels, selected, zero_division=0))
    random_f1 = (
        float(2.0 * prevalence * pred_rate / (prevalence + pred_rate))
        if prevalence + pred_rate > 0
        else 0.0
    )
    top_return = np.nan
    if "forward_return" in frame.columns and len(scores) >= 10:
        threshold = float(scores.quantile(0.90))
        top_mask = scores >= threshold
        returns = pd.to_numeric(frame.loc[valid, "forward_return"], errors="coerce")
        if top_mask.any():
            top_return = float(returns[top_mask].mean())
    return {
        "prauc_lift_vs_prevalence": ap / prevalence
        if prevalence > 0 and np.isfinite(ap)
        else np.nan,
        "precision_lift_vs_prevalence": precision / prevalence
        if prevalence > 0
        else np.nan,
        "f1_skill_vs_rate_matched_random": f1 - random_f1,
        "top_10_forward_return": top_return,
    }


def _model_evidence_uncertainty_frame(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "block_length",
        "bootstrap_repeats",
        "confidence_level",
        "metric",
        "point_estimate",
        "ci_low",
        "ci_high",
        "probability_above_gate",
        "gate",
        "conclusion",
    ]
    cfg = _cfg(config, ["validation", "model_evidence_uncertainty"], {}) or {}
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=columns)
    repeats = max(20, int(cfg.get("bootstrap_repeats", 300)))
    confidence = float(cfg.get("confidence_level", 0.95))
    block_lengths = [int(item) for item in cfg.get("block_lengths", [24, 168])]
    random_seed = int(cfg.get("random_seed", 42))
    gates = {
        "prauc_lift_vs_prevalence": float(
            _cfg(config, ["validation", "classification_skill", "min_prauc_lift_vs_prevalence"], 1.05)
        ),
        "precision_lift_vs_prevalence": float(
            _cfg(config, ["validation", "classification_skill", "min_precision_lift_vs_prevalence"], 1.05)
        ),
        "f1_skill_vs_rate_matched_random": float(
            _cfg(config, ["validation", "classification_skill", "min_f1_skill_vs_rate_random"], 0.0)
        ),
        "top_10_forward_return": 0.0,
    }
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        frame = _official_test_frame(entry)
        if frame.empty:
            continue
        point = _evidence_metrics(frame)
        for block_length in block_lengths:
            rng = np.random.default_rng(random_seed + block_length)
            samples = {metric: [] for metric in gates}
            for _ in range(repeats):
                sampled = frame.iloc[
                    _moving_block_sample_indices(
                        len(frame),
                        block_length=block_length,
                        rng=rng,
                    )
                ].reset_index(drop=True)
                metrics = _evidence_metrics(sampled)
                for metric in samples:
                    samples[metric].append(metrics[metric])
            for metric, gate in gates.items():
                low, high = _interval(samples[metric], confidence)
                clean = np.asarray(
                    [value for value in samples[metric] if np.isfinite(value)],
                    dtype=float,
                )
                probability = (
                    float(np.mean(clean > gate)) if clean.size else np.nan
                )
                conclusion = (
                    "robustly_above_gate"
                    if np.isfinite(low) and low > gate
                    else (
                        "uncertain_near_gate"
                        if np.isfinite(high) and high > gate
                        else "not_supported_above_gate"
                    )
                )
                rows.append(
                    {
                        "candidate": str(entry.get("profile", "")),
                        "candidate_type": _diagnostic_candidate_type(fold_scope),
                        "fold_scope": fold_scope,
                        "block_length": block_length,
                        "bootstrap_repeats": repeats,
                        "confidence_level": confidence,
                        "metric": metric,
                        "point_estimate": point[metric],
                        "ci_low": low,
                        "ci_high": high,
                        "probability_above_gate": probability,
                        "gate": gate,
                        "conclusion": conclusion,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _calibration_line(labels: pd.Series, scores: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"label": labels, "score": scores}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if frame.empty or frame["label"].nunique() < 2 or frame["score"].nunique() < 2:
        return np.nan, np.nan
    clipped = frame["score"].astype(float).clip(1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).to_numpy().reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    model.fit(logits, frame["label"].astype(int))
    return float(model.intercept_[0]), float(model.coef_[0][0])


def _probability_row(
    frame: pd.DataFrame,
    *,
    score_column: str,
    bins: int,
) -> dict[str, float]:
    labels = pd.to_numeric(frame["label"], errors="coerce")
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    valid = labels.notna() & scores.notna()
    labels = labels[valid].astype(int)
    scores = scores[valid].astype(float).clip(1e-6, 1.0 - 1e-6)
    if labels.empty:
        return {}
    metrics = _binary_probability_metrics(labels, scores, bins=bins)
    prevalence = float(labels.mean())
    baseline_brier = float(np.mean((labels - prevalence) ** 2))
    baseline_log_loss = float(
        -np.mean(
            labels * np.log(max(prevalence, 1e-6))
            + (1.0 - labels) * np.log(max(1.0 - prevalence, 1e-6))
        )
    )
    intercept, slope = _calibration_line(labels, scores)
    return {
        **metrics,
        "label_prevalence": prevalence,
        "baseline_brier_score": baseline_brier,
        "brier_skill_vs_climatology": (
            1.0 - metrics["brier_score"] / baseline_brier
            if baseline_brier > 0
            else np.nan
        ),
        "baseline_log_loss": baseline_log_loss,
        "log_loss_skill_vs_climatology": (
            1.0 - metrics["log_loss"] / baseline_log_loss
            if baseline_log_loss > 0
            else np.nan
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _probability_calibration_comparison_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    bins = int(_cfg(config, ["validation", "calibration_bins"], 10))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        method_frames: dict[str, pd.DataFrame] = {
            "raw": _test_predictions(predictions).copy()
        }
        for method in ("platt", "isotonic"):
            try:
                calibrated = calibrate_split_probabilities_from_val(
                    predictions,
                    method=method,
                )
            except (ValueError, TypeError):
                continue
            method_frames[method] = calibrated.loc[
                calibrated["split"].astype(str).eq("test")
            ].copy()
        for method, test in method_frames.items():
            score_column = (
                "prob_long" if method == "raw" else "prob_long_calibrated"
            )
            for fold_raw, part in test.groupby("fold"):
                metrics = _probability_row(
                    part,
                    score_column=score_column,
                    bins=bins,
                )
                if not metrics:
                    continue
                detail_rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "fold": int(fold_raw),
                        "method": method,
                        "fit_source": (
                            "none_raw_scores"
                            if method == "raw"
                            else "fold_validation_only"
                        ),
                        "test_rows": int(len(part)),
                        **metrics,
                    }
                )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for keys, part in detail.groupby(
        ["candidate", "candidate_type", "fold_scope", "method"],
        dropna=False,
    ):
        candidate, candidate_type, fold_scope, method = keys
        numeric_columns = [
            "brier_score",
            "baseline_brier_score",
            "brier_skill_vs_climatology",
            "log_loss",
            "baseline_log_loss",
            "log_loss_skill_vs_climatology",
            "average_precision",
            "ece_equal_count",
            "prob_long_mean",
            "label_prevalence",
            "calibration_intercept",
            "calibration_slope",
        ]
        values = {
            column: float(pd.to_numeric(part[column], errors="coerce").mean())
            for column in numeric_columns
        }
        probability_valid = bool(
            values["brier_skill_vs_climatology"] > 0.0
            and values["log_loss_skill_vs_climatology"] > 0.0
            and values["ece_equal_count"] < 0.05
        )
        summary_rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "method": method,
                "fold_count": int(part["fold"].nunique()),
                **{f"mean_{key}": value for key, value in values.items()},
                "probability_quality_passed": probability_valid,
                "deployment_status": "diagnostic_only_not_part_of_frozen_candidate",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["candidate_type", "candidate", "fold_scope", "mean_brier_score"]
    ).reset_index(drop=True)
    return detail, summary


def _write_model_evidence_uncertainty(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "model_evidence_uncertainty.csv", index=False)
    (path / "model_evidence_uncertainty.md").write_text(
        _table_markdown("Model Evidence Uncertainty", frame),
        encoding="utf-8",
    )
    _write_json(
        path / "model_evidence_uncertainty.json",
        {"rows": frame.to_dict(orient="records")},
    )


def _write_probability_calibration_comparison(
    path: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "probability_calibration_comparison_by_fold.csv", index=False)
    summary.to_csv(path / "probability_calibration_comparison.csv", index=False)
    (path / "probability_calibration_comparison.md").write_text(
        _table_markdown("Probability Calibration Comparison", summary),
        encoding="utf-8",
    )
    _write_json(
        path / "probability_calibration_comparison.json",
        {
            "summary": summary.to_dict(orient="records"),
            "by_fold": detail.to_dict(orient="records"),
        },
    )
