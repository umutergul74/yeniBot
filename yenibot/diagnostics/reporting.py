from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from scipy.stats import ks_2samp

from yenibot.diagnostics.metrics import classification_metrics, rank_ic
from yenibot.features.builder import (
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    RAW_COLUMNS,
    raw_order_flow_v2_model_exclusions,
    resolve_feature_profile,
)


def fold_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in predictions.groupby("fold"):
        metrics = classification_metrics(part["label"], part["prob_long"])
        rows.append(
            {
                "fold": int(fold),
                "count": int(len(part)),
                "start": str(part["timestamp"].min()) if "timestamp" in part.columns else "",
                "end": str(part["timestamp"].max()) if "timestamp" in part.columns else "",
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "long_f1": metrics["long_f1"],
                "prauc": metrics["prauc"],
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float((part["prob_long"] >= 0.5).mean()),
                "prob_long_mean": float(part["prob_long"].mean()),
                "prob_long_std": float(part["prob_long"].std(ddof=0)),
                "prob_long_p10": float(part["prob_long"].quantile(0.10)),
                "prob_long_p50": float(part["prob_long"].quantile(0.50)),
                "prob_long_p90": float(part["prob_long"].quantile(0.90)),
                "forward_return_mean": float(part["forward_return"].mean()),
                "forward_return_std": float(part["forward_return"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def regime_diagnostics(predictions: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    regime_columns = [column for column in predictions.columns if column.startswith("regime_prob_")]
    if not regime_columns:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["regime"] = frame[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    rows = []
    for regime, part in frame.groupby("regime"):
        y_true = part["label"].astype(int)
        y_pred = (part["prob_long"] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "regime": int(regime),
                "count": int(len(part)),
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float(y_pred.mean()),
                "precision_050": float(precision),
                "recall_050": float(recall),
                "long_f1_050": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def good_bad_fold_summary(fold_metrics: pd.DataFrame, *, good_ic: float = 0.10, bad_ic: float = -0.08) -> dict[str, Any]:
    good = fold_metrics.loc[fold_metrics["rank_ic"] >= good_ic, "fold"].astype(int).tolist()
    bad = fold_metrics.loc[fold_metrics["rank_ic"] <= bad_ic, "fold"].astype(int).tolist()
    return {
        "good_ic_threshold": good_ic,
        "bad_ic_threshold": bad_ic,
        "good_folds": good,
        "bad_folds": bad,
        "good_fold_count": len(good),
        "bad_fold_count": len(bad),
    }


def threshold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    threshold_source: str = "val",
) -> pd.DataFrame:
    """Report whether bad F1 is mostly a threshold/calibration problem."""

    rows = []
    has_splits = "split" in predictions.columns
    for fold, fold_part in predictions.groupby("fold"):
        if has_splits and threshold_source == "val":
            source = fold_part[fold_part["split"] == "val"]
            target = fold_part[fold_part["split"] == "test"]
            source_name = "val"
        else:
            source = fold_part
            target = fold_part
            source_name = "same_split_oracle"
        if source.empty or target.empty:
            continue

        source_best = best_f1_threshold(source["label"], source[score_column])
        target_at_source = _metrics_at_threshold(target["label"], target[score_column], source_best["threshold"])
        target_oracle = best_f1_threshold(target["label"], target[score_column])
        rows.append(
            {
                "fold": int(fold),
                "threshold_source": source_name,
                "selected_threshold": source_best["threshold"],
                "source_best_f1": source_best["f1"],
                "test_f1_at_selected_threshold": target_at_source["f1"],
                "test_precision_at_selected_threshold": target_at_source["precision"],
                "test_recall_at_selected_threshold": target_at_source["recall"],
                "test_pred_long_rate_at_selected_threshold": target_at_source["pred_long_rate"],
                "test_oracle_best_threshold": target_oracle["threshold"],
                "test_oracle_best_f1": target_oracle["f1"],
                "test_f1_at_050": _metrics_at_threshold(target["label"], target[score_column], 0.5)["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def threshold_summary_diagnostics(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    if threshold_metrics.empty:
        return pd.DataFrame()
    columns = [
        "selected_threshold",
        "source_best_f1",
        "test_f1_at_selected_threshold",
        "test_precision_at_selected_threshold",
        "test_recall_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "test_oracle_best_f1",
        "test_f1_at_050",
    ]
    rows = []
    for column in columns:
        if column not in threshold_metrics.columns:
            continue
        values = threshold_metrics[column].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def best_f1_threshold(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = scores.astype(float).to_numpy()
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        threshold = float(np.median(y_score)) if len(y_score) else 0.5
        return {"threshold": threshold, **_metrics_at_threshold(labels, scores, threshold)}

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    return {"threshold": threshold, "f1": float(f1[best_idx]), "precision": float(precision[best_idx]), "recall": float(recall[best_idx]), "pred_long_rate": float((y_score >= threshold).mean())}


def score_lift_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
) -> pd.DataFrame:
    required = {"label", score_column}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    frame = _assign_score_bins(predictions, score_column=score_column, bins=bins)
    if frame.empty:
        return pd.DataFrame()
    base_long_rate = float(frame["label"].mean())
    grouped = frame.groupby("score_bin", observed=True).agg(
        count=("label", "size"),
        mean_prob_long=(score_column, "mean"),
        actual_long_rate=("label", "mean"),
    )
    if "forward_return" in frame.columns:
        grouped["mean_forward_return"] = frame.groupby("score_bin", observed=True)["forward_return"].mean()
    grouped = grouped.reset_index()
    grouped["base_long_rate"] = base_long_rate
    grouped["lift_vs_base"] = grouped["actual_long_rate"] / base_long_rate if base_long_rate > 0 else np.nan
    grouped["is_top_bin"] = grouped["score_bin"] == grouped["score_bin"].max()
    return grouped


def score_band_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
    bands: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    required = {"label", score_column}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    frame = _assign_score_bins(predictions, score_column=score_column, bins=bins)
    if frame.empty:
        return pd.DataFrame()

    resolved_bands = _resolve_score_bands(bands, bins=int(frame["score_bin"].max()) + 1)
    base_long_rate = float(frame["label"].mean())
    rows = []
    for band in resolved_bands:
        part = frame[(frame["score_bin"] >= band["min_bin"]) & (frame["score_bin"] <= band["max_bin"])]
        if part.empty:
            continue
        row = {
            "band": band["name"],
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "count": int(len(part)),
            "selection_rate": float(len(part) / len(frame)),
            "mean_prob_long": float(part[score_column].mean()),
            "actual_long_rate": float(part["label"].mean()),
            "base_long_rate": base_long_rate,
            "lift_vs_base": float(part["label"].mean() / base_long_rate) if base_long_rate > 0 else np.nan,
        }
        if "forward_return" in part.columns:
            row["mean_forward_return"] = float(part["forward_return"].mean())
            row["rank_ic_within_band"] = rank_ic(part[score_column], part["forward_return"])
        rows.append(row)
    return pd.DataFrame(rows)


def score_band_by_fold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
    bands: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    if "fold" not in predictions.columns:
        return pd.DataFrame()
    for fold, part in predictions.groupby("fold"):
        band_lift = score_band_diagnostics(part, score_column=score_column, bins=bins, bands=bands)
        if band_lift.empty:
            continue
        band_lift = band_lift.copy()
        band_lift.insert(0, "fold", int(fold))
        rows.append(band_lift)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["fold", "min_bin", "max_bin"]).reset_index(drop=True)


def score_band_summary_diagnostics(score_band_by_fold: pd.DataFrame) -> pd.DataFrame:
    if score_band_by_fold is None or score_band_by_fold.empty:
        return pd.DataFrame()
    frame = score_band_by_fold.copy()
    aggregations: dict[str, Any] = {
        "folds": ("fold", "nunique"),
        "mean_selection_rate": ("selection_rate", "mean"),
        "mean_actual_long_rate": ("actual_long_rate", "mean"),
        "mean_lift_vs_base": ("lift_vs_base", "mean"),
        "median_lift_vs_base": ("lift_vs_base", "median"),
        "positive_lift_fold_rate": ("lift_vs_base", lambda values: float((values > 1.0).mean())),
    }
    if "mean_forward_return" in frame.columns:
        aggregations["mean_forward_return"] = ("mean_forward_return", "mean")
        aggregations["positive_forward_return_fold_rate"] = (
            "mean_forward_return",
            lambda values: float((values > 0).mean()),
        )
    if "rank_ic_within_band" in frame.columns:
        aggregations["mean_rank_ic_within_band"] = ("rank_ic_within_band", "mean")
    summary = frame.groupby(["band", "min_bin", "max_bin"], as_index=False).agg(**aggregations)
    sort_columns = ["mean_lift_vs_base"]
    if "mean_forward_return" in summary.columns:
        sort_columns.append("mean_forward_return")
    return summary.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def score_lift_by_fold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for fold, part in predictions.groupby("fold"):
        lift = score_lift_diagnostics(part, score_column=score_column, bins=bins)
        if lift.empty:
            continue
        bottom = lift.sort_values("score_bin").iloc[0]
        top = lift.sort_values("score_bin").iloc[-1]
        long_rate_spearman = lift["score_bin"].corr(lift["actual_long_rate"], method="spearman")
        row = {
            "fold": int(fold),
            "count": int(part["label"].notna().sum()),
            "base_long_rate": float(part["label"].mean()),
            "bottom_bin_long_rate": float(bottom["actual_long_rate"]),
            "top_bin_long_rate": float(top["actual_long_rate"]),
            "top_lift_vs_base": float(top["lift_vs_base"]),
            "top_minus_bottom_long_rate": float(top["actual_long_rate"] - bottom["actual_long_rate"]),
            "bin_long_rate_spearman": float(long_rate_spearman) if pd.notna(long_rate_spearman) else np.nan,
            "fold_rank_ic": rank_ic(part[score_column], part["forward_return"]) if "forward_return" in part.columns else np.nan,
        }
        if "mean_forward_return" in lift.columns:
            row["bottom_bin_forward_return"] = float(bottom["mean_forward_return"])
            row["top_bin_forward_return"] = float(top["mean_forward_return"])
            row["top_minus_bottom_forward_return"] = float(top["mean_forward_return"] - bottom["mean_forward_return"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def bad_fold_feature_forensics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
    target_column: str = "forward_return",
    min_abs_reference_ic: float = 0.02,
) -> pd.DataFrame:
    required = {"fold", target_column}
    if not required.issubset(predictions.columns) or fold_metrics.empty:
        return pd.DataFrame()

    summary = good_bad_fold_summary(fold_metrics, good_ic=good_ic, bad_ic=bad_ic)
    good_folds = set(summary["good_folds"])
    bad_folds = set(summary["bad_folds"])
    if not good_folds or not bad_folds:
        return pd.DataFrame()

    if feature_columns is None:
        feature_columns = _diagnostic_feature_columns(predictions)
    feature_columns = [column for column in feature_columns if column in predictions.columns]
    good = predictions[predictions["fold"].isin(good_folds)]
    fold_rank_ic = fold_metrics.set_index("fold")["rank_ic"].to_dict()
    rows = []
    for bad_fold in sorted(bad_folds):
        bad = predictions[predictions["fold"] == bad_fold]
        if bad.empty:
            continue
        for feature in feature_columns:
            if not pd.api.types.is_numeric_dtype(predictions[feature]):
                continue
            good_values = good[feature].replace([np.inf, -np.inf], np.nan).dropna()
            bad_values = bad[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(good_values) < 3 or len(bad_values) < 3:
                continue
            pooled_std = float(pd.concat([good_values, bad_values]).std(ddof=0))
            good_feature_ic = _feature_target_rank_ic(good, feature, target_column)
            bad_feature_ic = _feature_target_rank_ic(bad, feature, target_column)
            delta_feature_ic = (
                float(bad_feature_ic - good_feature_ic)
                if pd.notna(good_feature_ic) and pd.notna(bad_feature_ic)
                else np.nan
            )
            timeframe, family = classify_feature_column(feature)
            rows.append(
                {
                    "bad_fold": int(bad_fold),
                    "bad_fold_rank_ic": float(fold_rank_ic.get(bad_fold, np.nan)),
                    "feature": feature,
                    "timeframe": timeframe,
                    "family": family,
                    "good_feature_ic": good_feature_ic,
                    "bad_feature_ic": bad_feature_ic,
                    "delta_feature_ic_bad_minus_good": delta_feature_ic,
                    "signal_reversal": bool(
                        pd.notna(good_feature_ic)
                        and pd.notna(bad_feature_ic)
                        and abs(good_feature_ic) >= min_abs_reference_ic
                        and abs(bad_feature_ic) >= min_abs_reference_ic
                        and np.sign(good_feature_ic) != np.sign(bad_feature_ic)
                    ),
                    "good_mean": float(good_values.mean()),
                    "bad_mean": float(bad_values.mean()),
                    "mean_diff_bad_minus_good": float(bad_values.mean() - good_values.mean()),
                    "abs_standardized_diff": abs(float(bad_values.mean() - good_values.mean())) / pooled_std
                    if pooled_std > 0
                    else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["bad_fold", "signal_reversal", "abs_standardized_diff", "delta_feature_ic_bad_minus_good"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def bad_fold_group_forensics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
) -> pd.DataFrame:
    feature_frame = bad_fold_feature_forensics(
        predictions,
        fold_metrics,
        feature_columns=feature_columns,
        good_ic=good_ic,
        bad_ic=bad_ic,
    )
    if feature_frame.empty:
        return pd.DataFrame()

    rows = []
    for (bad_fold, bad_rank_ic, timeframe, family), part in feature_frame.groupby(
        ["bad_fold", "bad_fold_rank_ic", "timeframe", "family"],
        dropna=False,
    ):
        ranked_delta = part.reindex(part["delta_feature_ic_bad_minus_good"].abs().sort_values(ascending=False).index)
        shifted = part.sort_values("abs_standardized_diff", ascending=False)
        rows.append(
            {
                "bad_fold": int(bad_fold),
                "bad_fold_rank_ic": float(bad_rank_ic),
                "timeframe": timeframe,
                "family": family,
                "feature_count": int(part["feature"].nunique()),
                "mean_good_feature_ic": float(part["good_feature_ic"].mean()),
                "mean_bad_feature_ic": float(part["bad_feature_ic"].mean()),
                "mean_delta_feature_ic_bad_minus_good": float(part["delta_feature_ic_bad_minus_good"].mean()),
                "mean_abs_delta_feature_ic": float(part["delta_feature_ic_bad_minus_good"].abs().mean()),
                "signal_reversal_rate": float(part["signal_reversal"].mean()),
                "mean_abs_standardized_diff": float(part["abs_standardized_diff"].mean()),
                "top_delta_features": ",".join(ranked_delta["feature"].head(5).astype(str).tolist()),
                "top_shifted_features": ",".join(shifted["feature"].head(5).astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["bad_fold", "signal_reversal_rate", "mean_abs_delta_feature_ic", "mean_abs_standardized_diff"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def recent_fold_diagnostics(fold_metrics: pd.DataFrame, *, recent_folds: int = 5) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    ordered = fold_metrics.sort_values("fold")
    recent = ordered.tail(recent_folds)
    metrics = [
        "rank_ic",
        "long_f1",
        "prauc",
        "label_long_rate",
        "pred_long_rate_050",
        "prob_long_mean",
        "prob_long_std",
        "forward_return_mean",
    ]
    rows = []
    for metric in metrics:
        if metric not in ordered.columns:
            continue
        all_values = ordered[metric].replace([np.inf, -np.inf], np.nan).dropna()
        recent_values = recent[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if all_values.empty or recent_values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "all_mean": float(all_values.mean()),
                "recent_mean": float(recent_values.mean()),
                "recent_minus_all": float(recent_values.mean() - all_values.mean()),
                "recent_min": float(recent_values.min()),
                "recent_max": float(recent_values.max()),
                "recent_folds": ",".join(map(str, recent["fold"].astype(int).tolist())),
            }
        )
    return pd.DataFrame(rows)


def feature_group_diagnostics(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for position, feature in enumerate(feature_columns):
        timeframe, family = classify_feature_column(feature)
        rows.append({"position": position, "feature": feature, "timeframe": timeframe, "family": family})
    if not rows:
        return pd.DataFrame(columns=["position", "feature", "timeframe", "family", "count"])
    frame = pd.DataFrame(rows)
    counts = frame.groupby(["timeframe", "family"], as_index=False).agg(count=("feature", "size"))
    return frame.merge(counts, on=["timeframe", "family"], how="left")


def feature_profile_diagnostics(feature_columns: list[str], config: dict[str, Any] | None = None) -> pd.DataFrame:
    if config is None:
        return pd.DataFrame()
    profile = resolve_feature_profile(config)
    rows = [
        {
            "check": "active_feature_profile",
            "pattern": str(profile.get("name")),
            "matched_count": len(feature_columns),
            "matched_features": "",
        }
    ]
    for pattern in list(profile.get("include_patterns", []) or []):
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "profile_include_pattern",
                "pattern": pattern,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    for pattern in list(profile.get("exclude_patterns", []) or []):
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "profile_exclude_pattern_absent",
                "pattern": pattern,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    return pd.DataFrame(rows)


def feature_group_importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance is None or importance.empty or "feature" not in importance.columns:
        return pd.DataFrame()
    frame = importance.copy()
    groups = frame["feature"].map(classify_feature_column)
    frame["timeframe"] = [item[0] for item in groups]
    frame["family"] = [item[1] for item in groups]
    value_column = "rank_ic_drop"
    grouped = frame.groupby(["timeframe", "family"], as_index=False).agg(
        features=("feature", "nunique"),
        rows=("feature", "size"),
        mean_rank_ic_drop=(value_column, "mean"),
        median_rank_ic_drop=(value_column, "median"),
        min_rank_ic_drop=(value_column, "min"),
        max_rank_ic_drop=(value_column, "max"),
        positive_drop_rate=(value_column, lambda values: float((values > 0).mean())),
        total_positive_drop=(value_column, lambda values: float(values[values > 0].sum())),
    )
    return grouped.sort_values(["total_positive_drop", "mean_rank_ic_drop"], ascending=False).reset_index(drop=True)


def classify_feature_column(feature: str) -> tuple[str, str]:
    timeframe = "4h" if feature.startswith("4h_") else "1h"
    name = feature[3:] if timeframe == "4h" else feature
    if "_stable_" in name:
        base_name = name.split("_stable_", 1)[0]
        if any(token in base_name for token in ("log_return", "realized_vol", "gk_vol", "atr", "adx", "vwap", "denoised", "volume_log_zscore")):
            return timeframe, "volatility_structure_stable"
        return timeframe, "order_flow_v2_stable"
    if any(token in name for token in ("orderflow_efficiency", "absorption_pressure", "cvd_price_divergence", "large_trade_pressure", "cvd_pressure")):
        return timeframe, "order_flow_v2_raw"
    if any(token in name for token in ("taker_imbalance", "taker_buy_ratio_delta", "taker_buy_ratio_zscore")):
        return timeframe, "order_flow_v2_bounded"
    if any(token in name for token in ("true_cvd", "cvd_cumulative", "taker_buy_ratio", "taker_sell_ratio", "buy_sell_imbalance")):
        return timeframe, "order_flow_tier1"
    if any(token in name for token in ("vpt", "whale", "vol_per_trade", "large_trade_ratio")):
        return timeframe, "whale"
    if any(token in name for token in ("log_return", "realized_vol", "gk_vol", "atr", "adx", "vwap", "denoised")):
        return timeframe, "volatility_structure"
    return timeframe, "other"


def mtf_leakage_diagnostics(predictions: pd.DataFrame, *, htf_hours: int = 4) -> pd.DataFrame:
    required = {"timestamp", "4h_source_timestamp", "4h_available_timestamp"}
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        return pd.DataFrame([{"check": "mtf_alignment", "passed": False, "detail": f"missing columns: {missing}"}])

    timestamp = pd.to_datetime(predictions["timestamp"], utc=True)
    source = pd.to_datetime(predictions["4h_source_timestamp"], utc=True)
    available = pd.to_datetime(predictions["4h_available_timestamp"], utc=True)
    expected_available = source + pd.Timedelta(hours=htf_hours)
    availability_violations = int((available > timestamp).sum())
    shift_violations = int((available != expected_available).sum())
    rows = [
        {
            "check": "4h_available_lte_primary_timestamp",
            "passed": availability_violations == 0,
            "violations": availability_violations,
            "detail": "4H feature availability must never be after the 1H row timestamp.",
        },
        {
            "check": "4h_source_plus_period_equals_available",
            "passed": shift_violations == 0,
            "violations": shift_violations,
            "detail": f"4H source timestamps must be shifted forward by {htf_hours} hours.",
        },
    ]
    return pd.DataFrame(rows)


def stationarity_policy_diagnostics(feature_columns: list[str], config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Verify that configured nonstationary feature patterns are absent from model inputs."""

    patterns = []
    if config is not None:
        patterns = list(_config_get(config, ["features", "stationarity", "exclude_patterns"], []) or [])
    rows = []
    for pattern in patterns:
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "nonstationary_feature_excluded",
                "pattern": pattern,
                "passed": len(matches) == 0,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    raw_v2_matches = sorted(set(feature_columns) & raw_order_flow_v2_model_exclusions(config or {}))
    if raw_v2_matches:
        rows.append(
            {
                "check": "order_flow_v2_stable_only",
                "pattern": "<raw_order_flow_v2_exact_columns>",
                "passed": False,
                "matched_count": len(raw_v2_matches),
                "matched_features": ",".join(raw_v2_matches),
            }
        )
    elif config is not None and _config_get(config, ["features", "order_flow_v2", "stable_only"], False):
        rows.append(
            {
                "check": "order_flow_v2_stable_only",
                "pattern": "<raw_order_flow_v2_exact_columns>",
                "passed": True,
                "matched_count": 0,
                "matched_features": "",
            }
        )
    if rows:
        total_matches = sum(row["matched_count"] for row in rows)
        rows.append(
            {
                "check": "stationarity_policy_overall",
                "pattern": "<all>",
                "passed": total_matches == 0,
                "matched_count": total_matches,
                "matched_features": ",".join(
                    sorted(
                        {
                            feature
                            for row in rows
                            for feature in str(row["matched_features"]).split(",")
                            if feature
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=["check", "pattern", "passed", "matched_count", "matched_features"])


def model_feature_columns_frame(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"position": range(len(feature_columns)), "feature": list(feature_columns)})


def good_bad_feature_audit(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
    top_n: int = 30,
) -> pd.DataFrame:
    summary = good_bad_fold_summary(fold_metrics, good_ic=good_ic, bad_ic=bad_ic)
    good_folds = set(summary["good_folds"])
    bad_folds = set(summary["bad_folds"])
    if not good_folds or not bad_folds:
        return pd.DataFrame()

    good = predictions[predictions["fold"].isin(good_folds)]
    bad = predictions[predictions["fold"].isin(bad_folds)]
    rows = []
    for column in _diagnostic_feature_columns(predictions):
        good_values = good[column].replace([np.inf, -np.inf], np.nan).dropna()
        bad_values = bad[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(good_values) < 3 or len(bad_values) < 3:
            continue
        pooled_std = float(pd.concat([good_values, bad_values]).std(ddof=0))
        mean_diff = float(good_values.mean() - bad_values.mean())
        rows.append(
            {
                "feature": column,
                "good_mean": float(good_values.mean()),
                "bad_mean": float(bad_values.mean()),
                "mean_diff_good_minus_bad": mean_diff,
                "abs_standardized_diff": abs(mean_diff) / pooled_std if pooled_std > 0 else 0.0,
                "ks_stat": float(ks_2samp(good_values, bad_values).statistic),
                "good_folds": ",".join(map(str, sorted(good_folds))),
                "bad_folds": ",".join(map(str, sorted(bad_folds))),
            }
        )
    columns = [
        "feature",
        "good_mean",
        "bad_mean",
        "mean_diff_good_minus_bad",
        "abs_standardized_diff",
        "ks_stat",
        "good_folds",
        "bad_folds",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["ks_stat", "abs_standardized_diff"], ascending=False).head(top_n).reset_index(drop=True)


def write_phase1_diagnostic_bundle(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
    tsne: pd.DataFrame | None = None,
    calibrated_report: dict[str, Any] | None = None,
    calibrated_calibration: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
    threshold_metrics: pd.DataFrame | None = None,
    threshold_summary: pd.DataFrame | None = None,
    mtf_leakage: pd.DataFrame | None = None,
    feature_audit: pd.DataFrame | None = None,
    stationarity_policy: pd.DataFrame | None = None,
    score_lift: pd.DataFrame | None = None,
    score_lift_by_fold: pd.DataFrame | None = None,
    score_band_lift: pd.DataFrame | None = None,
    score_band_by_fold: pd.DataFrame | None = None,
    score_band_summary: pd.DataFrame | None = None,
    recent_fold_summary: pd.DataFrame | None = None,
    feature_groups: pd.DataFrame | None = None,
    feature_profile: pd.DataFrame | None = None,
    feature_group_importance: pd.DataFrame | None = None,
    group_permutation_importance: pd.DataFrame | None = None,
    bad_fold_feature_forensics_table: pd.DataFrame | None = None,
    bad_fold_group_forensics_table: pd.DataFrame | None = None,
    model_feature_columns: list[str] | None = None,
    config: dict[str, Any] | None = None,
    prefix: str = "phase1_diagnostics",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_dir = output_path / f"{prefix}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    serializable_report = _json_safe(report)
    (bundle_dir / "phase1_report.json").write_text(
        json.dumps(serializable_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if config is not None:
        (bundle_dir / "config.json").write_text(
            json.dumps(_json_safe(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (bundle_dir / "summary.md").write_text(
        _summary_markdown(serializable_report, fold_metrics),
        encoding="utf-8",
    )

    _write_parquet_with_csv_fallback(predictions, bundle_dir / "test_predictions.parquet")
    calibration.to_csv(bundle_dir / "calibration.csv", index=False)
    fold_metrics.to_csv(bundle_dir / "fold_metrics.csv", index=False)
    if model_feature_columns is not None:
        model_feature_columns_frame(model_feature_columns).to_csv(bundle_dir / "model_feature_columns.csv", index=False)
        if feature_groups is None:
            feature_groups = feature_group_diagnostics(model_feature_columns)
        if feature_profile is None:
            feature_profile = feature_profile_diagnostics(model_feature_columns, config)
        if stationarity_policy is None:
            stationarity_policy = stationarity_policy_diagnostics(model_feature_columns, config)
    if regime_metrics is not None and not regime_metrics.empty:
        regime_metrics.to_csv(bundle_dir / "regime_metrics.csv", index=False)
    if importance is not None and not importance.empty:
        importance.to_csv(bundle_dir / "permutation_importance.csv", index=False)
        if feature_group_importance is None:
            feature_group_importance = feature_group_importance_summary(importance)
    if tsne is not None and not tsne.empty:
        _write_parquet_with_csv_fallback(tsne, bundle_dir / "tsne_embeddings.parquet")
    if calibrated_report is not None:
        (bundle_dir / "calibrated_phase1_report.json").write_text(
            json.dumps(_json_safe(calibrated_report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if calibrated_calibration is not None and not calibrated_calibration.empty:
        calibrated_calibration.to_csv(bundle_dir / "calibrated_calibration.csv", index=False)
    if calibrated_predictions is not None and not calibrated_predictions.empty:
        _write_parquet_with_csv_fallback(calibrated_predictions, bundle_dir / "calibrated_test_predictions.parquet")
    if threshold_metrics is not None and not threshold_metrics.empty:
        threshold_metrics.to_csv(bundle_dir / "threshold_metrics.csv", index=False)
        if threshold_summary is None:
            threshold_summary = threshold_summary_diagnostics(threshold_metrics)
    if threshold_summary is not None and not threshold_summary.empty:
        threshold_summary.to_csv(bundle_dir / "threshold_summary.csv", index=False)
    if mtf_leakage is not None and not mtf_leakage.empty:
        mtf_leakage.to_csv(bundle_dir / "mtf_leakage.csv", index=False)
    if feature_audit is not None and not feature_audit.empty:
        feature_audit.to_csv(bundle_dir / "good_bad_feature_audit.csv", index=False)
    if stationarity_policy is not None and not stationarity_policy.empty:
        stationarity_policy.to_csv(bundle_dir / "stationarity_policy.csv", index=False)
    if score_lift is None and config is not None:
        bins = int(_config_get(config, ["validation", "calibration_bins"], 10))
        score_lift = score_lift_diagnostics(predictions, bins=bins)
    if score_lift is not None and not score_lift.empty:
        score_lift.to_csv(bundle_dir / "score_lift.csv", index=False)
    if score_lift_by_fold is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_lift_by_fold = score_lift_by_fold_diagnostics(predictions, bins=bins)
    if score_lift_by_fold is not None and not score_lift_by_fold.empty:
        score_lift_by_fold.to_csv(bundle_dir / "score_lift_by_fold.csv", index=False)
    score_bands = _config_get(config, ["validation", "score_bands"], None) if config is not None else None
    if score_band_lift is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_band_lift = score_band_diagnostics(predictions, bins=bins, bands=score_bands)
    if score_band_lift is not None and not score_band_lift.empty:
        score_band_lift.to_csv(bundle_dir / "score_band_lift.csv", index=False)
    if score_band_by_fold is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_band_by_fold = score_band_by_fold_diagnostics(predictions, bins=bins, bands=score_bands)
    if score_band_by_fold is not None and not score_band_by_fold.empty:
        score_band_by_fold.to_csv(bundle_dir / "score_band_by_fold.csv", index=False)
        if score_band_summary is None:
            score_band_summary = score_band_summary_diagnostics(score_band_by_fold)
    if score_band_summary is not None and not score_band_summary.empty:
        score_band_summary.to_csv(bundle_dir / "score_band_summary.csv", index=False)
    if recent_fold_summary is None and config is not None:
        recent_fold_count = int(_config_get(config, ["validation", "recent_folds"], 5))
        recent_fold_summary = recent_fold_diagnostics(fold_metrics, recent_folds=recent_fold_count)
    if recent_fold_summary is not None and not recent_fold_summary.empty:
        recent_fold_summary.to_csv(bundle_dir / "recent_fold_summary.csv", index=False)
    if feature_groups is not None and not feature_groups.empty:
        feature_groups.to_csv(bundle_dir / "feature_groups.csv", index=False)
    if feature_profile is not None and not feature_profile.empty:
        feature_profile.to_csv(bundle_dir / "feature_profile.csv", index=False)
    if feature_group_importance is not None and not feature_group_importance.empty:
        feature_group_importance.to_csv(bundle_dir / "feature_group_importance.csv", index=False)
    if group_permutation_importance is not None and not group_permutation_importance.empty:
        group_permutation_importance.to_csv(bundle_dir / "group_permutation_importance.csv", index=False)
    if bad_fold_feature_forensics_table is None:
        bad_fold_feature_forensics_table = bad_fold_feature_forensics(
            predictions,
            fold_metrics,
            feature_columns=model_feature_columns,
        )
    if bad_fold_feature_forensics_table is not None and not bad_fold_feature_forensics_table.empty:
        bad_fold_feature_forensics_table.to_csv(bundle_dir / "bad_fold_feature_forensics.csv", index=False)
    if bad_fold_group_forensics_table is None:
        bad_fold_group_forensics_table = bad_fold_group_forensics(
            predictions,
            fold_metrics,
            feature_columns=model_feature_columns,
        )
    if bad_fold_group_forensics_table is not None and not bad_fold_group_forensics_table.empty:
        bad_fold_group_forensics_table.to_csv(bundle_dir / "bad_fold_group_forensics.csv", index=False)

    fold_summary = good_bad_fold_summary(fold_metrics)
    (bundle_dir / "good_bad_folds.json").write_text(
        json.dumps(_json_safe(fold_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    zip_path = output_path / f"{bundle_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            archive.write(path, path.relative_to(bundle_dir))
    return zip_path


def _metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = scores.astype(float).to_numpy()
    y_pred = y_score >= threshold
    tp = int(((y_true == 1) & y_pred).sum())
    fp = int(((y_true == 0) & y_pred).sum())
    fn = int(((y_true == 1) & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()),
    }


def _assign_score_bins(predictions: pd.DataFrame, *, score_column: str, bins: int) -> pd.DataFrame:
    frame = predictions.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["label", score_column])
    if frame.empty:
        return frame
    frame["score_bin"] = pd.qcut(
        frame[score_column].rank(method="first"),
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    return frame


def _resolve_score_bands(bands: list[dict[str, Any]] | None, bins: int) -> list[dict[str, Any]]:
    max_bin = max(0, bins - 1)
    if bands is None:
        bands = [
            {"name": "top_10", "min_bin": max_bin, "max_bin": max_bin},
            {"name": "top_20", "min_bin": max(0, int(np.floor(bins * 0.80))), "max_bin": max_bin},
            {"name": "top_30", "min_bin": max(0, int(np.floor(bins * 0.70))), "max_bin": max_bin},
            {"name": "upper_half", "min_bin": max(0, int(np.floor(bins * 0.50))), "max_bin": max_bin},
            {
                "name": "mid_upper_40_90",
                "min_bin": max(0, int(np.floor(bins * 0.40))),
                "max_bin": max(0, max_bin - 1),
            },
        ]
    resolved = []
    for item in bands:
        name = str(item.get("name", f"bins_{item.get('min_bin')}_{item.get('max_bin')}"))
        min_bin = int(item.get("min_bin", max_bin))
        max_item_bin = int(item.get("max_bin", max_bin))
        min_bin = min(max(min_bin, 0), max_bin)
        max_item_bin = min(max(max_item_bin, 0), max_bin)
        if min_bin > max_item_bin:
            continue
        resolved.append({"name": name, "min_bin": min_bin, "max_bin": max_item_bin})
    return resolved


def _feature_target_rank_ic(frame: pd.DataFrame, feature: str, target_column: str) -> float:
    if feature not in frame.columns or target_column not in frame.columns:
        return np.nan
    pair = frame[[feature, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return np.nan
    return rank_ic(pair[feature], pair[target_column])


def _write_parquet_with_csv_fallback(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def _diagnostic_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = RAW_COLUMNS | METADATA_COLUMNS | LABEL_COLUMNS
    excluded |= {
        "fold",
        "source_row_position",
        "forward_return",
        "prob_long",
        "prob_long_raw",
        "prob_long_calibrated",
    }
    excluded_prefixes = ("regime_prob_", "pred_")
    columns = []
    for column in frame.columns:
        if column in excluded or any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(columns)


def _summary_markdown(report: dict[str, Any], fold_metrics: pd.DataFrame) -> str:
    top_good = fold_metrics.sort_values("rank_ic", ascending=False).head(5)[["fold", "rank_ic"]]
    top_bad = fold_metrics.sort_values("rank_ic", ascending=True).head(5)[["fold", "rank_ic"]]
    lines = [
        "# Phase 1 Diagnostics",
        "",
        f"Decision: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Mean Rank IC: {report.get('mean_rank_ic'):.6f}",
        f"Rank IC Std: {report.get('std_rank_ic'):.6f}",
        f"Positive IC Fraction: {report.get('positive_ic_fraction'):.6f}",
        f"Mean Long F1: {report.get('mean_long_f1'):.6f}",
        f"Mean PRAUC: {report.get('mean_prauc'):.6f}",
        f"Calibration Separation: {report.get('calibration_separation'):.6f}",
        "",
        "## Checks",
    ]
    checks = report.get("checks", {})
    lines.extend(f"- {name}: {value}" for name, value in checks.items())
    alerts = report.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("## Alerts")
        lines.extend(f"- {alert}" for alert in alerts)
    lines.append("")
    lines.append("## Best Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_good.itertuples())
    lines.append("")
    lines.append("## Worst Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_bad.itertuples())
    lines.append("")
    return "\n".join(lines)


def _config_get(config: object, path: list[str], default: object) -> object:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
