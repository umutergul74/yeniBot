"""Fold-level stability forensics and summaries."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
)

__all__ = [
    '_entry_official_threshold_source',
    '_entry_threshold_policy_frame',
    '_fold_stability_forensics_frame',
    '_fold_stability_summary_frame',
]

def _entry_official_threshold_source(entry: dict[str, Any]) -> str:
    row = (entry.get("diagnostics", {}) or {}).get("row", {}) or {}
    return str(row.get("official_threshold_source") or row.get("guarded_threshold_source") or "")

def _entry_threshold_policy_frame(entry: dict[str, Any]) -> pd.DataFrame:
    diagnostics = entry.get("diagnostics", {}) or {}
    threshold_metrics = diagnostics.get("threshold_metrics")
    if threshold_metrics is None or threshold_metrics.empty:
        return pd.DataFrame()

    frame = threshold_metrics.copy()
    calibrated = diagnostics.get("calibrated_threshold_metrics")
    if calibrated is not None and not calibrated.empty and "fold" in calibrated.columns:
        calibrated_keep = [
            column
            for column in (
                "fold",
                "selected_threshold",
                "test_f1_at_selected_threshold",
                "test_precision_at_selected_threshold",
                "test_recall_at_selected_threshold",
                "test_pred_long_rate_at_selected_threshold",
                "constrained_threshold",
                "source_constrained_f1",
                "source_constrained_precision",
                "source_constrained_recall",
                "source_constrained_pred_long_rate",
                "test_f1_at_constrained_threshold",
                "test_precision_at_constrained_threshold",
                "test_recall_at_constrained_threshold",
                "test_pred_long_rate_at_constrained_threshold",
            )
            if column in calibrated.columns
        ]
        calibrated_frame = calibrated[calibrated_keep].rename(
            columns={column: f"calibrated_{column}" for column in calibrated_keep if column != "fold"}
        )
        frame = frame.merge(calibrated_frame, on="fold", how="left")

    source = _entry_official_threshold_source(entry)
    use_calibrated = source.startswith("calibrated_")
    source_base = source.replace("calibrated_", "", 1) if use_calibrated else source
    if "selected" in source_base:
        family = "selected"
    elif "constrained" in source_base:
        family = "constrained"
    else:
        family = "constrained"
        if not source:
            source = "validation_constrained_threshold"
    prefix = "calibrated_" if use_calibrated else ""

    metric_map = {
        "threshold": f"{prefix}{family}_threshold",
        "f1": f"{prefix}test_f1_at_{family}_threshold",
        "precision": f"{prefix}test_precision_at_{family}_threshold",
        "recall": f"{prefix}test_recall_at_{family}_threshold",
        "pred_rate": f"{prefix}test_pred_long_rate_at_{family}_threshold",
    }

    def metric_series(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[column], errors="coerce")

    def official_metric_series(mapped_column: str, fallback_column: str) -> pd.Series:
        mapped = metric_series(mapped_column)
        if fallback_column in frame.columns:
            fallback = pd.to_numeric(frame[fallback_column], errors="coerce")
            mapped = mapped.where(mapped.notna(), fallback)
        return mapped

    frame["official_threshold_source"] = source
    frame["official_threshold_uses_calibration"] = bool(use_calibrated)
    frame["official_threshold"] = official_metric_series(metric_map["threshold"], "official_threshold")
    frame["test_f1_at_official_threshold"] = official_metric_series(
        metric_map["f1"], "test_f1_at_official_threshold"
    )
    frame["test_precision_at_official_threshold"] = official_metric_series(
        metric_map["precision"], "test_precision_at_official_threshold"
    )
    frame["test_recall_at_official_threshold"] = official_metric_series(
        metric_map["recall"], "test_recall_at_official_threshold"
    )
    frame["test_pred_long_rate_at_official_threshold"] = official_metric_series(
        metric_map["pred_rate"], "test_pred_long_rate_at_official_threshold"
    )
    return frame

def _fold_stability_forensics_frame(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "rank_ic",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_zscore",
        "rank_ic_abs_zscore",
        "rank_ic_variance_contribution",
        "rank_ic_std_driver_rank",
        "rank_ic_bucket",
        "long_f1_050",
        "test_f1_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "official_threshold_source",
        "official_threshold_uses_calibration",
        "test_f1_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "top_10_lift_vs_base",
        "top_10_forward_return",
        "primary_issue",
        "recommended_track",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    bad_ic = float(_cfg(config, ["validation", "bad_fold_ic_threshold"], -0.08))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        if fold_metrics is None or fold_metrics.empty:
            continue
        frame = fold_metrics.copy()
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is not None and not threshold_metrics.empty:
            merge_columns = [
                column
                for column in (
                    "fold",
                    "test_f1_at_selected_threshold",
                    "test_pred_long_rate_at_selected_threshold",
                    "test_f1_at_constrained_threshold",
                    "test_pred_long_rate_at_constrained_threshold",
                    "official_threshold_source",
                    "official_threshold_uses_calibration",
                    "test_f1_at_official_threshold",
                    "test_pred_long_rate_at_official_threshold",
                )
                if column in threshold_metrics.columns
            ]
            if "fold" in merge_columns:
                frame = frame.merge(threshold_metrics[merge_columns], on="fold", how="left")
        score_bands = diagnostics.get("score_band_by_fold")
        if score_bands is not None and not score_bands.empty and {"fold", "band"}.issubset(score_bands.columns):
            top_band = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            if not top_band.empty:
                rename = {
                    "lift_vs_base": "top_10_lift_vs_base",
                    "mean_forward_return": "top_10_forward_return",
                }
                keep = [column for column in ["fold", *rename.keys()] if column in top_band.columns]
                top_band = top_band[keep].rename(columns=rename)
                frame = frame.merge(top_band, on="fold", how="left")

        rank_values = pd.to_numeric(frame["rank_ic"], errors="coerce")
        mean_rank = float(rank_values.mean()) if rank_values.notna().any() else np.nan
        std_rank = float(rank_values.std(ddof=1)) if rank_values.notna().sum() > 1 else 0.0
        deviations = rank_values - mean_rank
        variance_total = float(np.square(deviations.dropna()).sum())
        frame["_rank_ic_mean"] = mean_rank
        frame["_rank_ic_std"] = std_rank
        frame["_rank_ic_zscore"] = deviations / std_rank if std_rank > 0 else np.nan
        frame["_rank_ic_variance_contribution"] = (
            np.square(deviations) / variance_total if variance_total > 0 else np.nan
        )
        frame["_rank_ic_std_driver_rank"] = (
            frame["_rank_ic_variance_contribution"].rank(method="first", ascending=False)
            if variance_total > 0
            else np.nan
        )

        candidate = str(entry.get("profile", ""))
        for _, item in frame.iterrows():
            row = item.to_dict()
            rank_ic = _float(row, "rank_ic")
            constrained_f1 = _float(row, "test_f1_at_constrained_threshold")
            constrained_rate = _float(row, "test_pred_long_rate_at_constrained_threshold")
            official_f1 = _float(row, "test_f1_at_official_threshold", constrained_f1)
            official_rate = _float(row, "test_pred_long_rate_at_official_threshold", constrained_rate)
            top_return = _float(row, "top_10_forward_return")
            zscore = _float(row, "_rank_ic_zscore")
            abs_zscore = abs(zscore) if np.isfinite(zscore) else np.nan
            if np.isfinite(rank_ic) and rank_ic <= bad_ic:
                bucket = "bad_rank_ic"
                issue = "bad_rank_ic"
                track = "fold_stability"
            elif np.isfinite(rank_ic) and rank_ic < 0:
                bucket = "negative_rank_ic"
                issue = "negative_rank_ic"
                track = "fold_stability"
            elif np.isfinite(rank_ic) and rank_ic < target_rank_ic:
                bucket = "below_target_rank_ic"
                issue = "below_target_rank_ic"
                track = "fold_stability"
            elif np.isfinite(abs_zscore) and abs_zscore >= 1.0:
                bucket = "variance_driver_high_side" if rank_ic >= mean_rank else "variance_driver_low_side"
                issue = bucket
                track = "fold_stability"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                bucket = "rank_ic_ok"
                issue = "official_threshold_f1"
                track = "threshold_calibration"
            elif np.isfinite(official_rate) and official_rate > max_pred_long_rate:
                bucket = "rank_ic_ok"
                issue = "official_threshold_pred_long_rate"
                track = "threshold_calibration"
            elif np.isfinite(top_return) and top_return <= 0:
                bucket = "rank_ic_ok"
                issue = "top_10_payoff"
                track = "score_band_policy"
            else:
                bucket = "rank_ic_ok"
                issue = "ok"
                track = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": _diagnostic_candidate_type(fold_scope),
                    "fold_scope": fold_scope,
                    "fold": int(row.get("fold")),
                    "start": str(row.get("start", "")),
                    "end": str(row.get("end", "")),
                    "rank_ic": rank_ic,
                    "rank_ic_mean": mean_rank,
                    "rank_ic_std": std_rank,
                    "rank_ic_zscore": zscore,
                    "rank_ic_abs_zscore": abs_zscore,
                    "rank_ic_variance_contribution": _float(row, "_rank_ic_variance_contribution"),
                    "rank_ic_std_driver_rank": _float(row, "_rank_ic_std_driver_rank"),
                    "rank_ic_bucket": bucket,
                    "long_f1_050": _float(row, "long_f1"),
                    "test_f1_at_selected_threshold": _float(row, "test_f1_at_selected_threshold"),
                    "test_pred_long_rate_at_selected_threshold": _float(row, "test_pred_long_rate_at_selected_threshold"),
                    "test_f1_at_constrained_threshold": constrained_f1,
                    "test_pred_long_rate_at_constrained_threshold": constrained_rate,
                    "official_threshold_source": str(row.get("official_threshold_source", "")),
                    "official_threshold_uses_calibration": bool(row.get("official_threshold_uses_calibration", False)),
                    "test_f1_at_official_threshold": official_f1,
                    "test_pred_long_rate_at_official_threshold": official_rate,
                    "top_10_lift_vs_base": _float(row, "top_10_lift_vs_base"),
                    "top_10_forward_return": top_return,
                    "primary_issue": issue,
                    "recommended_track": track,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "rank_ic_variance_contribution"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _fold_stability_summary_frame(forensics: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "rank_ic_mean",
        "rank_ic_std",
        "negative_fold_count",
        "bad_fold_count",
        "top_5_variance_contribution",
        "worst_fold",
        "worst_fold_rank_ic",
        "worst_fold_top_10_forward_return",
        "constrained_f1_fail_fold_rate",
        "constrained_pred_rate_fail_fold_rate",
        "official_f1_fail_fold_rate",
        "official_pred_rate_fail_fold_rate",
        "top_10_payoff_fail_fold_rate",
        "main_blocker",
    ]
    if forensics.empty:
        return pd.DataFrame(columns=columns)
    min_long_f1 = float(_cfg(config or {}, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config or {}, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    rows = []
    for (candidate, candidate_type, fold_scope), part in forensics.groupby(["candidate", "candidate_type", "fold_scope"]):
        sorted_var = part.sort_values("rank_ic_variance_contribution", ascending=False)
        worst = part.sort_values("rank_ic", ascending=True).iloc[0].to_dict()
        constrained_f1_fail = pd.to_numeric(part["test_f1_at_constrained_threshold"], errors="coerce") < min_long_f1
        constrained_rate_fail = pd.to_numeric(part["test_pred_long_rate_at_constrained_threshold"], errors="coerce") > max_pred_long_rate
        official_f1_fail = pd.to_numeric(part["test_f1_at_official_threshold"], errors="coerce") < min_long_f1
        official_rate_fail = pd.to_numeric(part["test_pred_long_rate_at_official_threshold"], errors="coerce") > max_pred_long_rate
        top_payoff_fail = pd.to_numeric(part["top_10_forward_return"], errors="coerce") <= 0.0
        issue_counts = part["primary_issue"].value_counts()
        main_blocker = str(issue_counts.index[0]) if not issue_counts.empty else ""
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "rank_ic_mean": float(pd.to_numeric(part["rank_ic"], errors="coerce").mean()),
                "rank_ic_std": float(pd.to_numeric(part["rank_ic"], errors="coerce").std(ddof=1)),
                "negative_fold_count": int((pd.to_numeric(part["rank_ic"], errors="coerce") < 0.0).sum()),
                "bad_fold_count": int((part["rank_ic_bucket"].astype(str) == "bad_rank_ic").sum()),
                "top_5_variance_contribution": float(
                    pd.to_numeric(sorted_var["rank_ic_variance_contribution"], errors="coerce").head(5).sum()
                ),
                "worst_fold": int(worst.get("fold")),
                "worst_fold_rank_ic": _float(worst, "rank_ic"),
                "worst_fold_top_10_forward_return": _float(worst, "top_10_forward_return"),
                "constrained_f1_fail_fold_rate": float(constrained_f1_fail.mean()),
                "constrained_pred_rate_fail_fold_rate": float(constrained_rate_fail.mean()),
                "official_f1_fail_fold_rate": float(official_f1_fail.mean()),
                "official_pred_rate_fail_fold_rate": float(official_rate_fail.mean()),
                "top_10_payoff_fail_fold_rate": float(top_payoff_fail.mean()),
                "main_blocker": main_blocker,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["candidate_type", "rank_ic_std"], ascending=[True, True]).reset_index(drop=True)
