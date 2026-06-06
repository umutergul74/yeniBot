"""Feature, score, probability, and reliability drift diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from yenibot.experiment.common import (
    _cfg,
    _clean_probability_inputs,
    _diagnostic_candidate_type,
    _float,
    _is_stability_scope,
    _mean_for_mask,
    _numeric_mean,
    _optional_float,
    _rank_ic_for_frame,
    _safe_average_precision,
    _score_ks_statistic,
    _table_markdown,
    _write_json,
)

from yenibot.experiment.folds import (
    _entry_threshold_policy_frame,
)

from yenibot.experiment.separation import (
    _score_separation_markdown,
)

from yenibot.experiment.training import (
    _test_predictions,
)

__all__ = [
    '_write_score_separation_forensics',
    '_feature_family',
    '_safe_spearman',
    '_label_gap',
    '_feature_drift_columns',
    '_feature_drift_forensics_frame',
    '_feature_family_drift_summary_frame',
    '_feature_drift_markdown',
    '_write_feature_drift_forensics',
    '_score_reversal_context_audit_frame',
    '_score_reversal_context_audit_markdown',
    '_write_score_reversal_context_audit',
    '_calibration_error',
    '_binary_probability_metrics',
    '_probability_quality_forensics_frame',
    '_probability_quality_summary_frame',
    '_population_stability_index',
    '_score_distribution_shift_frame',
    '_score_distribution_shift_summary_frame',
    '_validation_reliability_metrics',
    '_reliability_gate_definitions',
    '_gate_threshold_check',
    '_fold_reliability_gate_passed',
    '_fold_reliability_gate_frame',
    '_fold_reliability_gate_summary_frame',
    '_fold_reliability_gate_markdown',
    '_probability_quality_markdown',
    '_score_distribution_shift_markdown',
    '_write_probability_quality_forensics',
    '_write_score_distribution_shift',
    '_write_fold_reliability_gate',
    '_forensics_markdown',
    '_write_forensics_reports',
]

def _write_score_separation_forensics(
    path: Path,
    score_forensics: pd.DataFrame,
    bad_signature: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    score_forensics.to_csv(path / "score_separation_forensics.csv", index=False)
    bad_signature.to_csv(path / "bad_fold_signature.csv", index=False)
    (path / "bad_fold_signature.md").write_text(
        _score_separation_markdown(score_forensics, bad_signature),
        encoding="utf-8",
    )
    _write_json(
        path / "bad_fold_signature.json",
        {
            "score_separation_forensics": score_forensics.to_dict(orient="records"),
            "bad_fold_signature": bad_signature.to_dict(orient="records"),
        },
    )

def _feature_family(feature: str) -> str:
    name = str(feature)
    timeframe = "4h" if name.startswith("4h_") else "1h"
    base = name[3:] if timeframe == "4h" else name
    if base.startswith("ih15_"):
        return "ih15_intrahour_order_flow"
    if base.startswith("fut_") or any(token in base for token in ("funding", "open_interest", "positioning")):
        return "futures_context"
    if "large_trade_pressure" in base or "signed_ltp" in base or base.startswith("ltp"):
        family = "large_trade_pressure"
    elif "cvd_pressure" in base:
        family = "cvd_pressure"
    elif any(token in base for token in ("taker", "imbalance", "cvd", "orderflow", "absorption", "pressure")):
        family = "order_flow"
    elif any(token in base for token in ("whale", "vpt", "vol_per_trade", "large_trade_ratio")):
        family = "whale_ticket_size"
    elif "volume" in base or "trade_share" in base:
        family = "volume_context"
    elif any(token in base for token in ("realized_vol", "gk_vol", "atr", "adx", "vwap", "return")):
        family = "volatility_structure"
    else:
        family = "other"
    return f"{timeframe}_{family}" if timeframe == "4h" else family

def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4:
        return np.nan
    if frame["left"].nunique(dropna=True) < 2 or frame["right"].nunique(dropna=True) < 2:
        return np.nan
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if np.isfinite(value) else np.nan

def _label_gap(frame: pd.DataFrame, feature: str) -> float:
    if frame.empty or feature not in frame.columns or "label" not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    pos = values.loc[labels == 1].dropna()
    neg = values.loc[labels == 0].dropna()
    if pos.empty or neg.empty:
        return np.nan
    return float(pos.mean() - neg.mean())

def _feature_drift_columns() -> list[str]:
    return [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature",
        "feature_family",
        "bad_definition",
        "bad_fold_count",
        "good_fold_count",
        "bad_fold_ids",
        "good_fold_ids",
        "bad_count",
        "good_count",
        "bad_mean",
        "good_mean",
        "drift_effect_size",
        "bad_std",
        "good_std",
        "bad_feature_return_ic",
        "good_feature_return_ic",
        "return_ic_delta_bad_minus_good",
        "bad_label_gap",
        "good_label_gap",
        "label_gap_delta_bad_minus_good",
        "return_ic_reversal",
        "label_gap_reversal",
        "distribution_drift_flag",
        "suspect_score",
        "likely_issue",
    ]

def _feature_drift_forensics_frame(
    entries: list[dict[str, Any]],
    score_forensics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = _feature_drift_columns()
    if score_forensics.empty:
        return pd.DataFrame(columns=columns)

    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    score_groups = {
        (str(candidate), str(fold_scope)): part.copy()
        for (candidate, _candidate_type, fold_scope), part in score_forensics.groupby(
            ["candidate", "candidate_type", "fold_scope"],
            dropna=False,
        )
    }

    for entry in entries:
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        if candidate_type != "profile" or fold_scope != "full":
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        feature_columns = [str(column) for column in entry.get("feature_columns", [])]
        usable_features = [
            column
            for column in feature_columns
            if column in predictions.columns and pd.api.types.is_numeric_dtype(predictions[column])
        ]
        if not usable_features:
            continue
        test_predictions = _test_predictions(predictions)
        required = {"fold", "label", "forward_return"}
        if test_predictions.empty or not required.issubset(test_predictions.columns):
            continue
        score_part = score_groups.get((candidate, fold_scope))
        if score_part is None or score_part.empty or "rank_ic" not in score_part.columns:
            continue
        rank_values = pd.to_numeric(score_part["rank_ic"], errors="coerce")
        bad_score = score_part.loc[rank_values < 0.0].copy()
        bad_definition = "negative_rank_ic"
        if bad_score.empty:
            bad_score = score_part.loc[rank_values < target_rank_ic].copy()
            bad_definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        good_score = score_part.loc[rank_values >= target_rank_ic].copy()
        bad_folds = sorted({int(fold) for fold in bad_score["fold"].dropna().astype(int).tolist()})
        good_folds = sorted({int(fold) for fold in good_score["fold"].dropna().astype(int).tolist()})
        if not bad_folds or not good_folds:
            continue
        frame = test_predictions.copy().replace([np.inf, -np.inf], np.nan)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
        frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        bad_frame = frame.loc[frame["fold"].isin(bad_folds)].copy()
        good_frame = frame.loc[frame["fold"].isin(good_folds)].copy()
        if bad_frame.empty or good_frame.empty:
            continue

        for feature in usable_features:
            bad_values = pd.to_numeric(bad_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            good_values = pd.to_numeric(good_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if bad_values.empty or good_values.empty:
                continue
            bad_mean = float(bad_values.mean())
            good_mean = float(good_values.mean())
            bad_std = float(bad_values.std(ddof=0))
            good_std = float(good_values.std(ddof=0))
            pooled_std = float(np.sqrt((bad_std**2 + good_std**2) / 2.0))
            drift_effect = float((bad_mean - good_mean) / pooled_std) if pooled_std > 0 else np.nan
            bad_return_ic = _safe_spearman(bad_frame[feature], bad_frame["forward_return"])
            good_return_ic = _safe_spearman(good_frame[feature], good_frame["forward_return"])
            return_ic_delta = (
                bad_return_ic - good_return_ic
                if np.isfinite(bad_return_ic) and np.isfinite(good_return_ic)
                else np.nan
            )
            bad_gap = _label_gap(bad_frame, feature)
            good_gap = _label_gap(good_frame, feature)
            label_gap_delta = bad_gap - good_gap if np.isfinite(bad_gap) and np.isfinite(good_gap) else np.nan
            return_reversal = bool(
                np.isfinite(bad_return_ic)
                and np.isfinite(good_return_ic)
                and bad_return_ic * good_return_ic < 0
                and abs(bad_return_ic) >= 0.02
                and abs(good_return_ic) >= 0.02
            )
            label_reversal = bool(
                np.isfinite(bad_gap)
                and np.isfinite(good_gap)
                and bad_gap * good_gap < 0
                and abs(bad_gap) >= 0.03
                and abs(good_gap) >= 0.03
            )
            distribution_drift = bool(np.isfinite(drift_effect) and abs(drift_effect) >= 0.50)
            suspect_score = 0.0
            if np.isfinite(drift_effect):
                suspect_score += min(abs(drift_effect), 3.0)
            if np.isfinite(return_ic_delta):
                suspect_score += min(abs(return_ic_delta), 2.0)
            if np.isfinite(label_gap_delta):
                suspect_score += min(abs(label_gap_delta), 2.0) * 0.5
            if return_reversal:
                suspect_score += 1.0
            if label_reversal:
                suspect_score += 0.75
            if distribution_drift:
                suspect_score += 0.5

            if return_reversal and label_reversal:
                issue = "feature_signal_and_label_gap_reversal"
            elif return_reversal:
                issue = "feature_return_ic_reversal"
            elif label_reversal:
                issue = "feature_label_gap_reversal"
            elif distribution_drift:
                issue = "bad_fold_distribution_drift"
            elif np.isfinite(return_ic_delta) and return_ic_delta < -0.08:
                issue = "bad_fold_feature_return_ic_degrades"
            else:
                issue = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "bad_definition": bad_definition,
                    "bad_fold_count": len(bad_folds),
                    "good_fold_count": len(good_folds),
                    "bad_fold_ids": ",".join(str(fold) for fold in bad_folds),
                    "good_fold_ids": ",".join(str(fold) for fold in good_folds),
                    "bad_count": int(len(bad_values)),
                    "good_count": int(len(good_values)),
                    "bad_mean": bad_mean,
                    "good_mean": good_mean,
                    "drift_effect_size": drift_effect,
                    "bad_std": bad_std,
                    "good_std": good_std,
                    "bad_feature_return_ic": bad_return_ic,
                    "good_feature_return_ic": good_return_ic,
                    "return_ic_delta_bad_minus_good": return_ic_delta,
                    "bad_label_gap": bad_gap,
                    "good_label_gap": good_gap,
                    "label_gap_delta_bad_minus_good": label_gap_delta,
                    "return_ic_reversal": return_reversal,
                    "label_gap_reversal": label_reversal,
                    "distribution_drift_flag": distribution_drift,
                    "suspect_score": float(suspect_score),
                    "likely_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate", "fold_scope", "suspect_score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _feature_family_drift_summary_frame(feature_drift: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature_family",
        "feature_count",
        "top_suspect_feature",
        "top_suspect_score",
        "mean_abs_drift_effect",
        "mean_bad_return_ic",
        "mean_good_return_ic",
        "return_ic_reversal_count",
        "label_gap_reversal_count",
        "distribution_drift_count",
        "top_likely_issue",
        "recommended_next_action",
    ]
    if feature_drift.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, family), part in feature_drift.groupby(
        ["candidate", "candidate_type", "fold_scope", "feature_family"],
        dropna=False,
    ):
        sorted_part = part.sort_values("suspect_score", ascending=False)
        top = sorted_part.iloc[0].to_dict()
        reversal_count = int(part["return_ic_reversal"].astype(bool).sum())
        label_reversal_count = int(part["label_gap_reversal"].astype(bool).sum())
        drift_count = int(part["distribution_drift_flag"].astype(bool).sum())
        if reversal_count > 0 or label_reversal_count > 0:
            action = "inspect_or_ablate_family_in_pre_registered_future_oos_candidate"
        elif drift_count > 0:
            action = "prefer_stable_bounded_transforms_or_family_ablation_hypothesis"
        else:
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "feature_family": family,
                "feature_count": int(part["feature"].nunique()),
                "top_suspect_feature": str(top.get("feature", "")),
                "top_suspect_score": _float(top, "suspect_score"),
                "mean_abs_drift_effect": float(pd.to_numeric(part["drift_effect_size"], errors="coerce").abs().mean()),
                "mean_bad_return_ic": float(pd.to_numeric(part["bad_feature_return_ic"], errors="coerce").mean()),
                "mean_good_return_ic": float(pd.to_numeric(part["good_feature_return_ic"], errors="coerce").mean()),
                "return_ic_reversal_count": reversal_count,
                "label_gap_reversal_count": label_reversal_count,
                "distribution_drift_count": drift_count,
                "top_likely_issue": str(top.get("likely_issue", "")),
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["candidate", "top_suspect_score", "return_ic_reversal_count", "label_gap_reversal_count"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

def _feature_drift_markdown(detail: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = ["# Bad Fold Feature Drift Forensics", ""]
    if detail.empty and summary.empty:
        lines.append("No feature drift rows were produced.")
        return "\n".join(lines)
    lines.append(
        "These diagnostics compare bad folds against good folds using existing OOS predictions. "
        "They identify distribution drift and feature/return or feature/label signal reversals. "
        "They are diagnostic evidence, not automatic feature-selection output."
    )
    if not summary.empty:
        lines.append("")
        lines.append("## Feature Family Summary")
        display_cols = [
            "candidate",
            "fold_scope",
            "feature_family",
            "feature_count",
            "top_suspect_feature",
            "top_suspect_score",
            "return_ic_reversal_count",
            "label_gap_reversal_count",
            "distribution_drift_count",
            "recommended_next_action",
        ]
        visible = summary[[column for column in display_cols if column in summary.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not detail.empty:
        lines.append("")
        lines.append(f"Feature-level rows: {len(detail)}. See `feature_drift_forensics.csv` for detail.")
    return "\n".join(lines)

def _write_feature_drift_forensics(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "feature_drift_forensics.csv", index=False)
    summary.to_csv(path / "feature_family_drift_summary.csv", index=False)
    (path / "feature_drift_forensics.md").write_text(
        _feature_drift_markdown(detail, summary),
        encoding="utf-8",
    )
    _write_json(
        path / "feature_drift_forensics.json",
        {
            "feature_drift_forensics": detail.to_dict(orient="records"),
            "feature_family_drift_summary": summary.to_dict(orient="records"),
        },
    )

def _score_reversal_context_audit_frame(
    feature_drift: pd.DataFrame,
    historical_memory_audit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "profile",
        "suspect_feature",
        "context_feature",
        "mechanism",
        "suspect_score",
        "suspect_issue",
        "historical_status",
        "related_rejected_profile_count",
        "prior_failed_profiles",
        "why_not_repeat",
        "requires_02_03",
        "requires_04",
        "promotion_allowed_now",
        "recommended_action",
    ]
    cfg = _cfg(config, ["validation", "score_reversal_context"], {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(columns=columns)
    proposals = cfg.get("proposed_profiles", []) or []
    if not proposals:
        return pd.DataFrame(columns=columns)

    drift_by_feature = {}
    if not feature_drift.empty and "feature" in feature_drift.columns:
        for _, row in feature_drift.sort_values("suspect_score", ascending=False).iterrows():
            feature = str(row.get("feature", ""))
            if feature and feature not in drift_by_feature:
                drift_by_feature[feature] = row.to_dict()

    memory_by_feature = {}
    if not historical_memory_audit.empty and "suspect_feature" in historical_memory_audit.columns:
        for _, row in historical_memory_audit.iterrows():
            feature = str(row.get("suspect_feature", ""))
            if feature and feature not in memory_by_feature:
                memory_by_feature[feature] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        profile = str(proposal.get("profile", ""))
        suspect = str(proposal.get("suspect_feature", ""))
        context = str(proposal.get("context_feature", ""))
        drift = drift_by_feature.get(suspect, {})
        memory = memory_by_feature.get(suspect, {})
        related_count = _float(memory, "related_rejected_profile_count")
        rows.append(
            {
                "profile": profile,
                "suspect_feature": suspect,
                "context_feature": context,
                "mechanism": str(proposal.get("mechanism", "")),
                "suspect_score": _float(drift, "suspect_score"),
                "suspect_issue": str(drift.get("likely_issue", "")),
                "historical_status": str(memory.get("historical_status", memory.get("memory_status", ""))),
                "related_rejected_profile_count": int(related_count) if np.isfinite(related_count) else 0,
                "prior_failed_profiles": ",".join(str(item) for item in proposal.get("prior_failed_profiles", []) or []),
                "why_not_repeat": str(proposal.get("why_not_repeat", "")),
                "requires_02_03": True,
                "requires_04": True,
                "promotion_allowed_now": False,
                "recommended_action": (
                    "pre_registered_future_oos_candidate; run 02/03/04 only to create CV predictions, "
                    "then wait for future unseen OOS before any promotion"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _score_reversal_context_audit_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Score-Reversal Context Audit", ""]
    lines.append(
        "This report records pre-registered feature hypotheses for the current score-reversal blocker. "
        "It is not a promotion report and it must not be tuned against the frozen holdout."
    )
    if frame.empty:
        lines.extend(["", "No score-reversal context hypotheses are configured."])
        return "\n".join(lines)
    display_cols = [
        "profile",
        "suspect_feature",
        "context_feature",
        "mechanism",
        "suspect_score",
        "historical_status",
        "why_not_repeat",
        "requires_02_03",
        "requires_04",
        "recommended_action",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_score_reversal_context_audit(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "score_reversal_context_audit.csv", index=False)
    (path / "score_reversal_context_audit.md").write_text(
        _score_reversal_context_audit_markdown(frame),
        encoding="utf-8",
    )
    _write_json(path / "score_reversal_context_audit.json", {"rows": frame.to_dict(orient="records")})

def _calibration_error(labels: pd.Series, scores: pd.Series, *, bins: int, strategy: str) -> tuple[float, float, int]:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty:
        return np.nan, np.nan, 0
    bins = max(2, int(bins))
    if strategy == "equal_count":
        try:
            frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=min(bins, len(frame)), labels=False, duplicates="drop")
        except ValueError:
            return np.nan, np.nan, 0
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        frame["bin"] = pd.cut(frame["score"], bins=edges, labels=False, include_lowest=True)
    frame = frame.dropna(subset=["bin"]).copy()
    if frame.empty:
        return np.nan, np.nan, 0
    total = float(len(frame))
    weighted_error = 0.0
    max_error = 0.0
    used_bins = 0
    for _, part in frame.groupby("bin"):
        if part.empty:
            continue
        predicted = float(part["score"].mean())
        actual = float(part["label"].mean())
        error = abs(actual - predicted)
        weighted_error += (len(part) / total) * error
        max_error = max(max_error, error)
        used_bins += 1
    return float(weighted_error), float(max_error), int(used_bins)

def _binary_probability_metrics(labels: pd.Series, scores: pd.Series, *, bins: int) -> dict[str, float]:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty:
        return {
            "brier_score": np.nan,
            "log_loss": np.nan,
            "average_precision": np.nan,
            "ece_equal_width": np.nan,
            "mce_equal_width": np.nan,
            "ece_equal_count": np.nan,
            "mce_equal_count": np.nan,
            "score_entropy_mean": np.nan,
            "score_sharpness_mean": np.nan,
            "prob_long_mean": np.nan,
            "prob_long_std": np.nan,
            "prob_long_iqr": np.nan,
        }
    labels_array = frame["label"].astype(float)
    scores_array = frame["score"].astype(float)
    ece_width, mce_width, _ = _calibration_error(labels_array, scores_array, bins=bins, strategy="equal_width")
    ece_count, mce_count, _ = _calibration_error(labels_array, scores_array, bins=bins, strategy="equal_count")
    entropy = -(scores_array * np.log(scores_array) + (1.0 - scores_array) * np.log(1.0 - scores_array))
    return {
        "brier_score": float(np.mean((scores_array - labels_array) ** 2)),
        "log_loss": float(-np.mean(labels_array * np.log(scores_array) + (1.0 - labels_array) * np.log(1.0 - scores_array))),
        "average_precision": _safe_average_precision(labels_array, scores_array),
        "ece_equal_width": ece_width,
        "mce_equal_width": mce_width,
        "ece_equal_count": ece_count,
        "mce_equal_count": mce_count,
        "score_entropy_mean": float(entropy.mean()),
        "score_sharpness_mean": float((scores_array * (1.0 - scores_array)).mean()),
        "prob_long_mean": float(scores_array.mean()),
        "prob_long_std": float(scores_array.std(ddof=0)),
        "prob_long_iqr": float(scores_array.quantile(0.75) - scores_array.quantile(0.25)),
    }

def _probability_quality_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "count",
        "label_long_rate",
        "rank_ic",
        "rank_ic_bucket",
        "brier_score",
        "log_loss",
        "average_precision",
        "ece_equal_width",
        "mce_equal_width",
        "ece_equal_count",
        "mce_equal_count",
        "score_entropy_mean",
        "score_sharpness_mean",
        "prob_long_mean",
        "prob_long_std",
        "prob_long_iqr",
        "official_threshold",
        "official_f1",
        "official_pred_long_rate",
        "primary_issue",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    bins = int(_cfg(config, ["validation", "calibration_bins"], 10))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty or not {"fold", "label", "prob_long"}.issubset(test_predictions.columns):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, part in test_predictions.groupby("fold"):
            fold_id = int(fold)
            part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if part.empty:
                continue
            rank_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            rank_ic_value = _float(rank_row, "rank_ic")
            metrics = _binary_probability_metrics(part["label"], part["prob_long"], bins=bins)
            official_f1 = _float(threshold_row, "test_f1_at_official_threshold", _float(threshold_row, "test_f1_at_constrained_threshold"))
            official_rate = _float(
                threshold_row,
                "test_pred_long_rate_at_official_threshold",
                _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
            )
            if np.isfinite(rank_ic_value) and rank_ic_value < 0.0:
                issue = "negative_rank_ic"
            elif np.isfinite(metrics["ece_equal_count"]) and metrics["ece_equal_count"] > 0.10:
                issue = "calibration_error"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                issue = "official_f1_gap"
            elif np.isfinite(official_rate) and official_rate > max_pred_rate:
                issue = "pred_long_rate_guardrail"
            else:
                issue = "ok"
            bucket = (
                "negative_rank_ic"
                if np.isfinite(rank_ic_value) and rank_ic_value < 0.0
                else ("below_target_rank_ic" if np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic else "rank_ic_ok")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "start": str(part["timestamp"].min()) if "timestamp" in part.columns else str(rank_row.get("start", "")),
                    "end": str(part["timestamp"].max()) if "timestamp" in part.columns else str(rank_row.get("end", "")),
                    "count": int(len(part)),
                    "label_long_rate": float(pd.to_numeric(part["label"], errors="coerce").mean()),
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    **metrics,
                    "official_threshold": _float(threshold_row, "official_threshold"),
                    "official_f1": official_f1,
                    "official_pred_long_rate": official_rate,
                    "primary_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "rank_ic"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )

def _probability_quality_summary_frame(probability_quality: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "bad_definition",
        "bad_fold_count",
        "good_fold_count",
        "mean_brier_score",
        "mean_log_loss",
        "mean_average_precision",
        "mean_ece_equal_count",
        "mean_score_entropy",
        "bad_brier_score_mean",
        "good_brier_score_mean",
        "bad_average_precision_mean",
        "good_average_precision_mean",
        "bad_ece_equal_count_mean",
        "good_ece_equal_count_mean",
        "bad_score_entropy_mean",
        "good_score_entropy_mean",
        "probability_quality_issue",
        "recommended_next_action",
    ]
    if probability_quality.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in probability_quality.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank < 0.0
        bad_definition = "negative_rank_ic"
        if int(bad_mask.sum()) < 1:
            bad_mask = rank < target_rank_ic
            bad_definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        good_mask = rank >= target_rank_ic
        bad_count = int(bad_mask.sum())
        good_count = int(good_mask.sum())
        bad_brier = _mean_for_mask(part, bad_mask, "brier_score")
        good_brier = _mean_for_mask(part, good_mask, "brier_score")
        bad_ap = _mean_for_mask(part, bad_mask, "average_precision")
        good_ap = _mean_for_mask(part, good_mask, "average_precision")
        bad_ece = _mean_for_mask(part, bad_mask, "ece_equal_count")
        good_ece = _mean_for_mask(part, good_mask, "ece_equal_count")
        bad_entropy = _mean_for_mask(part, bad_mask, "score_entropy_mean")
        good_entropy = _mean_for_mask(part, good_mask, "score_entropy_mean")
        if np.isfinite(bad_ap) and np.isfinite(good_ap) and bad_ap < good_ap - 0.05:
            issue = "bad_folds_lose_ranking_resolution"
            action = "prioritize_score_separation_features_over_threshold_smoothing"
        elif np.isfinite(bad_ece) and np.isfinite(good_ece) and bad_ece > good_ece + 0.03:
            issue = "bad_folds_calibration_worsens"
            action = "review_calibration_by_fold_but_do_not_fit_on_test_or_holdout"
        elif np.isfinite(bad_entropy) and np.isfinite(good_entropy) and bad_entropy > good_entropy + 0.03:
            issue = "bad_folds_scores_become_uncertain"
            action = "inspect_feature_drift_and_score_distribution_shift"
        else:
            issue = "monitor"
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "bad_definition": bad_definition,
                "bad_fold_count": bad_count,
                "good_fold_count": good_count,
                "mean_brier_score": float(pd.to_numeric(part["brier_score"], errors="coerce").mean()),
                "mean_log_loss": float(pd.to_numeric(part["log_loss"], errors="coerce").mean()),
                "mean_average_precision": float(pd.to_numeric(part["average_precision"], errors="coerce").mean()),
                "mean_ece_equal_count": float(pd.to_numeric(part["ece_equal_count"], errors="coerce").mean()),
                "mean_score_entropy": float(pd.to_numeric(part["score_entropy_mean"], errors="coerce").mean()),
                "bad_brier_score_mean": bad_brier,
                "good_brier_score_mean": good_brier,
                "bad_average_precision_mean": bad_ap,
                "good_average_precision_mean": good_ap,
                "bad_ece_equal_count_mean": bad_ece,
                "good_ece_equal_count_mean": good_ece,
                "bad_score_entropy_mean": bad_entropy,
                "good_score_entropy_mean": good_entropy,
                "probability_quality_issue": issue,
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "mean_average_precision"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _population_stability_index(actual: pd.Series, expected: pd.Series, *, bins: int) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    expected_values = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(actual_values) == 0 or len(expected_values) == 0:
        return np.nan
    bins = max(2, int(bins))
    try:
        edges = np.unique(np.quantile(expected_values, np.linspace(0.0, 1.0, bins + 1)))
    except ValueError:
        return np.nan
    if len(edges) < 3:
        low = min(float(np.min(actual_values)), float(np.min(expected_values)))
        high = max(float(np.max(actual_values)), float(np.max(expected_values)))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            return 0.0
        edges = np.linspace(low, high, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    epsilon = 1e-6
    actual_pct = np.maximum(actual_counts / max(1, actual_counts.sum()), epsilon)
    expected_pct = np.maximum(expected_counts / max(1, expected_counts.sum()), epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def _score_distribution_shift_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "rank_ic",
        "rank_ic_bucket",
        "reference_definition",
        "reference_fold_count",
        "score_ks_vs_reference",
        "score_psi_vs_reference",
        "prob_long_mean",
        "reference_prob_long_mean",
        "prob_long_mean_delta",
        "prob_long_std",
        "reference_prob_long_std",
        "prob_long_std_ratio",
        "prob_long_iqr",
        "reference_prob_long_iqr",
        "score_entropy_mean",
        "reference_score_entropy_mean",
        "score_entropy_delta",
        "score_shift_issue",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    bins = int(_cfg(config, ["validation", "score_lift_bins"], 10))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty or not {"fold", "prob_long"}.issubset(test_predictions.columns):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        rank_by_id = {fold: _float(row, "rank_ic") for fold, row in fold_by_id.items()}
        all_folds = sorted({int(fold) for fold in pd.to_numeric(test_predictions["fold"], errors="coerce").dropna().astype(int)})
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold in all_folds:
            part = test_predictions.loc[pd.to_numeric(test_predictions["fold"], errors="coerce") == fold].copy()
            scores = pd.to_numeric(part["prob_long"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if scores.empty:
                continue
            good_reference_folds = [
                other_fold
                for other_fold in all_folds
                if other_fold != fold and np.isfinite(rank_by_id.get(other_fold, np.nan)) and rank_by_id[other_fold] >= target_rank_ic
            ]
            reference_definition = f"other_folds_rank_ic_ge_{target_rank_ic:.3f}"
            if not good_reference_folds:
                good_reference_folds = [other_fold for other_fold in all_folds if other_fold != fold]
                reference_definition = "all_other_folds"
            reference = test_predictions.loc[test_predictions["fold"].astype(int).isin(good_reference_folds), "prob_long"]
            reference_scores = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if reference_scores.empty:
                continue
            rank_ic_value = rank_by_id.get(fold, np.nan)
            metrics = _binary_probability_metrics(pd.Series(np.zeros(len(scores))), scores, bins=bins)
            reference_metrics = _binary_probability_metrics(pd.Series(np.zeros(len(reference_scores))), reference_scores, bins=bins)
            score_std = float(scores.std(ddof=0))
            ref_std = float(reference_scores.std(ddof=0))
            score_iqr = float(scores.quantile(0.75) - scores.quantile(0.25))
            ref_iqr = float(reference_scores.quantile(0.75) - reference_scores.quantile(0.25))
            ks = _score_ks_statistic(scores, reference_scores)
            psi = _population_stability_index(scores, reference_scores, bins=bins)
            entropy_delta = metrics["score_entropy_mean"] - reference_metrics["score_entropy_mean"]
            if np.isfinite(psi) and psi >= 0.25:
                issue = "major_score_distribution_shift"
            elif np.isfinite(ks) and ks >= 0.20:
                issue = "large_score_distribution_ks_shift"
            elif np.isfinite(entropy_delta) and abs(entropy_delta) >= 0.05:
                issue = "score_uncertainty_shift"
            else:
                issue = "monitor"
            bucket = (
                "negative_rank_ic"
                if np.isfinite(rank_ic_value) and rank_ic_value < 0.0
                else ("below_target_rank_ic" if np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic else "rank_ic_ok")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold,
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    "reference_definition": reference_definition,
                    "reference_fold_count": len(good_reference_folds),
                    "score_ks_vs_reference": ks,
                    "score_psi_vs_reference": psi,
                    "prob_long_mean": float(scores.mean()),
                    "reference_prob_long_mean": float(reference_scores.mean()),
                    "prob_long_mean_delta": float(scores.mean() - reference_scores.mean()),
                    "prob_long_std": score_std,
                    "reference_prob_long_std": ref_std,
                    "prob_long_std_ratio": float(score_std / ref_std) if ref_std > 0 else np.nan,
                    "prob_long_iqr": score_iqr,
                    "reference_prob_long_iqr": ref_iqr,
                    "score_entropy_mean": metrics["score_entropy_mean"],
                    "reference_score_entropy_mean": reference_metrics["score_entropy_mean"],
                    "score_entropy_delta": entropy_delta,
                    "score_shift_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "score_psi_vs_reference"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

def _score_distribution_shift_summary_frame(score_shift: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "bad_fold_count",
        "mean_score_ks",
        "max_score_ks",
        "mean_score_psi",
        "max_score_psi",
        "bad_score_psi_mean",
        "good_score_psi_mean",
        "bad_score_ks_mean",
        "good_score_ks_mean",
        "high_shift_fold_count",
        "high_shift_folds",
        "score_shift_issue",
        "recommended_next_action",
    ]
    if score_shift.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in score_shift.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank < target_rank_ic
        good_mask = rank >= target_rank_ic
        high_shift = (pd.to_numeric(part["score_psi_vs_reference"], errors="coerce") >= 0.25) | (
            pd.to_numeric(part["score_ks_vs_reference"], errors="coerce") >= 0.20
        )
        high_folds = sorted(part.loc[high_shift, "fold"].dropna().astype(int).tolist())
        bad_psi = _mean_for_mask(part, bad_mask, "score_psi_vs_reference")
        good_psi = _mean_for_mask(part, good_mask, "score_psi_vs_reference")
        bad_ks = _mean_for_mask(part, bad_mask, "score_ks_vs_reference")
        good_ks = _mean_for_mask(part, good_mask, "score_ks_vs_reference")
        if high_folds and np.isfinite(bad_psi) and np.isfinite(good_psi) and bad_psi > good_psi + 0.05:
            issue = "bad_folds_show_score_distribution_shift"
            action = "inspect_feature_drift_for_shifted_folds_before_new_profile_search"
        elif high_folds:
            issue = "score_distribution_shift_not_specific_to_bad_folds"
            action = "monitor_score_distribution_shift"
        else:
            issue = "monitor"
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "bad_fold_count": int(bad_mask.sum()),
                "mean_score_ks": float(pd.to_numeric(part["score_ks_vs_reference"], errors="coerce").mean()),
                "max_score_ks": float(pd.to_numeric(part["score_ks_vs_reference"], errors="coerce").max()),
                "mean_score_psi": float(pd.to_numeric(part["score_psi_vs_reference"], errors="coerce").mean()),
                "max_score_psi": float(pd.to_numeric(part["score_psi_vs_reference"], errors="coerce").max()),
                "bad_score_psi_mean": bad_psi,
                "good_score_psi_mean": good_psi,
                "bad_score_ks_mean": bad_ks,
                "good_score_ks_mean": good_ks,
                "high_shift_fold_count": len(high_folds),
                "high_shift_folds": ",".join(str(fold) for fold in high_folds),
                "score_shift_issue": issue,
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "max_score_psi"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _validation_reliability_metrics(part: pd.DataFrame) -> dict[str, float]:
    if part.empty:
        return {
            "val_rank_ic": np.nan,
            "val_score_gap": np.nan,
            "val_score_ks": np.nan,
            "val_average_precision": np.nan,
            "val_label_long_rate": np.nan,
            "val_prob_long_std": np.nan,
            "val_pred_long_rate_050": np.nan,
        }
    frame = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
    if frame.empty:
        return _validation_reliability_metrics(pd.DataFrame())
    labels = frame["label"].astype(int)
    scores = pd.to_numeric(frame["prob_long"], errors="coerce")
    pos_scores = scores.loc[labels == 1]
    neg_scores = scores.loc[labels == 0]
    score_gap = (
        float(pos_scores.mean() - neg_scores.mean())
        if not pos_scores.empty and not neg_scores.empty
        else np.nan
    )
    if len(labels.unique()) > 1 and scores.nunique(dropna=True) > 1:
        average_precision = float(average_precision_score(labels, scores))
    else:
        average_precision = np.nan
    return {
        "val_rank_ic": _rank_ic_for_frame(frame),
        "val_score_gap": score_gap,
        "val_score_ks": _score_ks_statistic(pos_scores, neg_scores),
        "val_average_precision": average_precision,
        "val_label_long_rate": float(labels.mean()) if len(labels) else np.nan,
        "val_prob_long_std": float(scores.std(ddof=0)) if scores.notna().any() else np.nan,
        "val_pred_long_rate_050": float((scores >= 0.5).mean()) if scores.notna().any() else np.nan,
    }

def _reliability_gate_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    gates = cfg.get("gates", []) or []
    if gates:
        return [dict(item) for item in gates if isinstance(item, dict) and str(item.get("name", "")).strip()]
    return [
        {"name": "val_rank_ic_positive", "min_val_rank_ic": 0.0},
        {"name": "val_score_gap_positive", "min_val_score_gap": 0.0},
        {"name": "val_rank_ic_and_score_gap_positive", "min_val_rank_ic": 0.0, "min_val_score_gap": 0.0},
        {
            "name": "val_rank_ic_score_gap_and_ks",
            "min_val_rank_ic": 0.0,
            "min_val_score_gap": 0.0,
            "min_val_score_ks": 0.08,
        },
    ]

def _gate_threshold_check(metrics: dict[str, float], gate: dict[str, Any], key: str, metric: str) -> bool:
    if key not in gate:
        return True
    value = _float(metrics, metric)
    threshold = _optional_float(gate.get(key))
    if threshold is None or not np.isfinite(threshold):
        return True
    if not np.isfinite(value):
        return False
    if key.startswith("min_"):
        return value >= threshold
    if key.startswith("max_"):
        return value <= threshold
    return True

def _fold_reliability_gate_passed(metrics: dict[str, float], gate: dict[str, Any]) -> bool:
    checks = [
        ("min_val_rank_ic", "val_rank_ic"),
        ("max_val_rank_ic", "val_rank_ic"),
        ("min_val_score_gap", "val_score_gap"),
        ("max_val_score_gap", "val_score_gap"),
        ("min_val_score_ks", "val_score_ks"),
        ("max_val_score_ks", "val_score_ks"),
        ("min_val_average_precision", "val_average_precision"),
        ("max_val_average_precision", "val_average_precision"),
        ("min_val_prob_long_std", "val_prob_long_std"),
        ("max_val_prob_long_std", "val_prob_long_std"),
        ("max_val_pred_long_rate_050", "val_pred_long_rate_050"),
    ]
    if "min_val_average_precision_lift_vs_base" in gate:
        ap = _float(metrics, "val_average_precision")
        base = _float(metrics, "val_label_long_rate")
        lift = ap - base if np.isfinite(ap) and np.isfinite(base) else np.nan
        metrics = {**metrics, "val_average_precision_lift_vs_base": lift}
        checks.append(("min_val_average_precision_lift_vs_base", "val_average_precision_lift_vs_base"))
    return all(_gate_threshold_check(metrics, gate, key, metric) for key, metric in checks)

def _fold_reliability_gate_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "gate_name",
        "fold",
        "gate_passed",
        "val_rank_ic",
        "val_score_gap",
        "val_score_ks",
        "val_average_precision",
        "val_label_long_rate",
        "val_prob_long_std",
        "val_pred_long_rate_050",
        "test_rank_ic",
        "test_official_f1",
        "test_official_pred_long_rate",
        "test_top_10_lift_vs_base",
        "test_top_10_forward_return",
    ]
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(columns=columns)
    gates = _reliability_gate_definitions(config)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "split", "label", "prob_long", "forward_return"}.issubset(predictions.columns):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        score_bands = diagnostics.get("score_band_by_fold")
        top10_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(score_bands, pd.DataFrame) and not score_bands.empty and {"fold", "band"}.issubset(score_bands.columns):
            top10 = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            top10_by_id = {int(row["fold"]): row.to_dict() for _, row in top10.dropna(subset=["fold"]).iterrows()}

        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in predictions.groupby("fold"):
            fold_id = int(fold)
            validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
            test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            if validation.empty or test.empty:
                continue
            val_metrics = _validation_reliability_metrics(validation)
            fold_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            top10_row = top10_by_id.get(fold_id, {})
            for gate in gates:
                row = {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "gate_name": str(gate.get("name", "")),
                    "fold": fold_id,
                    "gate_passed": _fold_reliability_gate_passed(dict(val_metrics), gate),
                    **val_metrics,
                    "test_rank_ic": _float(fold_row, "rank_ic", _rank_ic_for_frame(test)),
                    "test_official_f1": _float(
                        threshold_row,
                        "test_f1_at_official_threshold",
                        _float(threshold_row, "test_f1_at_constrained_threshold"),
                    ),
                    "test_official_pred_long_rate": _float(
                        threshold_row,
                        "test_pred_long_rate_at_official_threshold",
                        _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
                    ),
                    "test_top_10_lift_vs_base": _float(top10_row, "lift_vs_base"),
                    "test_top_10_forward_return": _float(top10_row, "mean_forward_return"),
                }
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "gate_name", "fold"])
        .reset_index(drop=True)
    )

def _fold_reliability_gate_summary_frame(detail: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "gate_name",
        "fold_count",
        "accepted_fold_count",
        "rejected_fold_count",
        "accepted_fraction",
        "all_rank_ic_mean",
        "all_rank_ic_std",
        "accepted_rank_ic_mean",
        "accepted_rank_ic_std",
        "accepted_positive_ic_fraction",
        "accepted_official_f1_mean",
        "accepted_top_10_forward_return_mean",
        "rejected_negative_fold_capture_rate",
        "false_reject_positive_fold_rate",
        "accepted_rank_ic_mean_delta",
        "accepted_rank_ic_std_delta",
        "accepted_official_f1_delta",
        "gate_passed_cv",
        "reject_reason",
        "next_action",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    min_fraction = float(cfg.get("min_accepted_fraction", 0.50))
    min_folds = int(cfg.get("min_accepted_folds", 12))
    min_positive = float(cfg.get("min_positive_ic_fraction", 0.75))
    max_std = float(cfg.get("max_rank_ic_std", 0.06))
    min_f1_delta = float(cfg.get("min_official_f1_delta", 0.0))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, gate_name), part in detail.groupby(
        ["candidate", "candidate_type", "fold_scope", "gate_name"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["test_rank_ic"], errors="coerce")
        passed = part["gate_passed"].astype(bool)
        accepted = part.loc[passed]
        rejected = part.loc[~passed]
        accepted_rank = pd.to_numeric(accepted["test_rank_ic"], errors="coerce")
        rejected_rank = pd.to_numeric(rejected["test_rank_ic"], errors="coerce")
        total_negative = int((rank < 0.0).sum())
        rejected_negative = int((rejected_rank < 0.0).sum())
        total_positive = int((rank > 0.0).sum())
        rejected_positive = int((rejected_rank > 0.0).sum())
        all_f1 = pd.to_numeric(part["test_official_f1"], errors="coerce")
        accepted_f1 = pd.to_numeric(accepted["test_official_f1"], errors="coerce")
        all_mean = float(rank.mean()) if rank.notna().any() else np.nan
        all_std = float(rank.std(ddof=1)) if rank.notna().sum() > 1 else np.nan
        accepted_mean = float(accepted_rank.mean()) if accepted_rank.notna().any() else np.nan
        accepted_std = float(accepted_rank.std(ddof=1)) if accepted_rank.notna().sum() > 1 else np.nan
        f1_delta = (
            float(accepted_f1.mean() - all_f1.mean())
            if accepted_f1.notna().any() and all_f1.notna().any()
            else np.nan
        )
        reasons: list[str] = []
        accepted_count = int(len(accepted))
        fold_count = int(part["fold"].nunique())
        accepted_fraction = float(accepted_count / fold_count) if fold_count else 0.0
        positive_fraction = float((accepted_rank > 0.0).mean()) if accepted_count else np.nan
        if accepted_count < min_folds:
            reasons.append("accepted_fold_count")
        if accepted_fraction < min_fraction:
            reasons.append("accepted_fraction")
        if not np.isfinite(positive_fraction) or positive_fraction < min_positive:
            reasons.append("accepted_positive_ic_fraction")
        if not np.isfinite(accepted_std) or accepted_std > max_std:
            reasons.append("accepted_rank_ic_std")
        if not np.isfinite(f1_delta) or f1_delta < min_f1_delta:
            reasons.append("accepted_official_f1_delta")
        if np.isfinite(accepted_std) and np.isfinite(all_std) and accepted_std >= all_std:
            reasons.append("does_not_reduce_rank_ic_std")
        reject_reason = ";".join(dict.fromkeys(reasons))
        if not reject_reason:
            next_action = "pre_register_reliability_gate_for_future_oos_review"
        elif "accepted_fold_count" in reject_reason or "accepted_fraction" in reject_reason:
            next_action = "gate_too_sparse_for_phase1_decision"
        elif "accepted_rank_ic_std" in reject_reason or "does_not_reduce_rank_ic_std" in reject_reason:
            next_action = "gate_does_not_reduce_fold_std"
        else:
            next_action = "gate_diagnostic_only_do_not_promote"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "gate_name": gate_name,
                "fold_count": fold_count,
                "accepted_fold_count": accepted_count,
                "rejected_fold_count": int(len(rejected)),
                "accepted_fraction": accepted_fraction,
                "all_rank_ic_mean": all_mean,
                "all_rank_ic_std": all_std,
                "accepted_rank_ic_mean": accepted_mean,
                "accepted_rank_ic_std": accepted_std,
                "accepted_positive_ic_fraction": positive_fraction,
                "accepted_official_f1_mean": float(accepted_f1.mean()) if accepted_f1.notna().any() else np.nan,
                "accepted_top_10_forward_return_mean": _numeric_mean(accepted, "test_top_10_forward_return"),
                "rejected_negative_fold_capture_rate": (
                    float(rejected_negative / total_negative) if total_negative else np.nan
                ),
                "false_reject_positive_fold_rate": (
                    float(rejected_positive / total_positive) if total_positive else np.nan
                ),
                "accepted_rank_ic_mean_delta": (
                    accepted_mean - all_mean if np.isfinite(accepted_mean) and np.isfinite(all_mean) else np.nan
                ),
                "accepted_rank_ic_std_delta": (
                    accepted_std - all_std if np.isfinite(accepted_std) and np.isfinite(all_std) else np.nan
                ),
                "accepted_official_f1_delta": f1_delta,
                "gate_passed_cv": not bool(reject_reason),
                "reject_reason": reject_reason,
                "next_action": next_action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["gate_passed_cv", "accepted_rank_ic_std_delta", "accepted_rank_ic_mean_delta"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )

def _fold_reliability_gate_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Fold Reliability Gates", ""]
    lines.append(
        "These rows test causal validation-fold reliability gates. They are diagnostics only: "
        "a gate can be considered for future unseen OOS only if it improves CV stability without using holdout feedback."
    )
    if summary.empty:
        lines.append("")
        lines.append("No fold-reliability gate rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "gate_name",
        "accepted_fold_count",
        "accepted_fraction",
        "accepted_rank_ic_mean",
        "accepted_rank_ic_std",
        "accepted_positive_ic_fraction",
        "accepted_official_f1_mean",
        "rejected_negative_fold_capture_rate",
        "gate_passed_cv",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _probability_quality_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Probability Quality Forensics", ""]
    if summary.empty:
        lines.append("No probability-quality rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "mean_brier_score",
        "mean_average_precision",
        "mean_ece_equal_count",
        "bad_average_precision_mean",
        "good_average_precision_mean",
        "bad_ece_equal_count_mean",
        "good_ece_equal_count_mean",
        "probability_quality_issue",
        "recommended_next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _score_distribution_shift_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Score Distribution Shift", ""]
    if summary.empty:
        lines.append("No score-distribution shift rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "mean_score_ks",
        "max_score_ks",
        "mean_score_psi",
        "max_score_psi",
        "bad_score_psi_mean",
        "good_score_psi_mean",
        "high_shift_folds",
        "score_shift_issue",
        "recommended_next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_probability_quality_forensics(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "probability_quality_forensics.csv", index=False)
    summary.to_csv(path / "probability_quality_summary.csv", index=False)
    (path / "probability_quality_forensics.md").write_text(_probability_quality_markdown(summary), encoding="utf-8")
    _write_json(
        path / "probability_quality_forensics.json",
        {
            "probability_quality_forensics": detail.to_dict(orient="records"),
            "probability_quality_summary": summary.to_dict(orient="records"),
        },
    )

def _write_score_distribution_shift(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "score_distribution_shift.csv", index=False)
    summary.to_csv(path / "score_distribution_shift_summary.csv", index=False)
    (path / "score_distribution_shift.md").write_text(_score_distribution_shift_markdown(summary), encoding="utf-8")
    _write_json(
        path / "score_distribution_shift.json",
        {
            "score_distribution_shift": detail.to_dict(orient="records"),
            "score_distribution_shift_summary": summary.to_dict(orient="records"),
        },
    )

def _write_fold_reliability_gate(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "fold_reliability_gate.csv", index=False)
    summary.to_csv(path / "fold_reliability_gate_summary.csv", index=False)
    (path / "fold_reliability_gate.md").write_text(_fold_reliability_gate_markdown(summary), encoding="utf-8")
    _write_json(
        path / "fold_reliability_gate.json",
        {
            "fold_reliability_gate": detail.to_dict(orient="records"),
            "fold_reliability_gate_summary": summary.to_dict(orient="records"),
        },
    )

def _forensics_markdown(title: str, frame: pd.DataFrame) -> str:
    return _table_markdown(title, frame)

def _write_forensics_reports(
    path: Path,
    *,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    reports = [
        ("fold_stability_forensics", "Fold Stability Forensics", fold_stability_forensics),
        ("fold_stability_summary", "Fold Stability Summary", fold_stability_summary),
        ("threshold_forensics", "Threshold Forensics", threshold_forensics),
    ]
    for stem, title, frame in reports:
        frame.to_csv(path / f"{stem}.csv", index=False)
        (path / f"{stem}.md").write_text(_forensics_markdown(title, frame), encoding="utf-8")
        _write_json(path / f"{stem}.json", {"rows": frame.to_dict(orient="records")})
