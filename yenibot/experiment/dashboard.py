"""Compact, evidence-focused Phase 1 model performance dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from yenibot.experiment.common import _json_ready, _write_json
from yenibot.experiment.training import _test_predictions

__all__ = [
    "attach_active_charter_status",
    "write_model_performance_dashboard",
]


GREEN = "#18864B"
RED = "#C43D3D"
AMBER = "#C68116"
BLUE = "#2563A6"
INK = "#17202A"
MUTED = "#66717E"
GRID = "#D9DEE5"
LIGHT = "#F4F6F8"


def _number(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _model_evidence_passed(phase2_readiness: dict[str, Any]) -> bool:
    future_only = {
        "future_unseen_oos_not_ready",
        "future_unseen_oos_not_evaluated",
    }
    blockers = {
        str(item)
        for item in phase2_readiness.get("blockers", []) or []
    }
    return not (blockers - future_only)


def _first(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    if frame.empty:
        return {}
    part = frame.loc[mask] if mask is not None else frame
    return part.iloc[0].to_dict() if not part.empty else {}


def _control_entry(entries: list[dict[str, Any]], control_profile: str) -> dict[str, Any]:
    for entry in entries:
        if (
            str(entry.get("profile")) == control_profile
            and str(entry.get("fold_scope")) == "full"
        ):
            return entry
    return {}


def _calibration_frame(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    columns = ["bin", "count", "mean_probability", "actual_long_rate", "absolute_gap"]
    if predictions.empty or not {"label", "prob_long"}.issubset(predictions.columns):
        return pd.DataFrame(columns=columns)
    clean = predictions[["label", "prob_long"]].copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce")
    clean["prob_long"] = pd.to_numeric(clean["prob_long"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.DataFrame(columns=columns)
    try:
        clean["bin"] = pd.qcut(clean["prob_long"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        clean["bin"] = 0
    frame = (
        clean.groupby("bin", observed=True)
        .agg(
            count=("label", "size"),
            mean_probability=("prob_long", "mean"),
            actual_long_rate=("label", "mean"),
        )
        .reset_index()
    )
    frame["absolute_gap"] = (frame["actual_long_rate"] - frame["mean_probability"]).abs()
    return frame[columns]


def _precision_recall_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["threshold", "precision", "recall"]
    if predictions.empty or not {"label", "prob_long"}.issubset(predictions.columns):
        return pd.DataFrame(columns=columns)
    clean = predictions[["label", "prob_long"]].copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce")
    clean["prob_long"] = pd.to_numeric(clean["prob_long"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean["label"].nunique() < 2:
        return pd.DataFrame(columns=columns)
    precision, recall, thresholds = precision_recall_curve(
        clean["label"].astype(int),
        clean["prob_long"],
    )
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame(
        {"threshold": threshold_values, "precision": precision, "recall": recall}
    )


def _metric_definitions() -> pd.DataFrame:
    rows = [
        ("Mean Rank IC", "Discrimination", "Spearman correlation between P(Long) and forward return across OOS folds.", "Higher and positive."),
        ("Positive fold fraction", "Temporal robustness", "Share of OOS folds with positive Rank IC.", "Persistent across market windows."),
        ("Random-effects lower CI", "Statistical confidence", "Lower confidence bound after accounting for fold uncertainty and dependence assumptions.", "Positive across configured block lengths."),
        ("PRAUC lift", "Imbalanced classification", "Average precision divided by positive-label prevalence.", "Above 1.0; active gate is 1.05."),
        ("Precision lift", "Decision quality", "Official-policy precision divided by unconditional long-label prevalence.", "Above 1.0; active gate is 1.05."),
        ("Rate-matched F1 skill", "Classification skill", "F1 minus random-selection F1 at the same prediction rate.", "Positive and persistent across folds."),
        ("Brier score", "Probability quality", "Mean squared probability error; combines calibration and resolution.", "Lower, compared over time/models."),
        ("Brier skill", "Probability quality", "One minus model Brier score divided by the fold-climatology Brier score.", "Positive; negative means worse than predicting the base rate."),
        ("Log loss", "Probability quality", "Proper scoring rule that strongly penalizes confident wrong probabilities.", "Lower, compared over time/models."),
        ("Log-loss skill", "Probability quality", "One minus model log loss divided by fold-climatology log loss.", "Positive; interpret beside calibration slope and ECE."),
        ("ECE", "Calibration", "Weighted gap between predicted probability and observed frequency.", "Lower; inspect with reliability diagram."),
        ("Score separation", "Discrimination", "Mean score for actual longs minus mean score for not-long rows.", "Positive; this is not a calibration metric."),
        ("Top-decile lift", "Economic ordering", "Long-label rate in the top score decile divided by base prevalence.", "Above 1.0 and paired with positive return."),
        ("Top-decile forward return", "Economic ordering", "Mean realized forward return for the top score decile.", "Positive on CV and confirmed on future unseen OOS."),
        ("Seed dispersion", "Training robustness", "Variation of key metrics across deterministic seeds.", "Small relative to the signal margin."),
        ("Future unseen OOS", "Promotion evidence", "Prediction-only evaluation after the frozen cutoff, with no refit.", "Required before Phase 2."),
    ]
    return pd.DataFrame(rows, columns=["metric", "category", "definition", "healthy_interpretation"])


def attach_active_charter_status(
    comparison: pd.DataFrame,
    *,
    phase2_readiness: dict[str, Any],
    control_profile: str,
) -> pd.DataFrame:
    """Add unambiguous active-charter and overall-readiness fields."""

    frame = comparison.copy()
    if "passed_phase1_legacy_v3" not in frame.columns:
        frame["passed_phase1_legacy_v3"] = frame.get("passed_phase1", False)
    frame["active_validation_charter"] = str(
        phase2_readiness.get("active_validation_charter") or "unknown"
    )
    frame["model_evidence_passed_active_charter"] = False
    frame["phase2_ready"] = False
    frame["phase1_status"] = "not_evaluated_under_active_charter"
    control_mask = (
        frame["profile"].astype(str).eq(control_profile)
        & frame["fold_scope"].astype(str).eq("full")
    )
    model_passed = _model_evidence_passed(phase2_readiness)
    frame.loc[control_mask, "model_evidence_passed_active_charter"] = model_passed
    frame.loc[control_mask, "phase2_ready"] = bool(
        phase2_readiness.get("ready_for_phase2", False)
    )
    if bool(phase2_readiness.get("ready_for_phase2", False)):
        status = "ready_for_phase2"
    elif model_passed and "future_unseen_oos_not_ready" in (
        phase2_readiness.get("blockers", []) or []
    ):
        status = "model_evidence_passed_future_oos_pending"
    elif model_passed:
        status = "model_evidence_passed_other_governance_blocker"
    else:
        status = "active_charter_model_evidence_failed"
    frame.loc[control_mask, "phase1_status"] = status
    return frame


def _scorecard_frame(
    *,
    phase2_readiness: dict[str, Any],
    comparison: pd.DataFrame,
    rank_ic_evidence: pd.DataFrame,
    classification_skill: pd.DataFrame,
    probability_quality: pd.DataFrame,
    fold_stability: pd.DataFrame,
    payoff_alignment: pd.DataFrame,
    seed_stability: pd.DataFrame,
    future_oos: dict[str, Any],
    control_profile: str,
) -> pd.DataFrame:
    columns = [
        "category",
        "metric",
        "scope",
        "value",
        "target",
        "status",
        "role",
        "interpretation",
        "source",
    ]
    rows: list[dict[str, Any]] = []

    category_map = {
        "mean_rank_ic": "Discrimination",
        "positive_fold_fraction": "Temporal robustness",
        "positive_fold_sign_test_pvalue": "Statistical confidence",
        "random_effects_positive_all_blocks": "Statistical confidence",
        "rank_ic_std": "Risk monitor",
        "prauc_lift_vs_prevalence": "Imbalanced classification",
        "precision_lift_vs_prevalence": "Decision quality",
        "f1_skill_vs_rate_matched_random": "Classification skill",
        "positive_f1_skill_fold_fraction": "Temporal robustness",
        "positive_forward_return_fold_fraction": "Economic ordering",
        "prediction_long_rate": "Decision guardrail",
        "raw_long_f1": "Risk monitor",
        "calibration_separation": "Score separation",
        "mtf_leakage": "Integrity",
        "stationarity_policy": "Integrity",
        "seed_audit_coverage": "Training robustness",
        "future_unseen_oos_ready": "Promotion evidence",
        "frozen_candidate_manifest": "Integrity",
        "future_unseen_oos_evaluated": "Promotion evidence",
        "future_unseen_oos_passed": "Promotion evidence",
    }
    interpretation_map = {
        "rank_ic_std": "Visible legacy monitor; interpret beside block-bootstrap noise and random-effects evidence.",
        "raw_long_f1": "Visible legacy monitor; interpret beside prevalence, prediction rate, and rate-matched random skill.",
        "future_unseen_oos_ready": "Fresh labeled rows must reach the preregistered minimum before scoring.",
        "future_unseen_oos_evaluated": "Must be prediction-only with zero fit operations.",
        "future_unseen_oos_passed": "Final preregistered promotion evidence before Phase 2.",
        "calibration_separation": (
            "Mean score gap between actual long and not-long rows. This is not "
            "probability calibration."
        ),
    }
    for item in phase2_readiness.get("checks", []) or []:
        metric = str(item.get("check"))
        rows.append(
            {
                "category": category_map.get(metric, "Validation"),
                "metric": metric,
                "scope": "walk_forward_oos",
                "value": item.get("value"),
                "target": item.get("target"),
                "status": item.get("status"),
                "role": item.get("role", "gate"),
                "interpretation": interpretation_map.get(
                    metric,
                    "Active validation-charter criterion.",
                ),
                "source": item.get("source", "phase2_readiness.json"),
            }
        )

    control = _first(
        comparison,
        (comparison["profile"].astype(str) == control_profile)
        & comparison["fold_scope"].astype(str).eq("full")
        if not comparison.empty
        else None,
    )
    rank = _first(
        rank_ic_evidence,
        (rank_ic_evidence["candidate"].astype(str) == control_profile)
        & rank_ic_evidence["fold_scope"].astype(str).eq("full")
        if not rank_ic_evidence.empty
        else None,
    )
    skill = _first(
        classification_skill,
        (classification_skill["candidate"].astype(str) == control_profile)
        & classification_skill["fold_scope"].astype(str).eq("full")
        & classification_skill["policy_name"].astype(str).eq("official_threshold")
        if not classification_skill.empty
        else None,
    )
    quality = _first(
        probability_quality,
        (probability_quality["candidate"].astype(str) == control_profile)
        & probability_quality["fold_scope"].astype(str).eq("full")
        if not probability_quality.empty
        else None,
    )
    stability = _first(
        fold_stability,
        (fold_stability["candidate"].astype(str) == control_profile)
        & fold_stability["fold_scope"].astype(str).eq("full")
        if not fold_stability.empty
        else None,
    )
    seed = _first(
        seed_stability,
        seed_stability["profile"].astype(str).eq(control_profile)
        if not seed_stability.empty
        else None,
    )
    cv_payoff = _first(
        payoff_alignment,
        (payoff_alignment["candidate"].astype(str) == control_profile)
        & payoff_alignment["evaluation_scope"].astype(str).eq("cv_test")
        & payoff_alignment["band"].astype(str).eq("top_10")
        if not payoff_alignment.empty
        else None,
    )
    holdout_payoff = _first(
        payoff_alignment,
        (payoff_alignment["candidate"].astype(str) == control_profile)
        & payoff_alignment["evaluation_scope"].astype(str).eq("holdout")
        & payoff_alignment["band"].astype(str).eq("top_10")
        if not payoff_alignment.empty
        else None,
    )

    extras = [
        ("Statistical confidence", "random_effects_lower_ci_min", "walk_forward_oos", rank.get("random_effects_ci_low_min"), "> 0", "monitor", "confidence", "Minimum lower confidence bound across block assumptions.", "rank_ic_aggregate_evidence.csv"),
        ("Tail risk", "worst_fold_rank_ic", "walk_forward_oos", stability.get("worst_fold_rank_ic"), "monitor", "monitor", "risk", "Worst temporal fold; should remain visible even when aggregate gates pass.", "fold_stability_summary.csv"),
        ("Tail risk", "top_5_variance_contribution", "walk_forward_oos", stability.get("top_5_variance_contribution"), "monitor", "monitor", "risk", "Fraction of fold variance concentrated in the five largest contributors.", "fold_stability_summary.csv"),
        ("Probability quality", "brier_score", "walk_forward_oos", quality.get("mean_brier_score"), "lower is better", "monitor", "monitor", "Proper probability score; compare across frozen evaluations.", "probability_quality_summary.csv"),
        ("Probability quality", "brier_skill_vs_climatology", "walk_forward_oos", quality.get("mean_brier_skill_vs_climatology"), "> 0", "passed" if _number(quality.get("mean_brier_skill_vs_climatology")) > 0 else "failed", "monitor", "Probability improvement over fold climatology; negative means raw probabilities are worse than the base rate.", "probability_quality_summary.csv"),
        ("Probability quality", "log_loss", "walk_forward_oos", quality.get("mean_log_loss"), "lower is better", "monitor", "monitor", "Penalizes confident wrong probabilities.", "probability_quality_summary.csv"),
        ("Probability quality", "log_loss_skill_vs_climatology", "walk_forward_oos", quality.get("mean_log_loss_skill_vs_climatology"), "> 0", "passed" if _number(quality.get("mean_log_loss_skill_vs_climatology")) > 0 else "failed", "monitor", "Log-loss improvement over fold climatology.", "probability_quality_summary.csv"),
        ("Calibration", "ece_equal_count", "walk_forward_oos", quality.get("mean_ece_equal_count"), "lower is better", "monitor", "monitor", "Expected calibration error across equal-count bins.", "probability_quality_summary.csv"),
        ("Economic ordering", "top_10_cv_label_lift", "walk_forward_oos", cv_payoff.get("label_lift_vs_base", control.get("top_10_lift_global")), "> 1", "passed" if _number(cv_payoff.get("label_lift_vs_base", control.get("top_10_lift_global"))) > 1 else "failed", "monitor", "Top-decile label concentration; not sufficient without positive payoff.", "payoff_alignment.csv"),
        ("Economic ordering", "top_10_cv_forward_return", "walk_forward_oos", cv_payoff.get("mean_forward_return"), "> 0", "passed" if _number(cv_payoff.get("mean_forward_return")) > 0 else "failed", "monitor", "Top-decile realized forward return on CV test folds.", "payoff_alignment.csv"),
        ("Holdout diagnostic", "top_10_holdout_forward_return", "seen_holdout", holdout_payoff.get("mean_forward_return"), "diagnostic only", "warning" if _number(holdout_payoff.get("mean_forward_return")) <= 0 else "passed", "monitor", "Seen holdout payoff; cannot be used for retuning.", "payoff_alignment.csv"),
        ("Training robustness", "seed_mean_rank_ic_std", "seed_audit", seed.get("mean_rank_ic_seed_std"), "lower is better", "monitor", "monitor", "Dispersion of mean Rank IC across configured seeds.", "seed_stability.csv"),
        ("Promotion evidence", "future_oos_progress", "future_unseen_oos", future_oos.get("new_labeled_rows"), f">= {future_oos.get('min_rows')}", "passed" if future_oos.get("ready_for_evaluation") else "pending", "gate", "Fresh labeled rows accumulated after the frozen anchor.", "future_oos_readiness.json"),
    ]
    for category, metric, scope, value, target, status, role, interpretation, source in extras:
        rows.append(
            {
                "category": category,
                "metric": metric,
                "scope": scope,
                "value": value,
                "target": target,
                "status": status,
                "role": role,
                "interpretation": interpretation,
                "source": source,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _save_scorecard_figure(
    path: Path,
    *,
    scorecard: pd.DataFrame,
    phase2_readiness: dict[str, Any],
    future_oos: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    blockers = phase2_readiness.get("blockers", []) or []
    model_pass = _model_evidence_passed(phase2_readiness)
    headline_color = GREEN if model_pass else RED
    ax.text(0.02, 0.97, "yeniBot Phase 1 Model Evidence", fontsize=21, weight="bold", color=INK, va="top")
    ax.text(
        0.02,
        0.875,
        "MODEL EVIDENCE: PASS" if model_pass else "MODEL EVIDENCE: REVIEW REQUIRED",
        fontsize=15,
        weight="bold",
        color=headline_color,
    )
    ax.text(
        0.98,
        0.95,
        "PHASE 2: READY" if phase2_readiness.get("ready_for_phase2") else "PHASE 2: BLOCKED",
        fontsize=15,
        weight="bold",
        color=GREEN if phase2_readiness.get("ready_for_phase2") else AMBER,
        ha="right",
        va="top",
    )
    key_metrics = [
        ("mean_rank_ic", "Mean Rank IC"),
        ("positive_fold_fraction", "Positive folds"),
        ("prauc_lift_vs_prevalence", "PRAUC lift"),
        ("precision_lift_vs_prevalence", "Precision lift"),
        ("f1_skill_vs_rate_matched_random", "F1 skill"),
        ("positive_forward_return_fold_fraction", "Positive-return folds"),
    ]
    lookup = scorecard.set_index("metric").to_dict(orient="index") if not scorecard.empty else {}
    for idx, (metric, label) in enumerate(key_metrics):
        col = idx % 3
        row = idx // 3
        x = 0.02 + col * 0.325
        y = 0.69 - row * 0.20
        box = FancyBboxPatch(
            (x, y),
            0.29,
            0.14,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            linewidth=1,
            edgecolor=GRID,
            facecolor="white",
        )
        ax.add_patch(box)
        item = lookup.get(metric, {})
        value = _number(item.get("value"))
        if metric in {"positive_fold_fraction", "positive_forward_return_fold_fraction"}:
            shown = f"{value:.1%}" if np.isfinite(value) else "NA"
        elif metric == "f1_skill_vs_rate_matched_random":
            shown = f"{value:+.3f}" if np.isfinite(value) else "NA"
        else:
            shown = f"{value:.3f}" if np.isfinite(value) else "NA"
        ax.text(x + 0.018, y + 0.095, shown, fontsize=23, weight="bold", color=INK)
        ax.text(x + 0.018, y + 0.045, label, fontsize=10.5, color=MUTED)
        status = str(item.get("status", ""))
        ax.text(
            x + 0.27,
            y + 0.045,
            status.upper(),
            fontsize=9,
            color=GREEN if status == "passed" else AMBER,
            ha="right",
        )

    ax.text(0.02, 0.33, "Promotion gate", fontsize=13, weight="bold", color=INK)
    rows = _number(future_oos.get("new_labeled_rows"), 0)
    minimum = _number(future_oos.get("min_rows"), 1)
    progress = min(max(rows / minimum, 0.0), 1.0) if minimum > 0 else 0.0
    ax.add_patch(FancyBboxPatch((0.02, 0.26), 0.96, 0.045, boxstyle="round,pad=0", facecolor=LIGHT, edgecolor=GRID))
    ax.add_patch(FancyBboxPatch((0.02, 0.26), 0.96 * progress, 0.045, boxstyle="round,pad=0", facecolor=BLUE, edgecolor=BLUE))
    ax.text(
        0.02,
        0.21,
        f"Future unseen OOS: {int(rows)} / {int(minimum)} labeled rows "
        f"({int(_number(future_oos.get('min_rows_remaining'), 0))} remaining)",
        fontsize=11,
        color=INK,
    )
    ax.text(0.02, 0.14, "Active blocker", fontsize=12, weight="bold", color=INK)
    ax.text(0.02, 0.095, ", ".join(blockers) if blockers else "None", fontsize=11, color=AMBER if blockers else GREEN)
    ax.text(
        0.98,
        0.035,
        "CV/holdout diagnostics are not a backtest. Promotion requires frozen future unseen OOS.",
        fontsize=9.5,
        color=MUTED,
        ha="right",
    )
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_rank_ic_figure(
    path: Path,
    *,
    fold_stability: pd.DataFrame,
    rank_evidence: pd.DataFrame,
    control_profile: str,
) -> None:
    part = fold_stability.loc[
        (fold_stability["candidate"].astype(str) == control_profile)
        & fold_stability["fold_scope"].astype(str).eq("full")
    ].copy() if not fold_stability.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    if part.empty:
        ax.text(0.5, 0.5, "No fold Rank IC data", ha="center", va="center")
        ax.axis("off")
    else:
        part = part.sort_values("fold")
        values = pd.to_numeric(part["rank_ic"], errors="coerce")
        colors = [GREEN if value >= 0 else RED for value in values]
        ax.bar(part["fold"].astype(int), values, color=colors, width=0.78)
        mean_ic = float(values.mean())
        ax.axhline(0, color=INK, linewidth=1)
        ax.axhline(mean_ic, color=BLUE, linestyle="--", linewidth=1.6, label=f"Mean IC {mean_ic:.3f}")
        ax.axhline(0.03, color=AMBER, linestyle=":", linewidth=1.4, label="Mean IC gate 0.03")
        ax.set_xlabel("Walk-forward fold")
        ax.set_ylabel("Spearman Rank IC")
        ax.set_title("Out-of-sample Rank IC by fold", loc="left", fontsize=16, weight="bold")
        ax.grid(axis="y", color=GRID, alpha=0.7)
        ax.legend(frameon=False, loc="upper left")
        evidence = _first(
            rank_evidence,
            (rank_evidence["candidate"].astype(str) == control_profile)
            & rank_evidence["fold_scope"].astype(str).eq("full")
            if not rank_evidence.empty
            else None,
        )
        ax.text(
            0.99,
            0.98,
            f"Positive folds: {_number(evidence.get('positive_fold_fraction')):.1%}\n"
            f"Sign-test p: {_number(evidence.get('positive_fold_sign_test_pvalue')):.2g}\n"
            f"Min random-effects lower CI: {_number(evidence.get('random_effects_ci_low_min')):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.5"},
        )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_classification_figure(
    path: Path,
    *,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    pr_curve: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    prevalence = (
        _number(pd.to_numeric(predictions["label"], errors="coerce").mean())
        if "label" in predictions.columns
        else np.nan
    )
    ax = axes[0]
    if pr_curve.empty:
        ax.text(0.5, 0.5, "No precision-recall data", ha="center", va="center")
    else:
        ax.plot(pr_curve["recall"], pr_curve["precision"], color=BLUE, linewidth=2)
        if np.isfinite(prevalence):
            ax.axhline(prevalence, color=AMBER, linestyle="--", label=f"No-skill prevalence {prevalence:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(frameon=False)
    ax.set_title("Precision-recall curve", loc="left", fontsize=15, weight="bold")
    ax.grid(color=GRID, alpha=0.6)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    if calibration.empty:
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
    else:
        sizes = 30 + 170 * calibration["count"] / calibration["count"].max()
        ax.plot(calibration["mean_probability"], calibration["actual_long_rate"], color=GREEN, linewidth=2)
        ax.scatter(calibration["mean_probability"], calibration["actual_long_rate"], s=sizes, color=GREEN, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(Long)")
    ax.set_ylabel("Observed long rate")
    ax.set_title("Reliability diagram", loc="left", fontsize=15, weight="bold")
    ax.grid(color=GRID, alpha=0.6)
    ax.legend(frameon=False)
    fig.suptitle("Out-of-sample classification quality", x=0.02, ha="left", fontsize=17, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_payoff_figure(
    path: Path,
    *,
    payoff_alignment: pd.DataFrame,
    control_profile: str,
) -> None:
    part = payoff_alignment.loc[
        payoff_alignment["candidate"].astype(str).eq(control_profile)
        & payoff_alignment["band"].astype(str).isin(["top_10", "top_20", "top_30", "upper_half"])
    ].copy() if not payoff_alignment.empty else pd.DataFrame()
    order = ["top_10", "top_20", "top_30", "upper_half"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title, baseline in [
        (axes[0], "label_lift_vs_base", "Long-label lift by score band", 1.0),
        (axes[1], "mean_forward_return", "Forward return by score band", 0.0),
    ]:
        if part.empty:
            ax.text(0.5, 0.5, "No payoff data", ha="center", va="center")
            continue
        width = 0.36
        x = np.arange(len(order))
        for offset, scope, color, label in [
            (-width / 2, "cv_test", BLUE, "Walk-forward OOS"),
            (width / 2, "holdout", AMBER, "Seen holdout diagnostic"),
        ]:
            scope_part = part.loc[part["evaluation_scope"].astype(str).eq(scope)].set_index("band")
            series = scope_part.get(metric, pd.Series(dtype=float))
            values = [_number(series.get(band)) for band in order]
            ax.bar(x + offset, values, width=width, color=color, label=label)
        ax.axhline(baseline, color=INK, linewidth=1)
        ax.set_xticks(x, [item.replace("_", " ") for item in order])
        ax.set_title(title, loc="left", fontsize=15, weight="bold")
        ax.grid(axis="y", color=GRID, alpha=0.6)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Score concentration and economic ordering", x=0.02, ha="left", fontsize=17, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _dashboard_markdown(
    *,
    scorecard: pd.DataFrame,
    phase2_readiness: dict[str, Any],
    future_oos: dict[str, Any],
) -> str:
    blockers = phase2_readiness.get("blockers", []) or []
    lines = [
        "# yeniBot Phase 1 Model Performance Dashboard",
        "",
        f"- Active charter: `{phase2_readiness.get('active_validation_charter', 'unknown')}`",
        f"- Model evidence gates: `{'PASS' if _model_evidence_passed(phase2_readiness) else 'REVIEW'}`",
        f"- Phase 2 readiness: `{'READY' if phase2_readiness.get('ready_for_phase2') else 'BLOCKED'}`",
        f"- Active blockers: `{', '.join(blockers) if blockers else 'none'}`",
        f"- Future unseen OOS: `{future_oos.get('new_labeled_rows', 0)} / {future_oos.get('min_rows', 0)}` labeled rows",
        "",
        "![Model scorecard](model_scorecard.png)",
        "",
        "## Core Evidence",
        "",
    ]
    visible_metrics = [
        "mean_rank_ic",
        "positive_fold_fraction",
        "positive_fold_sign_test_pvalue",
        "random_effects_lower_ci_min",
        "prauc_lift_vs_prevalence",
        "precision_lift_vs_prevalence",
        "f1_skill_vs_rate_matched_random",
        "positive_f1_skill_fold_fraction",
        "positive_forward_return_fold_fraction",
        "prediction_long_rate",
        "rank_ic_std",
        "raw_long_f1",
        "top_10_cv_label_lift",
        "top_10_cv_forward_return",
        "top_10_holdout_forward_return",
        "brier_score",
        "brier_skill_vs_climatology",
        "log_loss",
        "log_loss_skill_vs_climatology",
        "ece_equal_count",
    ]
    visible = scorecard.loc[scorecard["metric"].isin(visible_metrics), ["category", "metric", "scope", "value", "target", "status", "role"]]
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    lines.extend(
        [
            "",
            "## Visual Diagnostics",
            "",
            "![Rank IC stability](rank_ic_stability.png)",
            "",
            "![Classification quality](classification_quality.png)",
            "",
            "![Score-band payoff](score_band_payoff.png)",
            "",
            "## Interpretation Rules",
            "",
            "- Walk-forward OOS measures model evidence before the frozen holdout.",
            "- The seen holdout is diagnostic only and must not be used for retuning.",
            "- Raw Rank IC std and raw F1 remain visible monitors under `v4_evidence`; they are not hidden.",
            "- Score lift is not sufficient without positive forward-return/payoff alignment.",
            "- Sharpe, drawdown, turnover, fees, slippage, and PnL belong to Phase 2 and are intentionally absent.",
            "- Promotion remains blocked until the frozen candidate passes prediction-only future unseen OOS.",
            "",
            "## Methodology References",
            "",
            "- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html",
            "- Scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html",
            "- Lahiri, *Bootstraps for Time Series*: https://projecteuclid.org/journals/statistical-science/volume-17/issue-1/Bootstraps-for-Time-Series/10.1214/ss/1023798998.pdf",
            "- Bailey et al., *The Probability of Backtest Overfitting*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
        ]
    )
    return "\n".join(lines)


def write_model_performance_dashboard(
    report_dir: str | Path,
    *,
    entries: list[dict[str, Any]],
    comparison: pd.DataFrame,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    rank_ic_aggregate_evidence: pd.DataFrame,
    classification_skill_summary: pd.DataFrame,
    probability_quality_summary: pd.DataFrame,
    payoff_alignment: pd.DataFrame,
    seed_stability: pd.DataFrame,
    phase2_readiness: dict[str, Any],
    future_oos_readiness: dict[str, Any],
    control_profile: str,
) -> dict[str, Any]:
    """Write compact tables and PNGs that summarize Phase 1 model evidence."""

    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    entry = _control_entry(entries, control_profile)
    predictions = _test_predictions(entry.get("predictions", pd.DataFrame())).copy() if entry else pd.DataFrame()
    calibration = _calibration_frame(predictions)
    pr_curve = _precision_recall_frame(predictions)
    scorecard = _scorecard_frame(
        phase2_readiness=phase2_readiness,
        comparison=comparison,
        rank_ic_evidence=rank_ic_aggregate_evidence,
        classification_skill=classification_skill_summary,
        probability_quality=probability_quality_summary,
        fold_stability=fold_stability_summary,
        payoff_alignment=payoff_alignment,
        seed_stability=seed_stability,
        future_oos=future_oos_readiness,
        control_profile=control_profile,
    )
    definitions = _metric_definitions()

    scorecard.to_csv(path / "model_performance_scorecard.csv", index=False)
    calibration.to_csv(path / "model_calibration_reliability.csv", index=False)
    pr_curve.to_csv(path / "model_precision_recall_curve.csv", index=False)
    definitions.to_csv(path / "model_metric_definitions.csv", index=False)
    _save_scorecard_figure(
        path / "model_scorecard.png",
        scorecard=scorecard,
        phase2_readiness=phase2_readiness,
        future_oos=future_oos_readiness,
    )
    _save_rank_ic_figure(
        path / "rank_ic_stability.png",
        fold_stability=fold_stability_forensics,
        rank_evidence=rank_ic_aggregate_evidence,
        control_profile=control_profile,
    )
    _save_classification_figure(
        path / "classification_quality.png",
        predictions=predictions,
        calibration=calibration,
        pr_curve=pr_curve,
    )
    _save_payoff_figure(
        path / "score_band_payoff.png",
        payoff_alignment=payoff_alignment,
        control_profile=control_profile,
    )
    markdown = _dashboard_markdown(
        scorecard=scorecard,
        phase2_readiness=phase2_readiness,
        future_oos=future_oos_readiness,
    )
    (path / "model_performance_dashboard.md").write_text(markdown, encoding="utf-8")
    summary = {
        "active_charter": phase2_readiness.get("active_validation_charter"),
        "model_evidence_passed": _model_evidence_passed(phase2_readiness),
        "phase2_ready": bool(phase2_readiness.get("ready_for_phase2", False)),
        "blockers": phase2_readiness.get("blockers", []) or [],
        "future_oos": future_oos_readiness,
        "artifacts": [
            "model_performance_dashboard.md",
            "model_performance_scorecard.csv",
            "model_metric_definitions.csv",
            "model_calibration_reliability.csv",
            "model_precision_recall_curve.csv",
            "model_scorecard.png",
            "rank_ic_stability.png",
            "classification_quality.png",
            "score_band_payoff.png",
        ],
    }
    _write_json(path / "model_performance_summary.json", _json_ready(summary))
    return {
        "scorecard": scorecard,
        "calibration": calibration,
        "precision_recall": pr_curve,
        "definitions": definitions,
        "summary": summary,
    }
