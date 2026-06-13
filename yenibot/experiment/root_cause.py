"""Phase 1 blocker diagnosis and compact root-cause reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _diagnostic_candidate_type,
    _first_frame_row,
    _float,
    _fmt_metric,
    _is_stability_scope,
    _write_json,
)

from yenibot.experiment.configuration import (
    _experiment_policy_guard,
)

from yenibot.experiment.training import (
    _test_predictions,
)

__all__ = [
    '_control_comparison_row',
    '_control_gap_row',
    '_phase1_blocker_action_plan_frame',
    '_phase1_blocker_action_plan_markdown',
    '_write_phase1_blocker_action_plan',
    '_threshold_oracle_gap_frame',
    '_memory_text_matches_family',
    '_historical_experiment_memory_audit_frame',
    '_bad_fold_mechanism_summary_frame',
    '_entry_forward_column',
    '_prediction_error_audit_frame',
    '_phase1_blocker_root_cause_frame',
    '_phase1_decision_ladder_payload',
    '_phase1_blocker_root_cause_markdown',
    '_write_root_cause_reports',
]

def _control_comparison_row(comparison: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    if comparison.empty:
        return {}
    full_mask = (comparison["profile"].astype(str) == str(control_profile)) & (
        comparison["fold_scope"].astype(str) == "full"
    )
    row = _first_frame_row(comparison, full_mask)
    if row:
        return row
    control_mask = comparison["profile"].astype(str) == str(control_profile)
    return _first_frame_row(comparison, control_mask)

def _control_gap_row(performance_gap_analysis: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    if performance_gap_analysis.empty:
        return {}
    mask = (performance_gap_analysis["candidate"].astype(str) == str(control_profile)) & (
        performance_gap_analysis["fold_scope"].astype(str) == "full"
    )
    row = _first_frame_row(performance_gap_analysis, mask)
    if row:
        return row
    return _first_frame_row(performance_gap_analysis, performance_gap_analysis["candidate"].astype(str) == str(control_profile))

def _phase1_blocker_action_plan_frame(
    *,
    comparison: pd.DataFrame,
    profile_blend: pd.DataFrame,
    performance_gap_analysis: pd.DataFrame,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
    payoff_policy_robustness_summary: pd.DataFrame,
    future_oos_candidate_plan: pd.DataFrame,
    phase2_readiness: dict[str, Any] | None,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "priority",
        "blocker",
        "severity",
        "control_profile",
        "metric_value",
        "target",
        "passed",
        "evidence",
        "recommended_action",
        "allowed_now",
        "requires_02_03",
        "requires_04",
        "next_notebook",
        "promotion_allowed_now",
        "source_files",
        "notes",
    ]
    control_profile = str(settings.get("control_profile", ""))
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    readiness = phase2_readiness or {}
    checks = readiness.get("checks", {}) or {}
    blockers = {str(item) for item in readiness.get("blockers", []) or []}
    active_charter = str(readiness.get("active_validation_charter") or "v3_legacy")
    evidence_charter_active = active_charter != "v3_legacy"
    rows: list[dict[str, Any]] = []

    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    max_rank_ic_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    min_positive_ic = float(_cfg(config, ["validation", "min_positive_ic_fraction"], 0.75))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))

    control = _control_comparison_row(comparison, control_profile)
    control_gap = _control_gap_row(performance_gap_analysis, control_profile)
    control_stability = _first_frame_row(
        fold_stability_summary,
        (fold_stability_summary["candidate"].astype(str) == control_profile)
        & (fold_stability_summary["fold_scope"].astype(str) == "full")
        if not fold_stability_summary.empty
        else None,
    )
    control_thresholds = (
        threshold_forensics.loc[
            (threshold_forensics["candidate"].astype(str) == control_profile)
            & (threshold_forensics["fold_scope"].astype(str) == "full")
        ].copy()
        if not threshold_forensics.empty
        else pd.DataFrame()
    )
    control_payoff = (
        payoff_policy_robustness_summary.loc[
            (payoff_policy_robustness_summary["candidate"].astype(str) == control_profile)
            & (payoff_policy_robustness_summary["evaluation_scope"].astype(str) == "cv_test")
        ].copy()
        if not payoff_policy_robustness_summary.empty
        and {"candidate", "evaluation_scope"}.issubset(payoff_policy_robustness_summary.columns)
        else pd.DataFrame()
    )

    def add_row(
        *,
        priority: int,
        blocker: str,
        severity: str,
        metric_value: Any = np.nan,
        target: str = "",
        passed: bool = False,
        evidence: str,
        recommended_action: str,
        allowed_now: bool,
        requires_02_03: bool,
        requires_04: bool,
        next_notebook: str,
        promotion_allowed_now: bool,
        source_files: str,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "priority": int(priority),
                "blocker": blocker,
                "severity": severity,
                "control_profile": control_profile,
                "metric_value": metric_value,
                "target": target,
                "passed": bool(passed),
                "evidence": evidence,
                "recommended_action": recommended_action,
                "allowed_now": bool(allowed_now),
                "requires_02_03": bool(requires_02_03),
                "requires_04": bool(requires_04),
                "next_notebook": next_notebook,
                "promotion_allowed_now": bool(promotion_allowed_now),
                "source_files": source_files,
                "notes": notes,
            }
        )

    missing_selected = performance_gap_analysis.empty and comparison.empty
    add_row(
        priority=1,
        blocker="experiment_integrity",
        severity="critical" if missing_selected else "ok",
        metric_value=0 if missing_selected else 1,
        target="all selected profiles present in comparison and diagnostics",
        passed=not missing_selected,
        evidence=(
            "No comparison rows were available; rerun 04 before trusting diagnostics."
            if missing_selected
            else f"{len(comparison)} comparison rows and {len(performance_gap_analysis)} performance-gap rows are available."
        ),
        recommended_action=(
            "Rerun 04_training_walk_forward.ipynb to produce completed profile predictions."
            if missing_selected
            else "Continue using 05 diagnostics; experiment integrity is sufficient for review."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=missing_selected,
        next_notebook="04" if missing_selected else "05",
        promotion_allowed_now=False,
        source_files="profile_comparison.csv;performance_gap_analysis.csv;experiment_selection.csv",
        notes="This check prevents empty or stale diagnostics from being interpreted as model performance.",
    )

    mean_rank_ic = _float(control, "mean_rank_ic")
    positive_fraction = _float(control, "positive_ic_fraction")
    mean_passed = bool(np.isfinite(mean_rank_ic) and mean_rank_ic > target_rank_ic)
    positive_passed = bool(np.isfinite(positive_fraction) and positive_fraction >= min_positive_ic)
    add_row(
        priority=2,
        blocker="signal_strength",
        severity="ok" if mean_passed and positive_passed else "high",
        metric_value=mean_rank_ic,
        target=f"mean_rank_ic>{target_rank_ic:.3f}; positive_ic_fraction>={min_positive_ic:.2f}",
        passed=mean_passed and positive_passed,
        evidence=(
            f"Control mean Rank IC={_fmt_metric(mean_rank_ic)}, "
            f"positive IC fraction={_fmt_metric(positive_fraction)}."
        ),
        recommended_action=(
            "Keep the current control as the safe benchmark; do not weaken it with holdout-derived promotions."
            if mean_passed and positive_passed
            else "Stop promotion review and return to feature quality only if this remains weak on future OOS."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none",
        promotion_allowed_now=False,
        source_files="profile_comparison.csv;phase2_readiness.json",
        notes="Mean signal is not the current bottleneck; stability and deployment-quality thresholding are.",
    )

    rank_ic_std = _float(control, "std_rank_ic")
    legacy_std_passed = bool(np.isfinite(rank_ic_std) and rank_ic_std <= max_rank_ic_std)
    std_passed = bool(evidence_charter_active or legacy_std_passed)
    worst_fold = control_stability.get("worst_fold", "NA") if control_stability else "NA"
    worst_ic = control_stability.get("worst_fold_rank_ic", np.nan) if control_stability else np.nan
    top5_var = control_stability.get("top_5_variance_contribution", np.nan) if control_stability else np.nan
    add_row(
        priority=3,
        blocker="fold_stability",
        severity="monitor" if evidence_charter_active else "critical" if not std_passed else "ok",
        metric_value=rank_ic_std,
        target=(
            f"monitor_only; legacy std_rank_ic<={max_rank_ic_std:.3f}"
            if evidence_charter_active
            else f"std_rank_ic<={max_rank_ic_std:.3f}"
        ),
        passed=std_passed,
        evidence=(
            f"Control Rank IC std={_fmt_metric(rank_ic_std)}; worst fold={worst_fold} "
            f"Rank IC={_fmt_metric(worst_ic)}; top-5 variance contribution={_fmt_metric(top5_var)}."
        ),
        recommended_action=(
            "Keep fold stability visible as a risk monitor. Do not retrain or create a new profile while the frozen "
            "future-OOS candidate is awaiting evaluation."
            if evidence_charter_active
            else "Use fold_stability_forensics to isolate recurring bad-fold regimes. Do not create new broad profiles or tune "
            "against holdout; any new hypothesis must be pre-registered and checked on CV/future OOS."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files=(
            "fold_stability_summary.csv;fold_stability_forensics.csv;bad_fold_signature.csv;"
            "score_separation_forensics.csv;feature_drift_forensics.csv;feature_family_drift_summary.csv;"
            "score_distribution_shift.csv;score_distribution_shift_summary.csv;"
            "fold_reliability_gate.csv;fold_reliability_gate_summary.csv;"
            "regime_stability_forensics.csv;regime_stability_summary.csv"
        ),
        notes=(
            "The active evidence charter evaluates fold sign consistency and random-effects confidence bounds; raw std remains visible."
            if evidence_charter_active
            else "This is the main remaining statistical blocker after mean IC improved."
        ),
    )

    official_f1 = _float(control, "test_f1_at_official_threshold", _float(control, "test_f1_at_guarded_threshold"))
    official_rate = _float(
        control,
        "test_pred_long_rate_at_official_threshold",
        _float(control, "test_pred_long_rate_at_guarded_threshold"),
    )
    official_source = str(control.get("official_threshold_source") or control.get("guarded_threshold_source") or "")
    calibrated_f1 = _float(control, "test_f1_at_calibrated_guarded_threshold")
    calibrated_rate = _float(control, "test_pred_long_rate_at_calibrated_guarded_threshold")
    guarded_f1 = _float(control, "test_f1_at_guarded_threshold")
    guarded_rate = _float(control, "test_pred_long_rate_at_guarded_threshold")
    selected_rate = _float(control, "test_pred_long_rate_at_selected_threshold")
    guarded_source = str(control.get("guarded_threshold_source", ""))
    legacy_threshold_passed = bool(
        np.isfinite(official_f1)
        and official_f1 > min_long_f1
        and official_rate <= max_pred_rate
    )
    active_classification_blockers = {
        blocker
        for blocker in blockers
        if blocker.startswith("active_charter_")
        and any(
            token in blocker
            for token in (
                "prauc",
                "precision",
                "f1_skill",
                "forward_return",
                "prediction_long_rate",
            )
        )
    }
    threshold_passed = bool(
        not active_classification_blockers
        if evidence_charter_active
        else legacy_threshold_passed
    )
    issue_counts = (
        control_thresholds["primary_issue"].value_counts().to_dict()
        if not control_thresholds.empty and "primary_issue" in control_thresholds.columns
        else {}
    )
    add_row(
        priority=4,
        blocker="official_threshold_f1",
        severity="monitor" if evidence_charter_active else "critical" if not threshold_passed else "ok",
        metric_value=official_f1,
        target=(
            f"raw_f1_monitor; active skill and pred_long_rate<={max_pred_rate:.2f}"
            if evidence_charter_active
            else f"official_f1>{min_long_f1:.2f}; pred_long_rate<={max_pred_rate:.2f}"
        ),
        passed=threshold_passed,
        evidence=(
            f"Official F1={_fmt_metric(official_f1)} from {official_source or 'NA'}; "
            f"official pred-long rate={_fmt_metric(official_rate)}; "
            f"raw guarded F1={_fmt_metric(guarded_f1)} from {guarded_source or 'NA'}; "
            f"guarded pred-long rate={_fmt_metric(guarded_rate)}; selected pred-long rate={_fmt_metric(selected_rate)}; "
            f"calibrated guarded F1={_fmt_metric(calibrated_f1)}; calibrated pred-long rate={_fmt_metric(calibrated_rate)}; "
            f"threshold issue counts={issue_counts}."
        ),
        recommended_action=(
            "Keep threshold transfer and raw F1 visible as risk diagnostics. Do not alter the frozen threshold policy "
            "before future unseen OOS evaluation."
            if evidence_charter_active
            else "Optimize score separation/calibration on CV only. Selected-threshold F1 is not official when it exceeds "
            "the pred-long-rate guardrail."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files=(
            "threshold_forensics.csv;threshold_policy_review.csv;threshold_transfer_review.csv;"
            "regime_threshold_policy_by_fold.csv;regime_threshold_policy_summary.csv;"
            "score_separation_forensics.csv;probability_quality_forensics.csv;probability_quality_summary.csv;"
            "feature_drift_forensics.csv;profile_comparison.csv;phase2_readiness.json"
        ),
        notes=(
            "The active charter uses rate-matched F1 skill, PRAUC/precision lift, realized return consistency, and the prediction-rate guardrail."
            if evidence_charter_active
            else "This prevents an unrealistically broad long gate from masking deployment risk."
        ),
    )

    top10_lift = _float(control, "top_10_lift_global")
    holdout_top10_return = _float(control_gap, "holdout_top_10_forward_return_global")
    top10_forward = _float(control_gap, "cv_top_10_forward_return_global")
    payoff_pass = bool(np.isfinite(top10_lift) and top10_lift > 1.0)
    if np.isfinite(holdout_top10_return) and holdout_top10_return <= 0:
        payoff_pass = False
    top_payoff_rows = (
        control_payoff.loc[control_payoff["band"].astype(str) == "top_10"]
        if not control_payoff.empty and "band" in control_payoff.columns
        else pd.DataFrame()
    )
    top_payoff = top_payoff_rows.iloc[0].to_dict() if not top_payoff_rows.empty else {}
    add_row(
        priority=5,
        blocker="score_band_payoff",
        severity="high" if not payoff_pass else "medium",
        metric_value=top10_lift,
        target="top_10 lift>1.0 and forward-return alignment positive",
        passed=payoff_pass,
        evidence=(
            f"CV top-10 lift={_fmt_metric(top10_lift)}; CV top-10 forward return={_fmt_metric(top10_forward, 6)}; "
            f"holdout top-10 forward return={_fmt_metric(holdout_top10_return, 6)}; "
            f"CV payoff alignment fold rate={_fmt_metric(top_payoff.get('payoff_alignment_fold_rate'))}."
        ),
        recommended_action=(
            "Treat score-band rows as policy diagnostics only. If holdout payoff is weak, wait for future OOS instead of "
            "changing the score band on the seen holdout."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files="payoff_policy_robustness_summary.csv;bad_fold_signature.csv;performance_gap_analysis.csv;holdout_evaluation.csv",
        notes="Label lift alone is insufficient; forward-return payoff must survive unseen data.",
    )

    best_blend = {}
    if not profile_blend.empty:
        sortable = profile_blend.copy()
        if "reviewable" in sortable.columns:
            sortable = sortable.sort_values(
                ["reviewable", "mean_rank_ic", "top_10_lift_global"],
                ascending=[False, False, False],
            )
        else:
            sortable = sortable.sort_values(["mean_rank_ic", "top_10_lift_global"], ascending=[False, False])
        best_blend = sortable.iloc[0].to_dict() if not sortable.empty else {}
    best_blend_name = str(best_blend.get("blend_name") or best_blend.get("profile") or "none")
    best_blend_ic = _float(best_blend, "mean_rank_ic")
    best_blend_std = _float(best_blend, "std_rank_ic")
    best_blend_reviewable = bool(best_blend.get("reviewable", False)) if best_blend else False
    add_row(
        priority=6,
        blocker="candidate_promotion",
        severity="medium",
        metric_value=best_blend_ic,
        target="candidate/blend must beat control gates without worse stability",
        passed=False,
        evidence=(
            f"Best blend by review ordering={best_blend_name}; mean IC={_fmt_metric(best_blend_ic)}; "
            f"std={_fmt_metric(best_blend_std)}; reviewable={best_blend_reviewable}; "
            f"profile search locked={bool(guard.get('profile_search_locked', False))}."
        ),
        recommended_action=(
            "Keep the control profile as the working baseline. Use pre-registered future-OOS candidates only; do not "
            "promote current-holdout winners."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none",
        promotion_allowed_now=False,
        source_files="profile_blend.csv;profile_comparison.csv;future_oos_candidate_plan.csv",
        notes="This row intentionally blocks opportunistic promotion from already-seen diagnostics.",
    )

    future_ready = bool(guard.get("future_oos_ready", False))
    future_plan = _first_frame_row(future_oos_candidate_plan)
    min_remaining = guard.get("min_new_bars_remaining", future_plan.get("min_new_bars_remaining", "NA"))
    min_ready_at = future_plan.get("min_ready_at") or guard.get("min_ready_at") or ""
    preferred_ready_at = future_plan.get("preferred_ready_at") or guard.get("preferred_ready_at") or ""
    future_blocked = "future_unseen_oos_not_ready" in blockers or not future_ready
    add_row(
        priority=7,
        blocker="future_unseen_oos",
        severity="critical" if future_blocked else "ok",
        metric_value=0 if future_blocked else 1,
        target="future_oos_ready=True before promotion",
        passed=not future_blocked,
        evidence=(
            f"future_oos_ready={future_ready}; min bars remaining={min_remaining}; "
            f"min_ready_at={min_ready_at}; preferred_ready_at={preferred_ready_at}."
        ),
        recommended_action=(
            "Wait for fresh unseen bars after the anchor, then evaluate only pre-registered candidates. Do not roll "
            "or retune the frozen holdout."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none_until_future_oos_ready",
        promotion_allowed_now=future_ready,
        source_files="future_oos_candidate_plan.csv;experiment_policy_guard.csv;phase2_readiness.json",
        notes="This is a governance gate, not a model-performance tweak.",
    )

    phase2_passed = bool(readiness.get("ready_for_phase2", False) or readiness.get("passed", False))
    add_row(
        priority=8,
        blocker="phase2_decision",
        severity="critical" if not phase2_passed else "ok",
        metric_value=1 if phase2_passed else 0,
        target="all Phase 1 readiness checks pass",
        passed=phase2_passed,
        evidence=(
            f"Phase 2 status={readiness.get('status', 'NA')}; blockers={sorted(blockers)}; "
            f"checks={checks}."
        ),
        recommended_action=(
            "Do not build Phase 2 execution/backtest code. Continue diagnostics and future-OOS monitoring until the "
            "official gates pass."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=phase2_passed,
        source_files="phase2_readiness.json;phase1_transition_plan.json;auto_review.json",
        notes="This final row keeps the project aligned with SKILLS.md and the Phase 1 boundary.",
    )

    return pd.DataFrame(rows, columns=columns).sort_values("priority").reset_index(drop=True)

def _phase1_blocker_action_plan_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Phase 1 Blocker Action Plan", ""]
    if frame.empty:
        lines.append("No blocker action plan rows were produced.")
        return "\n".join(lines)
    lines.append(
        "This file translates the current diagnostics into operational actions. "
        "Rows marked as not promotion-allowed must not be used to justify Phase 2."
    )
    lines.append("")
    display_cols = [
        "priority",
        "blocker",
        "severity",
        "metric_value",
        "target",
        "passed",
        "recommended_action",
        "next_notebook",
        "promotion_allowed_now",
        "source_files",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append("## Evidence")
    for _, row in frame.iterrows():
        lines.append("")
        lines.append(f"### {int(row['priority'])}. {row['blocker']}")
        lines.append(str(row["evidence"]))
        if str(row.get("notes", "")):
            lines.append("")
            lines.append(f"Notes: {row['notes']}")
    return "\n".join(lines)

def _write_phase1_blocker_action_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "phase1_blocker_action_plan.csv", index=False)
    (path / "phase1_blocker_action_plan.md").write_text(
        _phase1_blocker_action_plan_markdown(frame),
        encoding="utf-8",
    )
    _write_json(path / "phase1_blocker_action_plan.json", {"rows": frame.to_dict(orient="records")})

def _threshold_oracle_gap_frame(threshold_forensics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "oracle_f1_mean",
        "official_f1_mean",
        "selected_f1_mean",
        "constrained_f1_mean",
        "official_gap_to_oracle_mean",
        "oracle_minus_official_f1",
        "selected_gap_to_oracle_mean",
        "constrained_gap_to_oracle_mean",
        "oracle_pass_fold_rate",
        "oracle_reaches_target_rate",
        "official_pass_fold_rate",
        "official_reaches_target_rate",
        "selected_pass_fold_rate",
        "selected_reaches_target_rate",
        "constrained_pass_fold_rate",
        "constrained_reaches_target_rate",
        "official_pred_long_rate_mean",
        "selected_pred_long_rate_mean",
        "constrained_pred_long_rate_mean",
        "official_pred_rate_guardrail_fail_rate",
        "selected_pred_rate_guardrail_fail_rate",
        "dominant_threshold_issue",
        "primary_threshold_issue",
        "threshold_transfer_blocker",
        "root_cause_hint",
        "recommended_action",
    ]
    if threshold_forensics.empty:
        return pd.DataFrame(columns=columns)
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in threshold_forensics.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        oracle = pd.to_numeric(part.get("test_oracle_best_f1"), errors="coerce")
        official = pd.to_numeric(part.get("test_f1_at_official_threshold"), errors="coerce")
        selected = pd.to_numeric(part.get("test_f1_at_selected_threshold"), errors="coerce")
        constrained = pd.to_numeric(part.get("test_f1_at_constrained_threshold"), errors="coerce")
        official_rate = pd.to_numeric(part.get("test_pred_long_rate_at_official_threshold"), errors="coerce")
        selected_rate = pd.to_numeric(part.get("test_pred_long_rate_at_selected_threshold"), errors="coerce")
        constrained_rate = pd.to_numeric(part.get("test_pred_long_rate_at_constrained_threshold"), errors="coerce")
        oracle_pass = oracle >= min_long_f1
        official_pass = (official >= min_long_f1) & (official_rate <= max_pred_rate)
        selected_pass = (selected >= min_long_f1) & (selected_rate <= max_pred_rate)
        constrained_pass = (constrained >= min_long_f1) & (constrained_rate <= max_pred_rate)
        issue_counts = (
            part["primary_issue"].astype(str).value_counts().to_dict()
            if "primary_issue" in part.columns
            else {}
        )
        primary_issue = max(issue_counts, key=issue_counts.get) if issue_counts else ""
        if oracle.notna().any() and float(oracle_pass.mean()) < 0.50:
            hint = "score_not_separable_enough_even_with_oracle_threshold"
            action = "do_not_tune_threshold_first; inspect score ranking and feature drift"
        elif float(official_pass.mean()) < float(oracle_pass.mean()) - 0.20:
            hint = "threshold_transfer_gap_after_oracle_succeeds"
            action = "diagnose threshold transfer and calibration before new features"
        elif float(selected_rate.gt(max_pred_rate).mean()) > 0.25:
            hint = "selected_threshold_too_broad_for_deployment_guardrail"
            action = "use guarded official threshold only; do not count broad selected F1 as Phase 1 pass"
        else:
            hint = "threshold_gap_secondary_to_fold_stability"
            action = "monitor threshold diagnostics while prioritizing fold stability"
        rows.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "fold_scope": str(fold_scope),
                "fold_count": int(part["fold"].nunique()) if "fold" in part.columns else int(len(part)),
                "oracle_f1_mean": float(oracle.mean()) if oracle.notna().any() else np.nan,
                "official_f1_mean": float(official.mean()) if official.notna().any() else np.nan,
                "selected_f1_mean": float(selected.mean()) if selected.notna().any() else np.nan,
                "constrained_f1_mean": float(constrained.mean()) if constrained.notna().any() else np.nan,
                "official_gap_to_oracle_mean": float((oracle - official).mean()) if oracle.notna().any() else np.nan,
                "oracle_minus_official_f1": float((oracle - official).mean()) if oracle.notna().any() else np.nan,
                "selected_gap_to_oracle_mean": float((oracle - selected).mean()) if oracle.notna().any() else np.nan,
                "constrained_gap_to_oracle_mean": float((oracle - constrained).mean()) if oracle.notna().any() else np.nan,
                "oracle_pass_fold_rate": float(oracle_pass.mean()) if oracle.notna().any() else np.nan,
                "oracle_reaches_target_rate": float(oracle_pass.mean()) if oracle.notna().any() else np.nan,
                "official_pass_fold_rate": float(official_pass.mean()) if official.notna().any() else np.nan,
                "official_reaches_target_rate": float(official_pass.mean()) if official.notna().any() else np.nan,
                "selected_pass_fold_rate": float(selected_pass.mean()) if selected.notna().any() else np.nan,
                "selected_reaches_target_rate": float(selected_pass.mean()) if selected.notna().any() else np.nan,
                "constrained_pass_fold_rate": float(constrained_pass.mean()) if constrained.notna().any() else np.nan,
                "constrained_reaches_target_rate": float(constrained_pass.mean()) if constrained.notna().any() else np.nan,
                "official_pred_long_rate_mean": float(official_rate.mean()) if official_rate.notna().any() else np.nan,
                "selected_pred_long_rate_mean": float(selected_rate.mean()) if selected_rate.notna().any() else np.nan,
                "constrained_pred_long_rate_mean": float(constrained_rate.mean()) if constrained_rate.notna().any() else np.nan,
                "official_pred_rate_guardrail_fail_rate": float(official_rate.gt(max_pred_rate).mean())
                if official_rate.notna().any()
                else np.nan,
                "selected_pred_rate_guardrail_fail_rate": float(selected_rate.gt(max_pred_rate).mean())
                if selected_rate.notna().any()
                else np.nan,
                "dominant_threshold_issue": json.dumps(issue_counts, sort_keys=True),
                "primary_threshold_issue": primary_issue,
                "threshold_transfer_blocker": bool(hint == "threshold_transfer_gap_after_oracle_succeeds"),
                "root_cause_hint": hint,
                "recommended_action": action,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate_type", "candidate", "fold_scope"],
    ).reset_index(drop=True)

def _memory_text_matches_family(text: str, family: str, feature: str) -> bool:
    haystack = str(text).lower().replace("-", "_")
    family_text = str(family).lower()
    feature_text = str(feature).lower()
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", f"{family_text} {feature_text}")
        if len(token) >= 4 and token not in {"stable", "rank", "zscore", "feature", "profile"}
    }
    if not tokens:
        return False
    return any(token in haystack for token in tokens)

def _historical_experiment_memory_audit_frame(
    feature_family_drift_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "fold_scope",
        "feature_family",
        "top_suspect_feature",
        "suspect_feature",
        "top_likely_issue",
        "related_rejected_profile_count",
        "related_rejected_profiles",
        "matched_rejected_profile",
        "related_rejected_reasons",
        "historical_status",
        "memory_status",
        "recommended_action",
        "recommendation",
    ]
    if feature_family_drift_summary.empty:
        return pd.DataFrame(columns=columns)
    memory = _cfg(config, ["experiments", "experiment_memory"], {}) or {}
    rejected = memory.get("rejected_profiles", {}) or {}
    rejected_items: list[tuple[str, str]] = []
    if isinstance(rejected, dict):
        for profile, value in rejected.items():
            reason = value.get("reason", "") if isinstance(value, dict) else str(value)
            rejected_items.append((str(profile), str(reason)))
    rows: list[dict[str, Any]] = []
    for _, row in feature_family_drift_summary.iterrows():
        family = str(row.get("feature_family", ""))
        feature = str(row.get("top_suspect_feature", ""))
        related = [
            (profile, reason)
            for profile, reason in rejected_items
            if _memory_text_matches_family(f"{profile} {reason}", family, feature)
        ]
        if related:
            status = "related_rejections_found"
            action = "do_not_repeat_direct_ablation; use as diagnostic context only unless a new mechanism is explicit"
        else:
            status = "no_direct_rejection_match"
            action = "eligible_for_hypothesis_design_only_after_root_cause_report_requests_new_features"
        first_profile = related[0][0] if related else ""
        rows.append(
            {
                "candidate": str(row.get("candidate", "")),
                "fold_scope": str(row.get("fold_scope", "")),
                "feature_family": family,
                "top_suspect_feature": feature,
                "suspect_feature": feature,
                "top_likely_issue": str(row.get("top_likely_issue", "")),
                "related_rejected_profile_count": len(related),
                "related_rejected_profiles": ",".join(profile for profile, _ in related[:8]),
                "matched_rejected_profile": first_profile,
                "related_rejected_reasons": " | ".join(reason for _, reason in related[:3]),
                "historical_status": status,
                "memory_status": status,
                "recommended_action": action,
                "recommendation": action,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate", "related_rejected_profile_count"],
        ascending=[True, False],
    ).reset_index(drop=True)

def _bad_fold_mechanism_summary_frame(
    *,
    bad_fold_signature: pd.DataFrame,
    feature_family_drift_summary: pd.DataFrame,
    score_distribution_shift_summary: pd.DataFrame,
    probability_quality_summary: pd.DataFrame,
    historical_memory_audit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "bad_fold_count",
        "bad_rank_ic_mean",
        "bad_score_gap_mean",
        "good_score_gap_mean",
        "bad_label_long_rate_mean",
        "good_label_long_rate_mean",
        "bad_top_10_lift_mean",
        "bad_top_10_forward_return_mean",
        "likely_signature",
        "top_drift_family",
        "top_suspect_feature",
        "historical_status",
        "score_shift_issue",
        "probability_quality_issue",
        "dominant_mechanism",
        "requires_04",
        "recommended_action",
    ]
    if bad_fold_signature.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, sig in bad_fold_signature.iterrows():
        candidate = str(sig.get("candidate", ""))
        fold_scope = str(sig.get("fold_scope", ""))
        drift = _first_frame_row(
            feature_family_drift_summary,
            (feature_family_drift_summary["candidate"].astype(str) == candidate)
            & (feature_family_drift_summary["fold_scope"].astype(str) == fold_scope)
            if not feature_family_drift_summary.empty
            else None,
        )
        shift = _first_frame_row(
            score_distribution_shift_summary,
            (score_distribution_shift_summary["candidate"].astype(str) == candidate)
            & (score_distribution_shift_summary["fold_scope"].astype(str) == fold_scope)
            if not score_distribution_shift_summary.empty
            else None,
        )
        prob = _first_frame_row(
            probability_quality_summary,
            (probability_quality_summary["candidate"].astype(str) == candidate)
            & (probability_quality_summary["fold_scope"].astype(str) == fold_scope)
            if not probability_quality_summary.empty
            else None,
        )
        memory = _first_frame_row(
            historical_memory_audit,
            (historical_memory_audit["candidate"].astype(str) == candidate)
            & (historical_memory_audit["fold_scope"].astype(str) == fold_scope)
            if not historical_memory_audit.empty
            else None,
        )
        likely = str(sig.get("likely_signature", ""))
        if "score_separation_compresses_or_reverses" in likely:
            mechanism = "score_ranking_reversal_not_label_balance"
            action = "inspect compact prediction_error_audit; do not solve with threshold smoothing alone"
        elif "label_distribution_shift" in likely:
            mechanism = "label_distribution_shift"
            action = "review label quality by regime before changing features"
        elif str(drift.get("top_likely_issue", "")).startswith("feature"):
            mechanism = "feature_family_signal_reversal"
            action = "design only a new transform/gating hypothesis; do not repeat rejected direct ablations"
        else:
            mechanism = "mixed_or_unresolved_fold_instability"
            action = "keep 04 paused and use diagnostics to narrow a pre-registered hypothesis"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": str(sig.get("candidate_type", "")),
                "fold_scope": fold_scope,
                "bad_fold_count": _float(sig, "bad_fold_count"),
                "bad_rank_ic_mean": _float(sig, "bad_rank_ic_mean"),
                "bad_score_gap_mean": _float(sig, "bad_score_gap_mean"),
                "good_score_gap_mean": _float(sig, "good_score_gap_mean"),
                "bad_label_long_rate_mean": _float(sig, "bad_label_long_rate_mean"),
                "good_label_long_rate_mean": _float(sig, "good_label_long_rate_mean"),
                "bad_top_10_lift_mean": _float(sig, "bad_top_10_lift_mean"),
                "bad_top_10_forward_return_mean": _float(sig, "bad_top_10_forward_return_mean"),
                "likely_signature": likely,
                "top_drift_family": str(drift.get("feature_family", "")),
                "top_suspect_feature": str(drift.get("top_suspect_feature", "")),
                "historical_status": str(memory.get("historical_status", "")),
                "score_shift_issue": str(shift.get("score_shift_issue", "")),
                "probability_quality_issue": str(prob.get("probability_quality_issue", "")),
                "dominant_mechanism": mechanism,
                "requires_04": False,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate_type", "candidate", "bad_fold_count"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

def _entry_forward_column(frame: pd.DataFrame) -> str:
    if "forward_return" in frame.columns:
        return "forward_return"
    for column in frame.columns:
        if str(column).startswith("fwd_return_"):
            return str(column)
    return ""

def _prediction_error_audit_frame(
    entries: list[dict[str, Any]],
    score_separation_forensics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "case_type",
        "fold",
        "timestamp",
        "prob_long",
        "label",
        "forward_return",
        "tb_return",
        "score_rank_pct",
        "rank_ic",
        "score_gap_pos_minus_neg",
        "official_threshold",
        "is_top_decile",
        "note",
    ]
    rows: list[dict[str, Any]] = []
    max_rows_per_case = int(_cfg(config, ["experiments", "diagnostics", "prediction_error_audit_rows_per_case"], 5))
    if score_separation_forensics.empty:
        return pd.DataFrame(columns=columns)
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        candidate = str(entry.get("profile", ""))
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test = _test_predictions(predictions).copy()
        if not {"fold", "label", "prob_long"}.issubset(test.columns):
            continue
        forward_col = _entry_forward_column(test)
        if not forward_col:
            test["forward_return"] = np.nan
            forward_col = "forward_return"
        sf = score_separation_forensics.loc[
            (score_separation_forensics["candidate"].astype(str) == candidate)
            & (score_separation_forensics["fold_scope"].astype(str) == fold_scope)
        ].copy()
        if sf.empty:
            continue
        worst_folds = list(
            pd.to_numeric(sf.sort_values("rank_ic", ascending=True)["fold"], errors="coerce").dropna().astype(int).head(3)
        )
        good_folds = list(
            pd.to_numeric(sf.sort_values("rank_ic", ascending=False)["fold"], errors="coerce").dropna().astype(int).head(2)
        )
        score_by_fold = {int(row["fold"]): row.to_dict() for _, row in sf.dropna(subset=["fold"]).iterrows()}

        def add_case(case_type: str, part: pd.DataFrame, note: str) -> None:
            for _, item in part.head(max_rows_per_case).iterrows():
                fold_id = int(item.get("fold"))
                sf_row = score_by_fold.get(fold_id, {})
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": _diagnostic_candidate_type(fold_scope),
                        "fold_scope": fold_scope,
                        "case_type": case_type,
                        "fold": fold_id,
                        "timestamp": str(item.get("timestamp", "")),
                        "prob_long": _float(item, "prob_long"),
                        "label": int(item.get("label")) if pd.notna(item.get("label")) else np.nan,
                        "forward_return": _float(item, forward_col),
                        "tb_return": _float(item, "tb_return"),
                        "score_rank_pct": _float(item, "_score_rank_pct"),
                        "rank_ic": _float(sf_row, "rank_ic"),
                        "score_gap_pos_minus_neg": _float(sf_row, "score_gap_pos_minus_neg"),
                        "official_threshold": _float(sf_row, "official_threshold"),
                        "is_top_decile": bool(_float(item, "_score_rank_pct") >= 0.90),
                        "note": note,
                    }
                )

        clean = test.replace([np.inf, -np.inf], np.nan).dropna(subset=["fold", "label", "prob_long"]).copy()
        clean["prob_long"] = pd.to_numeric(clean["prob_long"], errors="coerce")
        clean[forward_col] = pd.to_numeric(clean[forward_col], errors="coerce")
        clean["_score_rank_pct"] = clean.groupby("fold")["prob_long"].rank(pct=True, method="average")
        for fold_id in worst_folds:
            part = clean.loc[clean["fold"].astype(int) == fold_id].copy()
            if part.empty:
                continue
            false_pos = part.loc[(part["label"].astype(int) == 0) & (part["_score_rank_pct"] >= 0.90)].sort_values(
                ["prob_long", forward_col],
                ascending=[False, True],
            )
            add_case("worst_fold_top_score_false_positive", false_pos, "Top-score not-long rows in a bad fold.")
            false_neg = part.loc[part["label"].astype(int) == 1].sort_values(
                ["prob_long", forward_col],
                ascending=[True, False],
            )
            add_case("worst_fold_low_score_false_negative", false_neg, "Long-label rows ranked too low in a bad fold.")
        for fold_id in good_folds:
            part = clean.loc[clean["fold"].astype(int) == fold_id].copy()
            if part.empty:
                continue
            true_pos = part.loc[(part["label"].astype(int) == 1) & (part["_score_rank_pct"] >= 0.80)].sort_values(
                ["prob_long", forward_col],
                ascending=[False, False],
            )
            add_case("good_fold_reference_true_positive", true_pos, "High-score long rows in a good fold.")
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate_type", "candidate", "fold_scope", "case_type", "fold"],
    ).reset_index(drop=True)

def _phase1_blocker_root_cause_frame(
    *,
    phase1_blocker_action_plan: pd.DataFrame,
    threshold_oracle_gap: pd.DataFrame,
    bad_fold_mechanism_summary: pd.DataFrame,
    historical_experiment_memory_audit: pd.DataFrame,
    phase2_readiness: dict[str, Any] | None,
    settings: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "priority",
        "blocker",
        "root_cause",
        "evidence",
        "recommended_action",
        "requires_02_03",
        "requires_04",
        "run_04_now",
        "full_zip_required",
        "promotion_allowed_now",
        "source_files",
    ]
    control = str(settings.get("control_profile") or _cfg(config, ["experiments", "control_profile"], ""))
    rows: list[dict[str, Any]] = []
    readiness = phase2_readiness or {}
    blockers = {str(item) for item in readiness.get("blockers", []) or []}
    active_charter = str(readiness.get("active_validation_charter") or "v3_legacy")
    evidence_charter_active = active_charter != "v3_legacy"
    model_blockers = {
        blocker
        for blocker in blockers
        if blocker
        not in {
            "future_unseen_oos_not_ready",
            "future_unseen_oos_not_evaluated",
        }
    }
    action_by_blocker = {
        str(row.get("blocker")): row.to_dict()
        for _, row in phase1_blocker_action_plan.iterrows()
    } if not phase1_blocker_action_plan.empty else {}
    control_oracle = _first_frame_row(
        threshold_oracle_gap,
        (threshold_oracle_gap["candidate"].astype(str) == control)
        & (threshold_oracle_gap["fold_scope"].astype(str) == "full")
        if not threshold_oracle_gap.empty
        else None,
    )
    control_mechanism = _first_frame_row(
        bad_fold_mechanism_summary,
        (bad_fold_mechanism_summary["candidate"].astype(str) == control)
        & (bad_fold_mechanism_summary["fold_scope"].astype(str) == "full")
        if not bad_fold_mechanism_summary.empty
        else None,
    )
    repeated_count = (
        int((pd.to_numeric(historical_experiment_memory_audit["related_rejected_profile_count"], errors="coerce") > 0).sum())
        if not historical_experiment_memory_audit.empty
        and "related_rejected_profile_count" in historical_experiment_memory_audit.columns
        else 0
    )

    def add(priority: int, blocker: str, root_cause: str, evidence: str, action: str, *, requires_04: bool = False) -> None:
        action_row = action_by_blocker.get(blocker, {})
        rows.append(
            {
                "priority": priority,
                "blocker": blocker,
                "root_cause": root_cause,
                "evidence": evidence,
                "recommended_action": action,
                "requires_02_03": False,
                "requires_04": bool(requires_04),
                "run_04_now": bool(requires_04 and "future_unseen_oos_not_ready" not in blockers),
                "full_zip_required": False,
                "promotion_allowed_now": bool(action_row.get("promotion_allowed_now", False)) and not blockers,
                "source_files": str(action_row.get("source_files", "")),
            }
        )

    fold_evidence = (
        f"Control bad-fold signature={control_mechanism.get('likely_signature', '')}; "
        f"top suspect={control_mechanism.get('top_drift_family', '')}/"
        f"{control_mechanism.get('top_suspect_feature', '')}; "
        f"related rejected feature hypotheses={repeated_count}."
    )
    threshold_evidence = (
        f"Oracle F1 mean={_fmt_metric(control_oracle.get('oracle_f1_mean'))}; "
        f"official F1 mean={_fmt_metric(control_oracle.get('official_f1_mean'))}; "
        f"official pass fold rate={_fmt_metric(control_oracle.get('official_pass_fold_rate'))}; "
        f"selected pred-long guardrail fail rate="
        f"{_fmt_metric(control_oracle.get('selected_pred_rate_guardrail_fail_rate'))}."
    )
    if evidence_charter_active and not model_blockers:
        add(
            1,
            "future_unseen_oos",
            "governance_gate_not_model_tuning",
            (
                f"Phase2 blockers={sorted(blockers)}; future OOS ready="
                f"{'future_unseen_oos_not_ready' not in blockers}."
            ),
            "Do not promote from the frozen holdout. Wait for future unseen OOS before Phase 2 promotion.",
        )
        add(
            2,
            "monitor_fold_stability",
            str(control_mechanism.get("dominant_mechanism") or "score_ranking_reversal_not_label_balance"),
            fold_evidence,
            "Monitor without retraining while the frozen future-OOS candidate is pending.",
        )
        add(
            3,
            "monitor_threshold_quality",
            str(control_oracle.get("root_cause_hint") or "threshold_gap_secondary_to_score_separation"),
            threshold_evidence,
            "Monitor threshold transfer, but do not change the frozen policy before unseen evaluation.",
        )
        add(
            4,
            "historical_experiment_memory",
            "repeated_direct_ablation_risk",
            f"{repeated_count} current drift rows match already rejected profile families or direct ablations.",
            "Before proposing a new profile, require historical_experiment_memory_audit to show the idea is not a repeated rejected method.",
        )
    else:
        add(
            1,
            "fold_stability",
            str(control_mechanism.get("dominant_mechanism") or "score_ranking_reversal_not_label_balance"),
            fold_evidence,
            "Use bad-fold mechanism and prediction_error_audit before designing a new pre-registered hypothesis.",
        )
        add(
            2,
            "official_threshold_f1",
            str(control_oracle.get("root_cause_hint") or "threshold_gap_secondary_to_score_separation"),
            threshold_evidence,
            "Separate threshold-transfer work from score-separation work.",
        )
        add(
            3,
            "historical_experiment_memory",
            "repeated_direct_ablation_risk",
            f"{repeated_count} current drift rows match already rejected profile families or direct ablations.",
            "Before proposing a new profile, require historical_experiment_memory_audit to show the idea is not a repeated rejected method.",
        )
        add(
            4,
            "future_unseen_oos",
            "governance_gate_not_model_tuning",
            (
                f"Phase2 blockers={sorted(blockers)}; future OOS ready="
                f"{'future_unseen_oos_not_ready' not in blockers}."
            ),
            "Do not promote from the frozen holdout. Wait for future unseen OOS before Phase 2 promotion.",
        )
    return pd.DataFrame(rows, columns=columns)

def _phase1_decision_ladder_payload(
    *,
    phase1_blocker_root_cause: pd.DataFrame,
    threshold_oracle_gap: pd.DataFrame,
    bad_fold_mechanism_summary: pd.DataFrame,
    phase2_readiness: dict[str, Any] | None,
    settings: dict[str, Any],
    recency_policy_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = phase2_readiness or {}
    recency_decision = recency_policy_decision or {}
    blockers = [str(item) for item in readiness.get("blockers", []) or []]
    only_future_oos_blocked = bool(blockers) and set(blockers).issubset(
        {
            "future_unseen_oos_not_ready",
            "future_unseen_oos_not_evaluated",
        }
    )
    future_oos_failed = "future_unseen_oos_candidate_failed" in blockers
    run_04_now = bool(
        not phase1_blocker_root_cause.empty
        and phase1_blocker_root_cause.get("run_04_now", pd.Series(dtype=bool)).astype(bool).any()
    )
    control = str(settings.get("control_profile", ""))
    control_root = _first_frame_row(
        phase1_blocker_root_cause,
        phase1_blocker_root_cause["blocker"].astype(str) == "fold_stability"
        if not phase1_blocker_root_cause.empty
        else None,
    )
    control_oracle = _first_frame_row(
        threshold_oracle_gap,
        (threshold_oracle_gap["candidate"].astype(str) == control)
        & (threshold_oracle_gap["fold_scope"].astype(str) == "full")
        if not threshold_oracle_gap.empty
        else None,
    )
    mechanism = _first_frame_row(
        bad_fold_mechanism_summary,
        (bad_fold_mechanism_summary["candidate"].astype(str) == control)
        & (bad_fold_mechanism_summary["fold_scope"].astype(str) == "full")
        if not bad_fold_mechanism_summary.empty
        else None,
    )
    threshold_transfer_blocker = bool(
        not only_future_oos_blocked
        and not future_oos_failed
        and control_oracle.get("threshold_transfer_blocker", False)
    )
    score_reversal_blocker = bool(
        not only_future_oos_blocked
        and not future_oos_failed
        and str(mechanism.get("dominant_mechanism", "")).startswith("score_ranking_reversal")
    )
    recency_ready = bool(
        recency_decision.get("candidate_ready_for_preregistration", False)
    )
    if future_oos_failed and recency_ready:
        recommended_next_action = (
            "explicitly_review_and_preregister_historical_recency_winner"
        )
    elif future_oos_failed:
        recommended_next_action = (
            "no_recency_policy_cleared_gates_design_new_pre_registered_hypothesis"
        )
    elif only_future_oos_blocked:
        recommended_next_action = "refresh_data_and_run_05_when_future_oos_minimum_is_available"
    elif run_04_now:
        recommended_next_action = "run_04_only_after_pre_registered_hypothesis"
    elif threshold_transfer_blocker:
        recommended_next_action = "run_05_threshold_transfer_diagnostics_only"
    elif score_reversal_blocker:
        recommended_next_action = "run_05_score_reversal_diagnostics_only"
    else:
        recommended_next_action = "run_05_only_and_review_root_cause_reports"
    return {
        "phase2_allowed": bool(readiness.get("ready_for_phase2", False) or readiness.get("passed", False)),
        "blockers": blockers,
        "control_profile": control,
        "run_05_first": True,
        "run_04_required_now": run_04_now,
        "run_02_03_required_now": False,
        "full_zip_required_now": False,
        "next_notebook": "04" if run_04_now else "05",
        "root_cause": (
            "failed_future_oos_ranking_and_payoff_breakdown"
            if future_oos_failed
            else "future_oos_governance_gate"
            if only_future_oos_blocked
            else str(control_root.get("root_cause") or mechanism.get("dominant_mechanism") or "")
        ),
        "threshold_oracle_hint": str(control_oracle.get("root_cause_hint") or ""),
        "threshold_transfer_blocker": threshold_transfer_blocker,
        "threshold_work_required": threshold_transfer_blocker,
        "bad_fold_mechanism": str(mechanism.get("dominant_mechanism") or ""),
        "score_reversal_work_required": score_reversal_blocker,
        "candidate_generation_allowed": bool(
            future_oos_failed
            or (run_04_now and "future_unseen_oos_not_ready" not in blockers)
        ),
        "recency_policy_decision_status": recency_decision.get("status"),
        "recency_recommended_policy": recency_decision.get("recommended_policy"),
        "replacement_candidate_ready_for_preregistration": recency_ready,
        "new_future_oos_anchor_required": bool(future_oos_failed),
        "why_no_04": (
            ""
            if run_04_now or future_oos_failed
            else "frozen candidate must remain unchanged until future unseen OOS evaluation"
            if only_future_oos_blocked
            else "no pre-registered feature or training hypothesis cleared the root-cause and memory audits"
        ),
        "decision": (
            "do_not_proceed_to_phase2"
            if blockers
            else "phase2_review_possible"
        ),
        "recommended_next_action": recommended_next_action,
    }

def _phase1_blocker_root_cause_markdown(
    root_cause: pd.DataFrame,
    decision_ladder: dict[str, Any],
) -> str:
    lines = ["# Phase 1 Root-Cause Review", ""]
    lines.append(f"Decision: `{decision_ladder.get('decision')}`")
    lines.append(f"Next action: `{decision_ladder.get('recommended_next_action')}`")
    lines.append(f"Run 04 required now: `{decision_ladder.get('run_04_required_now')}`")
    lines.append(f"Full zip required now: `{decision_ladder.get('full_zip_required_now')}`")
    lines.append(f"Threshold transfer work required: `{decision_ladder.get('threshold_work_required')}`")
    lines.append(f"Score reversal work required: `{decision_ladder.get('score_reversal_work_required')}`")
    if decision_ladder.get("why_no_04"):
        lines.append(f"Why no 04 now: `{decision_ladder.get('why_no_04')}`")
    lines.append("")
    if root_cause.empty:
        lines.append("No root-cause rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "priority",
        "blocker",
        "root_cause",
        "evidence",
        "recommended_action",
        "requires_04",
        "promotion_allowed_now",
    ]
    visible = root_cause[[column for column in display_cols if column in root_cause.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_root_cause_reports(
    path: Path,
    *,
    phase1_blocker_root_cause: pd.DataFrame,
    threshold_oracle_gap: pd.DataFrame,
    bad_fold_mechanism_summary: pd.DataFrame,
    prediction_error_audit: pd.DataFrame,
    historical_experiment_memory_audit: pd.DataFrame,
    decision_ladder: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    phase1_blocker_root_cause.to_csv(path / "phase1_blocker_root_cause.csv", index=False)
    (path / "phase1_blocker_root_cause.md").write_text(
        _phase1_blocker_root_cause_markdown(phase1_blocker_root_cause, decision_ladder),
        encoding="utf-8",
    )
    _write_json(
        path / "phase1_blocker_root_cause.json",
        {
            "decision_ladder": decision_ladder,
            "rows": phase1_blocker_root_cause.to_dict(orient="records"),
        },
    )
    threshold_oracle_gap.to_csv(path / "threshold_oracle_gap.csv", index=False)
    bad_fold_mechanism_summary.to_csv(path / "bad_fold_mechanism_summary.csv", index=False)
    prediction_error_audit.to_csv(path / "prediction_error_audit.csv", index=False)
    historical_experiment_memory_audit.to_csv(path / "historical_experiment_memory_audit.csv", index=False)
    _write_json(path / "phase1_decision_ladder.json", decision_ladder)
