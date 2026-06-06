"""Score-band payoff alignment and frozen-policy robustness diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from yenibot.experiment.common import (
    _cfg,
    _float,
    _numeric_mean,
    _rank_ic_for_frame,
    _table_markdown,
    _write_json,
)

from yenibot.experiment.holdout import (
    _evaluate_score_policy_on_holdout,
)

from yenibot.experiment.training import (
    _test_predictions,
)

__all__ = [
    '_assign_payoff_score_bins',
    '_resolve_payoff_score_bands',
    '_hit_rate',
    '_payoff_alignment_blockers',
    '_payoff_alignment_action',
    '_payoff_alignment_rows_for_entry',
    '_payoff_alignment_frame',
    '_payoff_alignment_summary_frame',
    '_payoff_alignment_markdown',
    '_write_payoff_alignment',
    '_payoff_policy_rows_for_frame',
    '_payoff_policy_robustness_frame',
    '_payoff_policy_reject_reasons',
    '_payoff_policy_robustness_summary_frame',
    '_payoff_policy_robustness_markdown',
    '_write_payoff_policy_robustness',
    '_frozen_policy_robustness_frame',
    '_write_frozen_policy_robustness',
]

def _assign_payoff_score_bins(predictions: pd.DataFrame, *, score_column: str, bins: int) -> pd.DataFrame:
    required = {"label", score_column}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    frame = predictions.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["label", score_column])
    if frame.empty:
        return frame
    q = max(1, min(int(bins), len(frame)))
    frame["score_bin"] = pd.qcut(
        frame[score_column].rank(method="first"),
        q=q,
        labels=False,
        duplicates="drop",
    )
    return frame.dropna(subset=["score_bin"]).copy()

def _resolve_payoff_score_bands(config: dict[str, Any], actual_bins: int) -> list[dict[str, Any]]:
    max_bin = max(0, int(actual_bins) - 1)
    configured = _cfg(config, ["validation", "score_bands"], None)
    if not configured:
        configured = [
            {"name": "top_10", "min_bin": max_bin, "max_bin": max_bin},
            {"name": "top_20", "min_bin": max(0, int(np.floor(actual_bins * 0.80))), "max_bin": max_bin},
            {"name": "top_30", "min_bin": max(0, int(np.floor(actual_bins * 0.70))), "max_bin": max_bin},
            {"name": "upper_half", "min_bin": max(0, int(np.floor(actual_bins * 0.50))), "max_bin": max_bin},
            {
                "name": "mid_upper_40_90",
                "min_bin": max(0, int(np.floor(actual_bins * 0.40))),
                "max_bin": max(0, max_bin - 1),
            },
        ]
    bands = []
    for item in configured:
        name = str(item.get("name", f"bins_{item.get('min_bin')}_{item.get('max_bin')}"))
        min_bin = min(max(int(item.get("min_bin", max_bin)), 0), max_bin)
        max_item_bin = min(max(int(item.get("max_bin", max_bin)), 0), max_bin)
        if min_bin <= max_item_bin:
            bands.append({"name": name, "min_bin": min_bin, "max_bin": max_item_bin})
    return bands

def _hit_rate(frame: pd.DataFrame, hit_type: str) -> float:
    if "hit_type" not in frame.columns or frame.empty:
        return np.nan
    return float(frame["hit_type"].astype(str).eq(hit_type).mean())

def _payoff_alignment_blockers(row: dict[str, Any]) -> str:
    reasons = []
    if _float(row, "label_lift_vs_base") <= 1.0:
        reasons.append("label_lift_not_above_base")
    if _float(row, "mean_forward_return") <= 0.0:
        reasons.append("forward_return_not_positive")
    if np.isfinite(_float(row, "mean_tb_return")) and _float(row, "mean_tb_return") <= 0.0:
        reasons.append("tb_return_not_positive")
    if np.isfinite(_float(row, "sl_rate_delta_vs_base")) and _float(row, "sl_rate_delta_vs_base") > 0.0:
        reasons.append("sl_rate_above_base")
    if _float(row, "selection_rate") <= 0.0:
        reasons.append("empty_selection")
    return ";".join(reasons)

def _payoff_alignment_action(row: dict[str, Any]) -> str:
    blockers = str(row.get("payoff_blockers", ""))
    if not blockers:
        return "candidate_band_payoff_aligned_monitor_future_oos"
    if "label_lift_not_above_base" in blockers:
        return "weak_label_discrimination_do_not_use_band"
    if "forward_return_not_positive" in blockers or "tb_return_not_positive" in blockers:
        return "investigate_payoff_mismatch_before_new_profile_search"
    if "sl_rate_above_base" in blockers:
        return "inspect_stop_loss_regime_exposure"
    return "monitor"

def _payoff_alignment_rows_for_entry(
    entry: dict[str, Any],
    config: dict[str, Any],
    *,
    evaluation_scope: str,
) -> list[dict[str, Any]]:
    predictions = entry.get("predictions", pd.DataFrame())
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return []
    frame = _test_predictions(predictions)
    score_bins = int(_cfg(config, ["validation", "score_lift_bins"], _cfg(config, ["validation", "calibration_bins"], 10)))
    frame = _assign_payoff_score_bins(frame, score_column="prob_long", bins=score_bins)
    if frame.empty:
        return []

    actual_bins = int(pd.to_numeric(frame["score_bin"], errors="coerce").max()) + 1
    bands = _resolve_payoff_score_bands(config, actual_bins)
    profile = str(entry.get("profile", ""))
    fold_scope = str(entry.get("fold_scope", ""))
    candidate_type = "blend" if fold_scope.startswith("blend_") or profile.startswith("blend_") else "profile"
    base_count = int(len(frame))
    base_long_rate = float(pd.to_numeric(frame["label"], errors="coerce").mean())
    base_forward_return = _numeric_mean(frame, "forward_return")
    base_tb_return = _numeric_mean(frame, "tb_return")
    base_tp_rate = _hit_rate(frame, "tp")
    base_sl_rate = _hit_rate(frame, "sl") + _hit_rate(frame, "both_sl_first")
    base_time_rate = _hit_rate(frame, "time")
    rows = []
    for band in bands:
        part = frame.loc[
            (pd.to_numeric(frame["score_bin"], errors="coerce") >= int(band["min_bin"]))
            & (pd.to_numeric(frame["score_bin"], errors="coerce") <= int(band["max_bin"]))
        ].copy()
        if part.empty:
            continue
        selected_count = int(len(part))
        selected_long_rate = float(pd.to_numeric(part["label"], errors="coerce").mean())
        mean_forward_return = _numeric_mean(part, "forward_return")
        mean_tb_return = _numeric_mean(part, "tb_return")
        tp_rate = _hit_rate(part, "tp")
        sl_rate = _hit_rate(part, "sl") + _hit_rate(part, "both_sl_first")
        time_rate = _hit_rate(part, "time")
        row = {
            "candidate": profile,
            "candidate_type": candidate_type,
            "evaluation_scope": evaluation_scope,
            "fold_scope": fold_scope,
            "band": str(band["name"]),
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "base_count": base_count,
            "selected_count": selected_count,
            "selection_rate": float(selected_count / base_count) if base_count else np.nan,
            "mean_prob_long": _numeric_mean(part, "prob_long"),
            "base_long_rate": base_long_rate,
            "selected_long_rate": selected_long_rate,
            "label_lift_vs_base": float(selected_long_rate / base_long_rate) if base_long_rate > 0 else np.nan,
            "base_forward_return": base_forward_return,
            "mean_forward_return": mean_forward_return,
            "forward_return_delta_vs_base": mean_forward_return - base_forward_return,
            "base_tb_return": base_tb_return,
            "mean_tb_return": mean_tb_return,
            "tb_return_delta_vs_base": mean_tb_return - base_tb_return,
            "base_tp_rate": base_tp_rate,
            "tp_rate": tp_rate,
            "tp_rate_delta_vs_base": tp_rate - base_tp_rate,
            "base_sl_rate": base_sl_rate,
            "sl_rate": sl_rate,
            "sl_rate_delta_vs_base": sl_rate - base_sl_rate,
            "base_time_rate": base_time_rate,
            "time_rate": time_rate,
            "time_rate_delta_vs_base": time_rate - base_time_rate,
            "label_lift_positive_payoff_mismatch": bool(
                selected_long_rate > base_long_rate and np.isfinite(mean_forward_return) and mean_forward_return <= 0.0
            ),
        }
        row["payoff_blockers"] = _payoff_alignment_blockers(row)
        row["payoff_alignment_pass"] = not bool(row["payoff_blockers"])
        row["next_action"] = _payoff_alignment_action(row)
        rows.append(row)
    return rows

def _payoff_alignment_frame(
    entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "fold_scope",
        "band",
        "min_bin",
        "max_bin",
        "base_count",
        "selected_count",
        "selection_rate",
        "mean_prob_long",
        "base_long_rate",
        "selected_long_rate",
        "label_lift_vs_base",
        "base_forward_return",
        "mean_forward_return",
        "forward_return_delta_vs_base",
        "base_tb_return",
        "mean_tb_return",
        "tb_return_delta_vs_base",
        "base_tp_rate",
        "tp_rate",
        "tp_rate_delta_vs_base",
        "base_sl_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
        "base_time_rate",
        "time_rate",
        "time_rate_delta_vs_base",
        "label_lift_positive_payoff_mismatch",
        "payoff_alignment_pass",
        "payoff_blockers",
        "next_action",
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope == "full" or fold_scope.startswith("blend_"):
            rows.extend(_payoff_alignment_rows_for_entry(entry, config, evaluation_scope="cv_test"))
    for entry in holdout_entries:
        rows.extend(_payoff_alignment_rows_for_entry(entry, config, evaluation_scope="holdout"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "candidate", "min_bin", "max_bin"])
        .reset_index(drop=True)
    )

def _payoff_alignment_summary_frame(payoff_alignment: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "top_10_label_lift_vs_base",
        "top_10_mean_forward_return",
        "top_10_mean_tb_return",
        "top_10_tp_rate",
        "top_10_sl_rate",
        "top_10_payoff_alignment_pass",
        "top_10_payoff_blockers",
        "best_forward_return_band",
        "best_forward_return",
        "best_forward_return_label_lift",
        "best_lift_band",
        "best_lift",
        "best_lift_forward_return",
        "payoff_aligned_band_count",
        "payoff_mismatch_band_count",
        "next_action",
    ]
    if payoff_alignment.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (candidate, candidate_type, evaluation_scope), part in payoff_alignment.groupby(
        ["candidate", "candidate_type", "evaluation_scope"],
        dropna=False,
    ):
        part = part.copy()
        top_10 = part.loc[part["band"].astype(str).eq("top_10")]
        top = top_10.iloc[0].to_dict() if not top_10.empty else {}
        best_return = part.sort_values("mean_forward_return", ascending=False).iloc[0].to_dict()
        best_lift = part.sort_values("label_lift_vs_base", ascending=False).iloc[0].to_dict()
        mismatch_count = int(part["label_lift_positive_payoff_mismatch"].astype(bool).sum())
        aligned_count = int(part["payoff_alignment_pass"].astype(bool).sum())
        if top and bool(top.get("payoff_alignment_pass", False)):
            action = "top_10_payoff_aligned_monitor_future_oos"
        elif top and bool(top.get("label_lift_positive_payoff_mismatch", False)):
            action = "top_10_label_lift_payoff_mismatch_investigate"
        elif aligned_count > 0:
            action = "review_non_top10_payoff_aligned_band_before_future_oos"
        else:
            action = "no_payoff_aligned_band_do_not_promote"
        rows.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "evaluation_scope": str(evaluation_scope),
                "top_10_label_lift_vs_base": _float(top, "label_lift_vs_base") if top else np.nan,
                "top_10_mean_forward_return": _float(top, "mean_forward_return") if top else np.nan,
                "top_10_mean_tb_return": _float(top, "mean_tb_return") if top else np.nan,
                "top_10_tp_rate": _float(top, "tp_rate") if top else np.nan,
                "top_10_sl_rate": _float(top, "sl_rate") if top else np.nan,
                "top_10_payoff_alignment_pass": bool(top.get("payoff_alignment_pass", False)) if top else False,
                "top_10_payoff_blockers": str(top.get("payoff_blockers", "")) if top else "missing_top_10",
                "best_forward_return_band": str(best_return.get("band", "")),
                "best_forward_return": _float(best_return, "mean_forward_return"),
                "best_forward_return_label_lift": _float(best_return, "label_lift_vs_base"),
                "best_lift_band": str(best_lift.get("band", "")),
                "best_lift": _float(best_lift, "label_lift_vs_base"),
                "best_lift_forward_return": _float(best_lift, "mean_forward_return"),
                "payoff_aligned_band_count": aligned_count,
                "payoff_mismatch_band_count": mismatch_count,
                "next_action": action,
            }
        )
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "top_10_mean_forward_return"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

def _payoff_alignment_markdown(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    lines = ["# Payoff Alignment", ""]
    if summary.empty:
        lines.append("No payoff alignment rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "top_10_label_lift_vs_base",
        "top_10_mean_forward_return",
        "top_10_mean_tb_return",
        "top_10_tp_rate",
        "top_10_sl_rate",
        "top_10_payoff_alignment_pass",
        "top_10_payoff_blockers",
        "best_forward_return_band",
        "best_forward_return",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("## Summary")
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    mismatch = detail.loc[detail.get("label_lift_positive_payoff_mismatch", pd.Series(dtype=bool)).astype(bool)]
    if not mismatch.empty:
        lines.extend(["", "## Label Lift / Payoff Mismatches", ""])
        mismatch_cols = [
            "candidate",
            "evaluation_scope",
            "band",
            "label_lift_vs_base",
            "mean_forward_return",
            "mean_tb_return",
            "payoff_blockers",
        ]
        visible_mismatch = mismatch[[column for column in mismatch_cols if column in mismatch.columns]].copy()
        lines.append("| " + " | ".join(visible_mismatch.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible_mismatch.columns)) + " |")
        for _, row in visible_mismatch.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible_mismatch.columns) + " |")
    return "\n".join(lines)

def _write_payoff_alignment(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "payoff_alignment.csv", index=False)
    summary.to_csv(path / "payoff_alignment_summary.csv", index=False)
    (path / "payoff_alignment.md").write_text(_payoff_alignment_markdown(summary, detail), encoding="utf-8")
    _write_json(
        path / "payoff_alignment.json",
        {
            "summary": summary.to_dict(orient="records"),
            "detail": detail.to_dict(orient="records"),
        },
    )

def _payoff_policy_rows_for_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    candidate: str,
    candidate_type: str,
    evaluation_scope: str,
    fold_scope: str,
    fold: int,
) -> list[dict[str, Any]]:
    scored = _assign_payoff_score_bins(frame, score_column="prob_long", bins=int(_cfg(config, ["validation", "score_lift_bins"], 10)))
    if scored.empty:
        return []

    actual_bins = int(pd.to_numeric(scored["score_bin"], errors="coerce").max()) + 1
    bands = _resolve_payoff_score_bands(config, actual_bins)
    base_count = int(len(scored))
    base_long_rate = float(pd.to_numeric(scored["label"], errors="coerce").mean())
    base_forward_return = _numeric_mean(scored, "forward_return")
    base_tb_return = _numeric_mean(scored, "tb_return")
    base_tp_rate = _hit_rate(scored, "tp")
    base_sl_rate = _hit_rate(scored, "sl") + _hit_rate(scored, "both_sl_first")
    base_time_rate = _hit_rate(scored, "time")
    rows = []
    for band in bands:
        part = scored.loc[
            (pd.to_numeric(scored["score_bin"], errors="coerce") >= int(band["min_bin"]))
            & (pd.to_numeric(scored["score_bin"], errors="coerce") <= int(band["max_bin"]))
        ].copy()
        if part.empty:
            continue
        selected_count = int(len(part))
        selected_long_rate = float(pd.to_numeric(part["label"], errors="coerce").mean())
        mean_forward_return = _numeric_mean(part, "forward_return")
        mean_tb_return = _numeric_mean(part, "tb_return")
        tp_rate = _hit_rate(part, "tp")
        sl_rate = _hit_rate(part, "sl") + _hit_rate(part, "both_sl_first")
        time_rate = _hit_rate(part, "time")
        row = {
            "candidate": candidate,
            "candidate_type": candidate_type,
            "evaluation_scope": evaluation_scope,
            "fold_scope": fold_scope,
            "fold": int(fold),
            "band": str(band["name"]),
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "base_count": base_count,
            "selected_count": selected_count,
            "selection_rate": float(selected_count / base_count) if base_count else np.nan,
            "base_long_rate": base_long_rate,
            "selected_long_rate": selected_long_rate,
            "label_lift_vs_base": float(selected_long_rate / base_long_rate) if base_long_rate > 0 else np.nan,
            "base_forward_return": base_forward_return,
            "mean_forward_return": mean_forward_return,
            "forward_return_delta_vs_base": mean_forward_return - base_forward_return,
            "base_tb_return": base_tb_return,
            "mean_tb_return": mean_tb_return,
            "tb_return_delta_vs_base": mean_tb_return - base_tb_return,
            "base_tp_rate": base_tp_rate,
            "tp_rate": tp_rate,
            "tp_rate_delta_vs_base": tp_rate - base_tp_rate,
            "base_sl_rate": base_sl_rate,
            "sl_rate": sl_rate,
            "sl_rate_delta_vs_base": sl_rate - base_sl_rate,
            "base_time_rate": base_time_rate,
            "time_rate": time_rate,
            "time_rate_delta_vs_base": time_rate - base_time_rate,
        }
        row["payoff_blockers"] = _payoff_alignment_blockers(row)
        row["payoff_alignment_pass"] = not bool(row["payoff_blockers"])
        rows.append(row)
    return rows

def _payoff_policy_robustness_frame(
    entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "fold_scope",
        "fold",
        "band",
        "min_bin",
        "max_bin",
        "base_count",
        "selected_count",
        "selection_rate",
        "base_long_rate",
        "selected_long_rate",
        "label_lift_vs_base",
        "base_forward_return",
        "mean_forward_return",
        "forward_return_delta_vs_base",
        "base_tb_return",
        "mean_tb_return",
        "tb_return_delta_vs_base",
        "base_tp_rate",
        "tp_rate",
        "tp_rate_delta_vs_base",
        "base_sl_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
        "base_time_rate",
        "time_rate",
        "time_rate_delta_vs_base",
        "payoff_alignment_pass",
        "payoff_blockers",
    ]
    rows = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope != "full" and not fold_scope.startswith("blend_"):
            continue
        predictions = entry.get("predictions", pd.DataFrame())
        if not isinstance(predictions, pd.DataFrame) or predictions.empty or "fold" not in predictions.columns:
            continue
        test_predictions = _test_predictions(predictions)
        candidate = str(entry.get("profile", ""))
        candidate_type = "blend" if fold_scope.startswith("blend_") or candidate.startswith("blend_") else "profile"
        for fold, part in test_predictions.groupby("fold", dropna=False):
            rows.extend(
                _payoff_policy_rows_for_frame(
                    part,
                    config,
                    candidate=candidate,
                    candidate_type=candidate_type,
                    evaluation_scope="cv_test",
                    fold_scope=fold_scope,
                    fold=int(fold),
                )
            )
    for entry in holdout_entries:
        predictions = entry.get("predictions", pd.DataFrame())
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = "blend" if fold_scope.startswith("blend_") or candidate.startswith("blend_") else "profile"
        rows.extend(
            _payoff_policy_rows_for_frame(
                _test_predictions(predictions),
                config,
                candidate=candidate,
                candidate_type=candidate_type,
                evaluation_scope="holdout",
                fold_scope=fold_scope,
                fold=0,
            )
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "candidate", "band", "fold"])
        .reset_index(drop=True)
    )

def _payoff_policy_reject_reasons(row: dict[str, Any], config: dict[str, Any]) -> str:
    gates = _cfg(config, ["validation", "payoff_policy_robustness"], {}) or {}
    reasons = []
    if _float(row, "mean_label_lift_vs_base") < float(gates.get("min_mean_label_lift_vs_base", 1.05)):
        reasons.append("mean_label_lift_vs_base")
    if _float(row, "positive_label_lift_fold_rate") < float(gates.get("min_positive_label_lift_fold_rate", 0.60)):
        reasons.append("positive_label_lift_fold_rate")
    if _float(row, "mean_forward_return") <= float(gates.get("min_mean_forward_return", 0.0)):
        reasons.append("mean_forward_return")
    if _float(row, "positive_forward_return_fold_rate") < float(gates.get("min_positive_forward_return_fold_rate", 0.60)):
        reasons.append("positive_forward_return_fold_rate")
    if _float(row, "mean_tb_return") <= float(gates.get("min_mean_tb_return", 0.0)):
        reasons.append("mean_tb_return")
    if _float(row, "positive_tb_return_fold_rate") < float(gates.get("min_positive_tb_return_fold_rate", 0.55)):
        reasons.append("positive_tb_return_fold_rate")
    if _float(row, "payoff_alignment_fold_rate") < float(gates.get("min_payoff_alignment_fold_rate", 0.50)):
        reasons.append("payoff_alignment_fold_rate")
    if _float(row, "sl_rate_above_base_fold_rate") > float(gates.get("max_sl_rate_above_base_fold_rate", 0.70)):
        reasons.append("sl_rate_above_base_fold_rate")
    return ";".join(reasons)

def _payoff_policy_robustness_summary_frame(robustness: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "band",
        "folds",
        "mean_selection_rate",
        "mean_label_lift_vs_base",
        "positive_label_lift_fold_rate",
        "mean_forward_return",
        "positive_forward_return_fold_rate",
        "mean_tb_return",
        "positive_tb_return_fold_rate",
        "mean_tp_rate",
        "mean_sl_rate",
        "sl_rate_above_base_fold_rate",
        "payoff_alignment_fold_rate",
        "future_oos_policy_candidate",
        "reject_reason",
        "next_action",
    ]
    if robustness.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    frame = robustness.copy()
    numeric_columns = [
        "selection_rate",
        "label_lift_vs_base",
        "mean_forward_return",
        "mean_tb_return",
        "tp_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for (candidate, candidate_type, evaluation_scope, band), part in frame.groupby(
        ["candidate", "candidate_type", "evaluation_scope", "band"],
        dropna=False,
    ):
        row = {
            "candidate": str(candidate),
            "candidate_type": str(candidate_type),
            "evaluation_scope": str(evaluation_scope),
            "band": str(band),
            "folds": int(part["fold"].nunique()),
            "mean_selection_rate": float(part["selection_rate"].mean()),
            "mean_label_lift_vs_base": float(part["label_lift_vs_base"].mean()),
            "positive_label_lift_fold_rate": float((part["label_lift_vs_base"] > 1.0).mean()),
            "mean_forward_return": float(part["mean_forward_return"].mean()),
            "positive_forward_return_fold_rate": float((part["mean_forward_return"] > 0.0).mean()),
            "mean_tb_return": float(part["mean_tb_return"].mean()),
            "positive_tb_return_fold_rate": float((part["mean_tb_return"] > 0.0).mean()),
            "mean_tp_rate": float(part["tp_rate"].mean()),
            "mean_sl_rate": float(part["sl_rate"].mean()),
            "sl_rate_above_base_fold_rate": float((part["sl_rate_delta_vs_base"] > 0.0).mean()),
            "payoff_alignment_fold_rate": float(part["payoff_alignment_pass"].astype(bool).mean()),
        }
        reject_reason = _payoff_policy_reject_reasons(row, config)
        row["future_oos_policy_candidate"] = bool(str(evaluation_scope) == "cv_test" and not reject_reason)
        row["reject_reason"] = reject_reason
        if str(evaluation_scope) == "holdout":
            row["next_action"] = "diagnostic_only_do_not_select_from_current_holdout"
        elif row["future_oos_policy_candidate"]:
            row["next_action"] = "pre_register_for_future_oos_review"
        elif "mean_forward_return" in reject_reason or "mean_tb_return" in reject_reason:
            row["next_action"] = "payoff_not_robust_do_not_pre_register"
        else:
            row["next_action"] = "monitor_not_pre_registered"
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(
            ["evaluation_scope", "future_oos_policy_candidate", "mean_forward_return", "mean_label_lift_vs_base"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

def _payoff_policy_robustness_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Payoff Policy Robustness", ""]
    if summary.empty:
        lines.append("No score-band payoff policy robustness rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "evaluation_scope",
        "band",
        "folds",
        "mean_label_lift_vs_base",
        "positive_label_lift_fold_rate",
        "mean_forward_return",
        "positive_forward_return_fold_rate",
        "mean_tb_return",
        "positive_tb_return_fold_rate",
        "mean_sl_rate",
        "sl_rate_above_base_fold_rate",
        "payoff_alignment_fold_rate",
        "future_oos_policy_candidate",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_payoff_policy_robustness(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "payoff_policy_robustness.csv", index=False)
    summary.to_csv(path / "payoff_policy_robustness_summary.csv", index=False)
    (path / "payoff_policy_robustness.md").write_text(
        _payoff_policy_robustness_markdown(summary),
        encoding="utf-8",
    )
    _write_json(
        path / "payoff_policy_robustness.json",
        {
            "summary": summary.to_dict(orient="records"),
            "detail": detail.to_dict(orient="records"),
        },
    )

def _frozen_policy_robustness_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    robustness = policy_review.get("robustness", {}) or {}
    columns = [
        "window",
        "start",
        "end",
        "candidate",
        "policy_type",
        "policy_name",
        "rows",
        "selected_rows",
        "base_long_rate",
        "policy_precision",
        "policy_recall",
        "policy_f1",
        "policy_lift_vs_base",
        "policy_forward_return",
        "rank_ic",
        "window_pass",
        "reject_reason",
    ]
    if not bool(policy_review.get("enabled", False)) or not bool(robustness.get("enabled", False)):
        return pd.DataFrame(columns=columns)

    frozen_candidate = str(policy_review.get("frozen_candidate", ""))
    policy_type = str(policy_review.get("policy_type", ""))
    policy_name = str(policy_review.get("policy_name", ""))
    entry = next((item for item in entries if str(item.get("profile", "")) == frozen_candidate), None)
    if entry is None:
        return pd.DataFrame(
            [
                {
                    "window": "all",
                    "start": "",
                    "end": "",
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": 0,
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "missing_frozen_candidate_predictions",
                }
            ],
            columns=columns,
        )

    predictions = _test_predictions(entry.get("predictions", pd.DataFrame())).copy()
    if predictions.empty or "timestamp" not in predictions.columns:
        return pd.DataFrame(
            [
                {
                    "window": "all",
                    "start": "",
                    "end": "",
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": int(len(predictions)),
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "missing_frozen_candidate_timestamps",
                }
            ],
            columns=columns,
        )

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    windows = robustness.get("windows", []) or []
    if not windows:
        windows = [
            {
                "name": "all_available_cv",
                "start": str(predictions["timestamp"].min()),
                "end": str(predictions["timestamp"].max()),
            }
        ]

    min_rows = int(robustness.get("min_rows", 0) or 0)
    min_selected_rows = int(robustness.get("min_selected_rows", 0) or 0)
    min_rank_ic = float(robustness.get("min_rank_ic", 0.0))
    min_lift = float(robustness.get("min_lift_vs_base", 1.0))
    min_forward_return = float(robustness.get("min_forward_return", 0.0))
    policy = {"policy_type": policy_type, "policy_name": policy_name}
    rows = []
    for spec in windows:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "window"))
        start_raw = str(spec.get("start", ""))
        end_raw = str(spec.get("end", ""))
        try:
            start = pd.to_datetime(start_raw, utc=True)
            end = pd.to_datetime(end_raw, utc=True)
        except (TypeError, ValueError):
            rows.append(
                {
                    "window": name,
                    "start": start_raw,
                    "end": end_raw,
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": 0,
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "invalid_window_bounds",
                }
            )
            continue

        part = predictions.loc[(predictions["timestamp"] >= start) & (predictions["timestamp"] <= end)].copy()
        metrics = _evaluate_score_policy_on_holdout(part, policy, config) if not part.empty else {}
        rows_count = int(len(part))
        selection_rate = _float(metrics, "selection_rate", 0.0)
        selected_rows = int(round(rows_count * selection_rate))
        rank_ic = _rank_ic_for_frame(part)
        base_long_rate = float(part["label"].mean()) if rows_count and "label" in part.columns else np.nan
        reasons = []
        if rows_count < min_rows:
            reasons.append("rows")
        if selected_rows < min_selected_rows:
            reasons.append("selected_rows")
        if not bool(metrics.get("pass", False)):
            reasons.append(str(metrics.get("reject_reason", "policy")).strip(";") or "policy")
        if not np.isfinite(rank_ic) or rank_ic <= min_rank_ic:
            reasons.append("rank_ic")
        lift = _float(metrics, "lift_vs_base")
        forward_return = _float(metrics, "forward_return")
        if not np.isfinite(lift) or lift <= min_lift:
            reasons.append("lift_vs_base")
        if not np.isfinite(forward_return) or forward_return <= min_forward_return:
            reasons.append("forward_return")
        rows.append(
            {
                "window": name,
                "start": str(start),
                "end": str(end),
                "candidate": frozen_candidate,
                "policy_type": policy_type,
                "policy_name": policy_name,
                "rows": rows_count,
                "selected_rows": selected_rows,
                "base_long_rate": base_long_rate,
                "policy_precision": _float(metrics, "precision"),
                "policy_recall": _float(metrics, "recall"),
                "policy_f1": _float(metrics, "f1"),
                "policy_lift_vs_base": lift,
                "policy_forward_return": forward_return,
                "rank_ic": rank_ic,
                "window_pass": len(reasons) == 0,
                "reject_reason": ";".join(reason for reason in reasons if reason),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _write_frozen_policy_robustness(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "frozen_policy_robustness.csv", index=False)
    (path / "frozen_policy_robustness.md").write_text(
        _table_markdown("Frozen Policy Robustness", frame),
        encoding="utf-8",
    )
    _write_json(path / "frozen_policy_robustness.json", {"rows": frame.to_dict(orient="records")})
