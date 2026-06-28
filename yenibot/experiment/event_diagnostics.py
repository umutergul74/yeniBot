"""Event, label, and sample-information diagnostics for Phase 1 research.

These reports are intentionally diagnostic-only. They do not select profiles,
change thresholds, or mutate future-OOS state. Their purpose is to answer the
question that repeated profile tweaks cannot answer: whether the current target
and training rows contain enough event-level information for a durable decision
score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.diagnostics.metrics import rank_ic
from yenibot.experiment.common import _cfg, _diagnostic_candidate_type, _float, _json_ready, _write_json
from yenibot.experiment.training import _test_predictions

__all__ = [
    "_event_diagnostic_frames",
    "_write_event_diagnostic_reports",
]


_EVENT_FEATURE_PATTERNS = {
    "order_flow": (
        "taker_imbalance",
        "taker_buy_ratio",
        "true_cvd",
        "cvd_cumulative",
        "buy_sell_imbalance",
    ),
    "whale_ticket_size": (
        "large_trade",
        "vol_per_trade",
        "vpt_zscore",
        "whale_buy",
        "whale_sell",
    ),
    "volume_context": (
        "volume_log_zscore",
        "volume_denoised_log_zscore",
    ),
    "volatility_structure": (
        "realized_vol",
        "gk_vol",
        "atr_14_pct",
        "adx_14",
        "vwap_dist",
    ),
    "intrahour_flow": (
        "ih15_",
    ),
    "futures_context": (
        "fut_",
        "funding",
        "open_interest",
    ),
}


def _forward_column(frame: pd.DataFrame) -> str:
    if "forward_return" in frame.columns:
        return "forward_return"
    for column in frame.columns:
        if str(column).startswith("fwd_return_"):
            return str(column)
    return ""


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _candidate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope != "full":
            continue
        profile = str(entry.get("profile", ""))
        if profile.startswith("blend_") or fold_scope.startswith("seed_ensemble_"):
            continue
        predictions = entry.get("predictions")
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            selected.append(entry)
    if selected:
        return selected
    fallback: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        predictions = entry.get("predictions")
        if (
            isinstance(predictions, pd.DataFrame)
            and not predictions.empty
            and (fold_scope == "full" or fold_scope.startswith("blend_"))
        ):
            fallback.append(entry)
    return fallback


def _family_columns(feature_columns: list[str], frame: pd.DataFrame) -> dict[str, list[str]]:
    available = [column for column in feature_columns if column in frame.columns]
    families: dict[str, list[str]] = {}
    for family, patterns in _EVENT_FEATURE_PATTERNS.items():
        matches = [
            column
            for column in available
            if any(pattern in str(column) for pattern in patterns)
        ]
        families[family] = matches
    return families


def _row_event_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = frame.copy()
    families = _family_columns(feature_columns, out)
    score_parts: list[pd.Series] = []
    for family, columns in families.items():
        if not columns:
            out[f"event_{family}_score"] = np.nan
            continue
        values = out[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        family_score = values.abs().mean(axis=1)
        out[f"event_{family}_score"] = family_score
        score_parts.append(family_score)
    if score_parts:
        combined = pd.concat(score_parts, axis=1).mean(axis=1)
        out["event_score"] = combined
    else:
        out["event_score"] = np.nan
    if out["event_score"].notna().any():
        ranks = out.groupby("fold")["event_score"].rank(method="average", pct=True)
        out["event_rank_pct"] = ranks
    else:
        out["event_rank_pct"] = np.nan
    return out, families


def _band_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    rank = _numeric_series(frame, "event_rank_pct")
    return {
        "all": pd.Series(True, index=frame.index),
        "event_top_10": rank >= 0.90,
        "event_top_20": rank >= 0.80,
        "event_mid_40_80": (rank >= 0.40) & (rank < 0.80),
        "event_bottom_40": rank < 0.40,
    }


def _event_summary_row(
    frame: pd.DataFrame,
    *,
    candidate: str,
    candidate_type: str,
    fold_scope: str,
    band: str,
    mask: pd.Series,
) -> dict[str, Any]:
    part = frame.loc[mask].copy()
    forward_col = _forward_column(part)
    label = _numeric_series(part, "label")
    score = _numeric_series(part, "prob_long")
    forward = _numeric_series(part, forward_col) if forward_col else pd.Series(np.nan, index=part.index)
    score_rank = (
        part.groupby("fold")["prob_long"].rank(method="average", pct=True)
        if not part.empty and {"fold", "prob_long"}.issubset(part.columns)
        else pd.Series(np.nan, index=part.index)
    )
    top_score = score_rank >= 0.90
    return {
        "candidate": candidate,
        "candidate_type": candidate_type,
        "fold_scope": fold_scope,
        "event_band": band,
        "row_count": int(len(part)),
        "fold_count": int(part["fold"].nunique()) if "fold" in part.columns and not part.empty else 0,
        "row_fraction": float(len(part) / len(frame)) if len(frame) else np.nan,
        "label_rate": float(label.mean()) if label.notna().any() else np.nan,
        "mean_forward_return": float(forward.mean()) if forward.notna().any() else np.nan,
        "median_forward_return": float(forward.median()) if forward.notna().any() else np.nan,
        "mean_tb_return": float(_numeric_series(part, "tb_return").mean()) if "tb_return" in part.columns else np.nan,
        "mean_prob_long": float(score.mean()) if score.notna().any() else np.nan,
        "rank_ic": rank_ic(score, forward) if len(part) >= 3 and forward.notna().any() else np.nan,
        "top_score_label_rate": float(label.loc[top_score].mean()) if top_score.any() else np.nan,
        "top_score_forward_return": float(forward.loc[top_score].mean()) if top_score.any() else np.nan,
        "event_score_mean": float(_numeric_series(part, "event_score").mean())
        if "event_score" in part.columns and len(part)
        else np.nan,
        "diagnostic_interpretation": "",
    }


def _label_event_audit_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "event_band",
        "row_count",
        "fold_count",
        "row_fraction",
        "label_rate",
        "mean_forward_return",
        "median_forward_return",
        "mean_tb_return",
        "mean_prob_long",
        "rank_ic",
        "top_score_label_rate",
        "top_score_forward_return",
        "event_score_mean",
        "diagnostic_interpretation",
    ]
    rows: list[dict[str, Any]] = []
    for entry in _candidate_entries(entries):
        predictions = _test_predictions(entry["predictions"]).copy()
        if predictions.empty or "fold" not in predictions.columns:
            continue
        event_frame, _ = _row_event_frame(
            predictions,
            feature_columns=list(entry.get("feature_columns", [])),
        )
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        all_reference: dict[str, Any] = {}
        for band, mask in _band_masks(event_frame).items():
            row = _event_summary_row(
                event_frame,
                candidate=candidate,
                candidate_type=candidate_type,
                fold_scope=fold_scope,
                band=band,
                mask=mask.fillna(False),
            )
            if band != "all":
                base_return = _float(all_reference, "mean_forward_return")
                base_ic = _float(all_reference, "rank_ic")
                row["diagnostic_interpretation"] = (
                    "event_subset_has_better_return_or_rank_ic"
                    if (
                        np.isfinite(row["mean_forward_return"])
                        and np.isfinite(base_return)
                        and row["mean_forward_return"] > base_return
                    )
                    or (
                        np.isfinite(row["rank_ic"])
                        and np.isfinite(base_ic)
                        and row["rank_ic"] > base_ic
                    )
                    else "event_subset_not_better_than_all_rows"
                )
            else:
                row["diagnostic_interpretation"] = "all_rows_reference"
                all_reference = dict(row)
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _event_weighting_family_closed(config: dict[str, Any]) -> bool:
    completed = set(
        str(item)
        for item in list(
            _cfg(config, ["experiments", "research_focus", "completed_hypotheses"], [])
            or []
        )
    )
    return "effective_orderflow_event_weighting_v2" in completed


def _event_conditioned_performance_frame(
    label_event_audit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "best_event_band_by_forward_return",
        "best_event_band_forward_return",
        "all_rows_forward_return",
        "forward_return_delta",
        "best_event_band_by_rank_ic",
        "best_event_band_rank_ic",
        "all_rows_rank_ic",
        "rank_ic_delta",
        "event_conditioning_signal",
        "recommended_next_hypothesis",
    ]
    if label_event_audit.empty:
        return pd.DataFrame(columns=columns)
    weighting_closed = _event_weighting_family_closed(config)
    rows: list[dict[str, Any]] = []
    for keys, group in label_event_audit.groupby(["candidate", "candidate_type", "fold_scope"], dropna=False):
        candidate, candidate_type, fold_scope = [str(item) for item in keys]
        all_row = group.loc[group["event_band"].astype(str) == "all"]
        if all_row.empty:
            continue
        all_dict = all_row.iloc[0].to_dict()
        subsets = group.loc[group["event_band"].astype(str) != "all"].copy()
        if subsets.empty:
            continue
        subsets["mean_forward_return"] = pd.to_numeric(subsets["mean_forward_return"], errors="coerce")
        subsets["rank_ic"] = pd.to_numeric(subsets["rank_ic"], errors="coerce")
        best_return = subsets.sort_values("mean_forward_return", ascending=False).iloc[0].to_dict()
        best_ic = subsets.sort_values("rank_ic", ascending=False).iloc[0].to_dict()
        return_delta = _float(best_return, "mean_forward_return") - _float(all_dict, "mean_forward_return")
        ic_delta = _float(best_ic, "rank_ic") - _float(all_dict, "rank_ic")
        has_signal = bool((np.isfinite(return_delta) and return_delta > 0.0) or (np.isfinite(ic_delta) and ic_delta > 0.02))
        if has_signal:
            recommendation = (
                "historical_signal_only_event_weighting_family_closed"
                if weighting_closed
                else "design_event_weighted_training_v1"
            )
        else:
            recommendation = "do_not_add_event_weighting_without_stronger_evidence"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "best_event_band_by_forward_return": str(best_return.get("event_band", "")),
                "best_event_band_forward_return": _float(best_return, "mean_forward_return"),
                "all_rows_forward_return": _float(all_dict, "mean_forward_return"),
                "forward_return_delta": return_delta,
                "best_event_band_by_rank_ic": str(best_ic.get("event_band", "")),
                "best_event_band_rank_ic": _float(best_ic, "rank_ic"),
                "all_rows_rank_ic": _float(all_dict, "rank_ic"),
                "rank_ic_delta": ic_delta,
                "event_conditioning_signal": "present" if has_signal else "weak_or_absent",
                "recommended_next_hypothesis": recommendation,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _sample_information_audit_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    schema = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "family",
        "feature_count",
        "available_feature_count",
        "row_count",
        "event_score_mean",
        "event_score_p90",
        "event_top_20_row_fraction",
        "event_top_20_label_rate",
        "event_top_20_forward_return",
        "information_signal",
        "recommended_action",
    ]
    weighting_closed = _event_weighting_family_closed(config)
    rows: list[dict[str, Any]] = []
    for entry in _candidate_entries(entries):
        predictions = _test_predictions(entry["predictions"]).copy()
        if predictions.empty:
            continue
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        event_frame, families = _row_event_frame(
            predictions,
            feature_columns=list(entry.get("feature_columns", [])),
        )
        forward_col = _forward_column(event_frame)
        for family, family_columns in families.items():
            score_col = f"event_{family}_score"
            score = _numeric_series(event_frame, score_col)
            if not score.notna().any():
                continue
            rank = score.groupby(event_frame["fold"]).rank(method="average", pct=True) if "fold" in event_frame else score.rank(pct=True)
            mask = rank >= 0.80
            label = _numeric_series(event_frame.loc[mask], "label")
            forward = (
                _numeric_series(event_frame.loc[mask], forward_col)
                if forward_col
                else pd.Series(np.nan, index=event_frame.loc[mask].index)
            )
            all_forward = _numeric_series(event_frame, forward_col) if forward_col else pd.Series(np.nan, index=event_frame.index)
            top_return = float(forward.mean()) if forward.notna().any() else np.nan
            all_return = float(all_forward.mean()) if all_forward.notna().any() else np.nan
            signal = "positive" if np.isfinite(top_return) and np.isfinite(all_return) and top_return > all_return else "monitor"
            if signal == "positive":
                recommendation = (
                    "historical_signal_only_event_weighting_family_closed"
                    if weighting_closed
                    else "candidate_for_sample_weighting_or_event_filter_diagnostic"
                )
            else:
                recommendation = "monitor_do_not_weight_without_payoff_evidence"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "family": family,
                    "feature_count": int(len([c for c in entry.get("feature_columns", []) if any(p in str(c) for p in _EVENT_FEATURE_PATTERNS[family])])),
                    "available_feature_count": int(len(family_columns)),
                    "row_count": int(len(event_frame)),
                    "event_score_mean": float(score.mean()),
                    "event_score_p90": float(score.quantile(0.90)),
                    "event_top_20_row_fraction": float(mask.mean()) if len(mask) else np.nan,
                    "event_top_20_label_rate": float(label.mean()) if label.notna().any() else np.nan,
                    "event_top_20_forward_return": top_return,
                    "information_signal": signal,
                    "recommended_action": recommendation,
                }
            )
    return pd.DataFrame(rows, columns=schema)


def _overlap_uniqueness_audit_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "row_count",
        "label_horizon_bars",
        "mean_concurrency_at_label_start",
        "mean_uniqueness_weight",
        "overlap_information_fraction_proxy",
        "effective_sample_size",
        "effective_sample_fraction",
        "static_weight_cv",
        "static_weight_p90_p10_spread",
        "static_weight_dominant_fraction",
        "static_uniqueness_weighting_noop",
        "top_event_effective_sample_fraction",
        "overlap_risk",
        "recommended_next_hypothesis",
    ]
    horizon = int(_cfg(config, ["labeling", "max_holding_bars"], 10))
    weighting_closed = _event_weighting_family_closed(config)
    rows: list[dict[str, Any]] = []
    for entry in _candidate_entries(entries):
        predictions = _test_predictions(entry["predictions"]).copy()
        if predictions.empty or "fold" not in predictions.columns:
            continue
        event_frame, _ = _row_event_frame(predictions, feature_columns=list(entry.get("feature_columns", [])))
        uniqueness_values: list[float] = []
        concurrency_values: list[float] = []
        top_event_uniqueness: list[float] = []
        for _, fold_part in event_frame.groupby("fold"):
            ordered = fold_part.sort_values("timestamp" if "timestamp" in fold_part.columns else "source_row_position").copy()
            n = len(ordered)
            if n == 0:
                continue
            positions = np.arange(n)
            starts = positions
            ends = np.minimum(positions + horizon, n - 1)
            concurrency = np.zeros(n, dtype=float)
            for idx in range(n):
                concurrency[idx] = float(((starts <= idx) & (ends >= idx)).sum())
            inverse_concurrency = 1.0 / np.maximum(concurrency, 1.0)
            prefix = np.concatenate([[0.0], np.cumsum(inverse_concurrency)])
            label_uniqueness = np.asarray(
                [
                    (prefix[end + 1] - prefix[start]) / max(1, end - start + 1)
                    for start, end in zip(starts, ends)
                ],
                dtype=float,
            )
            uniqueness_values.extend(label_uniqueness.tolist())
            concurrency_values.extend(concurrency.tolist())
            event_rank = _numeric_series(ordered, "event_rank_pct")
            top_mask = event_rank >= 0.80
            if top_mask.any():
                top_event_uniqueness.extend(label_uniqueness[top_mask.to_numpy()].tolist())
        row_count = int(len(event_frame))
        uniqueness = np.asarray(uniqueness_values, dtype=float)
        uniqueness = uniqueness[np.isfinite(uniqueness) & (uniqueness > 0)]
        overlap_proxy = float(np.nanmean(uniqueness)) if uniqueness.size else np.nan
        if uniqueness.size:
            normalized = uniqueness / float(np.mean(uniqueness))
            eff_n = float(normalized.sum() ** 2 / (np.square(normalized).sum() + 1e-12))
            eff_frac = float(eff_n / normalized.size)
            static_cv = float(normalized.std(ddof=0) / max(normalized.mean(), 1e-12))
            static_spread = float(
                np.quantile(normalized, 0.90) - np.quantile(normalized, 0.10)
            )
            _, counts = np.unique(np.round(normalized, 6), return_counts=True)
            static_dominant = float(counts.max() / normalized.size)
        else:
            eff_n = np.nan
            eff_frac = np.nan
            static_cv = np.nan
            static_spread = np.nan
            static_dominant = np.nan
        static_noop = bool(
            np.isfinite(eff_frac)
            and np.isfinite(static_spread)
            and (eff_frac >= 0.98 or static_spread < 0.02)
        )
        top_eff_frac = float(np.nanmean(top_event_uniqueness)) if top_event_uniqueness else np.nan
        risk = (
            "high_overlap"
            if np.isfinite(overlap_proxy) and overlap_proxy < 0.25
            else "moderate_or_low_overlap"
        )
        if weighting_closed:
            recommendation = "static_and_event_weighting_families_closed"
        elif risk == "high_overlap" and static_noop:
            recommendation = (
                "do_not_use_static_uniqueness_weights_"
                "test_effective_event_weighting_or_sampling"
            )
        elif risk == "high_overlap":
            recommendation = "design_overlap_aware_sampling_not_static_weighting"
        else:
            recommendation = "monitor_overlap_before_changing_training"
        rows.append(
            {
                "candidate": str(entry.get("profile", "")),
                "candidate_type": _diagnostic_candidate_type(str(entry.get("fold_scope", ""))),
                "fold_scope": str(entry.get("fold_scope", "")),
                "fold_count": int(event_frame["fold"].nunique()) if "fold" in event_frame else 0,
                "row_count": row_count,
                "label_horizon_bars": horizon,
                "mean_concurrency_at_label_start": float(np.nanmean(concurrency_values)) if concurrency_values else np.nan,
                "mean_uniqueness_weight": overlap_proxy,
                "overlap_information_fraction_proxy": overlap_proxy,
                "effective_sample_size": eff_n,
                "effective_sample_fraction": eff_frac,
                "static_weight_cv": static_cv,
                "static_weight_p90_p10_spread": static_spread,
                "static_weight_dominant_fraction": static_dominant,
                "static_uniqueness_weighting_noop": static_noop,
                "top_event_effective_sample_fraction": top_eff_frac,
                "overlap_risk": risk,
                "recommended_next_hypothesis": recommendation,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _hypothesis_registry_summary_frame(config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "hypothesis_family",
        "status",
        "profile_count",
        "representative_profiles",
        "lesson",
        "allowed_next_step",
    ]
    memory = _cfg(config, ["experiments", "experiment_memory"], {}) or {}
    rejected = memory.get("rejected_profiles", {}) or {}
    reference = memory.get("reference_notes", {}) or {}
    buckets: dict[str, list[tuple[str, str]]] = {
        "seed_ensemble": [],
        "capacity": [],
        "loss_only": [],
        "direct_ablation_or_mask": [],
        "futures_context": [],
        "feature_profile_search": [],
        "other": [],
    }
    for profile, payload in rejected.items():
        text = f"{profile} {payload.get('reason', '') if isinstance(payload, dict) else payload}".lower()
        if "seed" in text and "ensemble" in text:
            key = "seed_ensemble"
        elif any(token in text for token in ("capacity", "tcn", "gru", "fusion", "encoder")):
            key = "capacity"
        elif any(token in text for token in ("loss", "pairwise", "margin")):
            key = "loss_only"
        elif any(token in text for token in ("ablation", "mask", "clipping", "clip", "delet", "remov")):
            key = "direct_ablation_or_mask"
        elif any(token in text for token in ("funding", "futures", "oi", "open interest", "positioning")):
            key = "futures_context"
        elif any(token in text for token in ("stable", "pressure", "whale", "flow", "interaction", "profile")):
            key = "feature_profile_search"
        else:
            key = "other"
        reason = payload.get("reason", "") if isinstance(payload, dict) else str(payload)
        buckets[key].append((str(profile), str(reason)))
    rows: list[dict[str, Any]] = []
    lessons = {
        "seed_ensemble": "Mean IC can rise while top-score payoff and F1 degrade; do not add weights/seeds without a new mechanism.",
        "capacity": "Uniform and component-only shrinkage did not preserve ranking stability, F1, and top-score payoff together.",
        "loss_only": "Score-separation loss tweaks did not fix bad-fold reversal without damaging core gates.",
        "direct_ablation_or_mask": "Direct deletion, hard masking, and clipping repeatedly failed to create a promotable replacement.",
        "futures_context": "Broad futures overlays added instability; only narrow pre-registered diagnostics should remain.",
        "feature_profile_search": "Broad profile search risks retesting renamed failures; require a mechanism and stop condition.",
        "other": "Archived failed ideas remain blocked unless explicitly allowed for retest.",
    }
    next_steps = {
        "seed_ensemble": "closed_no_weight_or_seed_search",
        "capacity": "closed_keep_baseline_tcn64_gru128_fusion128",
        "loss_only": "closed_until_multitask_or_event_weight_evidence_exists",
        "direct_ablation_or_mask": "closed_for_direct_retests_use_event_sample_diagnostics_instead",
        "futures_context": "monitor_only",
        "feature_profile_search": "require_label_event_or_sample_information_evidence",
        "other": "do_not_auto_retest",
    }
    for family, items in buckets.items():
        rows.append(
            {
                "hypothesis_family": family,
                "status": "rejected_or_closed" if items else "no_rejected_profiles_recorded",
                "profile_count": int(len(items)),
                "representative_profiles": ",".join(profile for profile, _ in items[:8]),
                "lesson": lessons[family],
                "allowed_next_step": next_steps[family],
            }
        )
    if reference:
        rows.append(
            {
                "hypothesis_family": "current_references",
                "status": "reference_only",
                "profile_count": int(len(reference)),
                "representative_profiles": ",".join(list(reference.keys())[:8]),
                "lesson": "Reference notes define the current benchmark, not an invitation to rerun old candidates.",
                "allowed_next_step": "compare_new_mechanisms_only_against_safe_control",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _event_diagnostic_frames(entries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    label_event = _label_event_audit_frame(entries)
    return {
        "label_event_audit": label_event,
        "event_conditioned_performance": _event_conditioned_performance_frame(
            label_event,
            config,
        ),
        "sample_information_audit": _sample_information_audit_frame(entries, config),
        "overlap_uniqueness_audit": _overlap_uniqueness_audit_frame(entries, config),
        "hypothesis_registry_summary": _hypothesis_registry_summary_frame(config),
    }


def _markdown_report(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No rows were produced.")
        return "\n".join(lines)
    visible = frame.copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_frame_bundle(path: Path, stem: str, title: str, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / f"{stem}.csv", index=False)
    (path / f"{stem}.md").write_text(_markdown_report(title, frame), encoding="utf-8")
    _write_json(path / f"{stem}.json", {"rows": _json_ready(frame.to_dict(orient="records"))})


def _write_event_diagnostic_reports(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    titles = {
        "label_event_audit": "Label And Event Audit",
        "event_conditioned_performance": "Event Conditioned Performance",
        "sample_information_audit": "Sample Information Audit",
        "overlap_uniqueness_audit": "Overlap And Uniqueness Audit",
        "hypothesis_registry_summary": "Hypothesis Registry Summary",
    }
    for stem, frame in frames.items():
        _write_frame_bundle(path, stem, titles.get(stem, stem.replace("_", " ").title()), frame)
