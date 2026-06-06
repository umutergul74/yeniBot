"""Frozen holdout reservation, future-OOS policy, and holdout diagnostics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
from yenibot.diagnostics import (
    score_band_diagnostics,
)
from yenibot.training.trainer import _add_regime_probs, _build_model, _device, _make_dataset, _predict_dataset

from yenibot.experiment.common import (
    _cfg,
    _float,
    _holdout_policy_action,
    _json_ready,
    _optional_float,
    _table_markdown,
    _write_json,
)

from yenibot.experiment.configuration import (
    _experiment_policy_guard,
    _future_oos_monitor_state,
    _future_oos_ready_at_fields,
    _holdout_latest_available_data_end,
    profile_config,
)

from yenibot.experiment.training import (
    _cv_score_policy,
    _cv_selected_threshold,
)

__all__ = [
    '_parquet_timestamps',
    '_default_holdout_path',
    '_resolve_holdout_settings',
    '_selection_frame_before_holdout',
    '_holdout_reservation_frame',
    '_holdout_reservation_markdown',
    '_write_holdout_reservation',
    'prepare_training_holdout_split',
    '_holdout_boundary_audit_frame',
    '_write_holdout_boundary_audit',
    '_read_holdout_context',
    '_load_torch_checkpoint',
    '_predict_holdout_for_profile',
    '_aggregate_holdout_predictions',
    '_holdout_markdown',
    '_write_holdout_files',
    '_holdout_policy_decision_frame',
    '_binary_metrics_at_threshold',
    '_attach_holdout_cv_threshold_metrics',
    '_policy_metrics_from_mask',
    '_evaluate_score_policy_on_holdout',
    '_attach_holdout_policy_metrics',
    '_attach_holdout_policy_consistency',
    '_holdout_signal_pass_reasons',
    '_holdout_threshold_pass_reasons',
    '_attach_holdout_soft_pass',
    '_frozen_policy_monitoring_plan_frame',
    '_write_frozen_policy_monitoring_plan',
    '_experiment_policy_guard_frame',
    '_write_experiment_policy_guard',
    '_recommendation_with_policy_guard',
    '_future_oos_candidate_plan_frame',
    '_write_future_oos_candidate_plan',
    '_performance_gap_reasons',
    '_holdout_gap_reasons',
    '_performance_gap_action',
    '_performance_gap_analysis_frame',
    '_performance_gap_markdown',
    '_write_performance_gap_analysis',
]

def _parquet_timestamps(path: Path) -> pd.Series:
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except (TypeError, ValueError):
        frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["timestamp"], utc=True).dropna()

def _default_holdout_path(config: dict[str, Any], holdout: dict[str, Any]) -> Path | None:
    explicit = str(holdout.get("holdout_path") or "").strip()
    if explicit:
        return Path(explicit)
    data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
    if not data_dir:
        return None
    filename = str(holdout.get("holdout_filename") or "holdout_1h.parquet")
    return Path(str(data_dir)) / "processed" / filename

def _resolve_holdout_settings(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Attach durable holdout metadata from config/default Drive paths.

    Notebook 04 injects holdout metadata into its in-memory config before training,
    but notebook 05 may be run in a fresh session. This resolver lets diagnostics
    recover the reserved holdout from config and the standard parquet location.
    """

    updated = copy.deepcopy(settings)
    holdout = copy.deepcopy(updated.get("holdout") or _cfg(config, ["experiments", "holdout"], {}) or {})
    if not holdout:
        return updated

    holdout.setdefault("enabled", True)
    holdout.setdefault("policy", "profile_selection_only_before_holdout; holdout is reserved for one-shot final validation")
    holdout_path = _default_holdout_path(config, holdout)
    if holdout_path is not None:
        holdout["holdout_path"] = str(holdout_path)
        if holdout_path.exists():
            timestamps = _parquet_timestamps(holdout_path)
            if not timestamps.empty:
                holdout.setdefault("holdout_rows", int(len(timestamps)))
                holdout.setdefault("holdout_bars", int(holdout.get("holdout_rows", len(timestamps))))
                holdout.setdefault("holdout_data_start", str(timestamps.min()))
                holdout.setdefault("holdout_data_end", str(timestamps.max()))

                data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
                labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet" if data_dir else None
                if labeled_path is not None and labeled_path.exists():
                    labeled_timestamps = _parquet_timestamps(labeled_path)
                    if not labeled_timestamps.empty:
                        holdout.setdefault("latest_available_data_end", str(labeled_timestamps.max()))
                        holdout_start = pd.to_datetime(holdout["holdout_data_start"], utc=True)
                        selection_timestamps = labeled_timestamps.loc[labeled_timestamps < holdout_start]
                        if not selection_timestamps.empty:
                            holdout.setdefault("selection_rows", int(len(selection_timestamps)))
                            holdout.setdefault("selection_data_start", str(selection_timestamps.min()))
                            holdout.setdefault("selection_data_end", str(selection_timestamps.max()))
    latest_data_end = _holdout_latest_available_data_end(holdout)
    if latest_data_end:
        monitor_state = _future_oos_monitor_state(config, latest_data_end)
        for key, value in monitor_state.items():
            holdout.setdefault(key, value)

    updated["holdout"] = holdout
    return updated

def _selection_frame_before_holdout(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)) or "timestamp" not in frame.columns:
        return frame
    holdout_start = holdout.get("holdout_data_start")
    if not holdout_start:
        return frame
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = pd.to_datetime(holdout_start, utc=True)
    if not (timestamps >= start).any():
        return frame
    return frame.loc[timestamps < start].copy().reset_index(drop=True)

def _holdout_reservation_frame(settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    columns = [
        "enabled",
        "holdout_bars",
        "selection_rows",
        "holdout_rows",
        "selection_data_start",
        "selection_data_end",
        "holdout_data_start",
        "holdout_data_end",
        "holdout_path",
        "policy",
        "split_mode",
        "unused_rows_after_anchor",
        "anchor_run_id",
        "anchor_data_end",
        "latest_available_data_end",
        "new_bars_since_anchor",
        "min_new_bars_remaining",
        "preferred_new_bars_remaining",
        "future_oos_ready",
        "future_oos_preferred_ready",
        "holdout_roll_forward_locked",
    ]
    if not holdout:
        return pd.DataFrame(columns=columns)
    row = {column: holdout.get(column, "") for column in columns}
    row["enabled"] = bool(holdout.get("enabled", False))
    return pd.DataFrame([row], columns=columns)

def _holdout_reservation_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Holdout Reservation", ""]
    if frame.empty:
        lines.append("No holdout reservation metadata was attached to this experiment run.")
        return "\n".join(lines)
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    row = frame.iloc[0].to_dict()
    for key, value in row.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)

def _write_holdout_reservation(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    frame = _holdout_reservation_frame(settings)
    frame.to_csv(path / "holdout_reservation.csv", index=False)
    (path / "holdout_reservation.md").write_text(_holdout_reservation_markdown(frame), encoding="utf-8")
    _write_json(path / "holdout_reservation.json", {"rows": frame.to_dict(orient="records")})
    return frame

def prepare_training_holdout_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    holdout_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create the training/holdout split without contaminating a failed clean holdout.

    After a clean holdout invalidates a frozen policy, the anchor holdout window
    must not silently roll forward just because fresher rows exist. Unless the
    config explicitly allows holdout roll-forward, rows after the anchor end stay
    unused for training and are counted only as future OOS monitoring bars.
    """

    if frame.empty or "timestamp" not in frame.columns:
        raise ValueError("Holdout split requires a non-empty frame with a timestamp column")

    data = frame.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    order = np.argsort(timestamps.to_numpy())
    data = data.iloc[order].reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)

    holdout_cfg = _cfg(config, ["experiments", "holdout"], {}) or {}
    holdout_bars = int(holdout_cfg.get("holdout_bars", 4320) or 4320)
    if len(data) <= holdout_bars:
        raise ValueError(f"Not enough rows for a {holdout_bars}-bar holdout: {len(data)} rows")

    latest_data_end = str(timestamps.max())
    monitor_state = _future_oos_monitor_state(config, latest_data_end)
    split_mode = "rolling_latest_holdout"
    unused_rows_after_anchor = 0
    split_data = data

    if monitor_state["holdout_roll_forward_locked"] and monitor_state["anchor_data_end"]:
        anchor_ts = pd.to_datetime(monitor_state["anchor_data_end"], utc=True)
        before_or_at_anchor = timestamps <= anchor_ts
        if before_or_at_anchor.any():
            split_data = data.loc[before_or_at_anchor].copy().reset_index(drop=True)
            unused_rows_after_anchor = int((timestamps > anchor_ts).sum())
            split_mode = "frozen_anchor_holdout"

    if len(split_data) <= holdout_bars:
        raise ValueError(
            f"Not enough rows for a {holdout_bars}-bar holdout after applying {split_mode}: {len(split_data)} rows"
        )

    holdout = split_data.tail(holdout_bars).copy().reset_index(drop=True)
    selection = split_data.iloc[:-holdout_bars].copy().reset_index(drop=True)
    if holdout_path is not None:
        Path(holdout_path).parent.mkdir(parents=True, exist_ok=True)
        holdout.to_parquet(holdout_path, index=False)

    holdout_ts = pd.to_datetime(holdout["timestamp"], utc=True)
    selection_ts = pd.to_datetime(selection["timestamp"], utc=True)
    meta = {
        "enabled": True,
        "holdout_bars": holdout_bars,
        "selection_rows": int(len(selection)),
        "holdout_rows": int(len(holdout)),
        "selection_data_start": str(selection_ts.min()),
        "selection_data_end": str(selection_ts.max()),
        "holdout_data_start": str(holdout_ts.min()),
        "holdout_data_end": str(holdout_ts.max()),
        "holdout_path": str(holdout_path or ""),
        "policy": str(
            holdout_cfg.get(
                "policy",
                "profile_selection_only_before_holdout; holdout is reserved for one-shot final validation",
            )
        ),
        "split_mode": split_mode,
        "unused_rows_after_anchor": unused_rows_after_anchor,
        **monitor_state,
    }
    return selection, holdout, meta

def _holdout_boundary_audit_frame(entries: list[dict[str, Any]], settings: dict[str, Any]) -> pd.DataFrame:
    """Verify experiment outputs stop before the reserved holdout window.

    This guards against accidentally diagnosing an old run that was trained before
    the holdout split existed. If any CV/blend/seed entry reaches into the reserved
    holdout period, holdout policy decisions must be treated as invalid.
    """

    columns = [
        "profile",
        "fold_scope",
        "data_start",
        "data_end",
        "holdout_data_start",
        "passed",
        "reason",
    ]
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return pd.DataFrame(columns=columns)

    holdout_start_raw = holdout.get("holdout_data_start")
    if not holdout_start_raw:
        return pd.DataFrame(
            [
                {
                    "profile": "",
                    "fold_scope": "",
                    "data_start": "",
                    "data_end": "",
                    "holdout_data_start": "",
                    "passed": False,
                    "reason": "missing_holdout_data_start",
                }
            ],
            columns=columns,
        )

    holdout_start = pd.to_datetime(holdout_start_raw, utc=True)
    rows = []
    for entry in entries:
        row = entry.get("diagnostics", {}).get("row", {}) or {}
        data_start = str(row.get("data_start", ""))
        data_end = str(row.get("data_end", ""))
        reason = ""
        passed = False
        if not data_end:
            reason = "missing_entry_data_end"
        else:
            try:
                end_ts = pd.to_datetime(data_end, utc=True)
                passed = bool(end_ts < holdout_start)
                if not passed:
                    reason = "entry_data_end_reaches_reserved_holdout"
            except (TypeError, ValueError):
                reason = "invalid_entry_data_end"
        rows.append(
            {
                "profile": str(entry.get("profile", "")),
                "fold_scope": str(entry.get("fold_scope", "")),
                "data_start": data_start,
                "data_end": data_end,
                "holdout_data_start": str(holdout_start),
                "passed": passed,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _write_holdout_boundary_audit(path: Path, audit: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path / "holdout_boundary_audit.csv", index=False)
    (path / "holdout_boundary_audit.md").write_text(
        _table_markdown("Holdout Boundary Audit", audit),
        encoding="utf-8",
    )
    _write_json(path / "holdout_boundary_audit.json", {"rows": audit.to_dict(orient="records")})

def _read_holdout_context(settings: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return pd.DataFrame(), None
    holdout_path = Path(str(holdout.get("holdout_path", "")))
    if not holdout_path.exists():
        return pd.DataFrame(), None

    holdout_frame = pd.read_parquet(holdout_path).copy()
    if holdout_frame.empty or "timestamp" not in holdout_frame.columns:
        return pd.DataFrame(), None
    holdout_frame["timestamp"] = pd.to_datetime(holdout_frame["timestamp"], utc=True)
    holdout_start = pd.to_datetime(holdout.get("holdout_data_start", holdout_frame["timestamp"].min()), utc=True)

    seq_len = int(_cfg(config, ["model", "seq_len"], 64))
    context_rows = max(seq_len - 1, 0)
    context = pd.DataFrame()
    data_dir = _cfg(config, ["paths", "data_dir"], None)
    if data_dir:
        labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet"
        if labeled_path.exists() and context_rows:
            full = pd.read_parquet(labeled_path)
            if "timestamp" in full.columns:
                full = full.copy()
                full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
                selection_end = pd.to_datetime(holdout.get("selection_data_end", holdout_start), utc=True)
                context = full.loc[full["timestamp"] <= selection_end].tail(context_rows).copy()

    if not context.empty:
        frame = pd.concat([context, holdout_frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    else:
        frame = holdout_frame.sort_values("timestamp").reset_index(drop=True)
    return frame, holdout_start

def _load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def _predict_holdout_for_profile(
    *,
    scope_dir: Path,
    manifest: dict[str, Any],
    holdout_context: pd.DataFrame,
    holdout_start: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    profile = str(manifest["profile"])
    cfg = profile_config(config, profile)
    feature_columns = list(manifest["feature_columns"])
    required = [*feature_columns, *list(_cfg(cfg, ["hmm", "features"], []) or []), "label"]
    forward_column = f"fwd_return_{int(_cfg(cfg, ['labeling', 'max_holding_bars'], 10))}h"
    if forward_column not in holdout_context.columns:
        forward_column = "fwd_return_10h"
    required.append(forward_column)
    missing = [column for column in dict.fromkeys(required) if column not in holdout_context.columns]
    if missing:
        raise ValueError(f"Holdout frame is missing columns for {profile}: {missing}")

    torch_device = _device(None)
    batch_size = int(_cfg(cfg, ["training", "batch_size"], 256))
    rows = []
    model_paths = sorted(scope_dir.glob("model_fold_*.pt"))
    for model_path in model_paths:
        fold = int(model_path.stem.rsplit("_", 1)[-1])
        scaler_path = scope_dir / f"scaler_fold_{fold:03d}.pkl"
        hmm_path = scope_dir / f"hmm_fold_{fold:03d}.pkl"
        if not scaler_path.exists() or not hmm_path.exists():
            continue

        part = holdout_context.copy().reset_index(drop=True)
        scaler = joblib.load(scaler_path)
        part.loc[:, feature_columns] = scaler.transform(part[feature_columns])
        hmm = joblib.load(hmm_path)
        part = _add_regime_probs(part, hmm, cfg)
        dataset = _make_dataset(part, feature_columns, cfg)

        checkpoint = _load_torch_checkpoint(model_path, torch_device)
        model = _build_model(len(feature_columns), cfg).to(torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction = _predict_dataset(model, dataset, part, batch_size=batch_size, device=torch_device)
        prediction = prediction.loc[pd.to_datetime(prediction["timestamp"], utc=True) >= holdout_start].copy()
        if prediction.empty:
            continue
        prediction["split"] = "test"
        prediction["fold"] = fold
        prediction["model_fold"] = fold
        prediction["profile"] = profile
        rows.append(prediction)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def _aggregate_holdout_predictions(predictions: pd.DataFrame, *, profile: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    group_keys = ["timestamp"]
    first_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
            "forward_return",
            "tb_return",
            "hit_type",
            "4h_source_timestamp",
            "4h_available_timestamp",
        )
        if column in frame.columns
    ]
    aggregations: dict[str, Any] = {column: (column, "first") for column in first_columns}
    aggregations["prob_long"] = ("prob_long", "mean")
    aggregations["model_fold_count"] = ("model_fold", "nunique")
    for column in [column for column in frame.columns if column.startswith("regime_prob_")]:
        aggregations[column] = (column, "mean")
    out = frame.groupby(group_keys, as_index=False).agg(**aggregations)
    out["split"] = "test"
    out["fold"] = 0
    out["source_row_position"] = np.arange(len(out))
    out["profile"] = profile
    return out.sort_values("timestamp").reset_index(drop=True)

def _holdout_markdown(holdout_evaluation: pd.DataFrame, holdout_decision: dict[str, Any]) -> str:
    lines = ["# Holdout Evaluation", ""]
    if holdout_evaluation.empty:
        lines.append("No holdout evaluation was produced.")
        if holdout_decision:
            lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "mean_rank_ic",
        "mean_long_f1",
        "mean_prauc",
        "calibration_separation",
        "top_10_lift_global",
        "top_10_forward_return_global",
        "cv_policy_name",
        "cv_policy_lift_vs_base",
        "cv_policy_forward_return",
        "holdout_cv_threshold_f1",
        "holdout_cv_threshold_pred_long_rate",
        "holdout_cv_threshold_source",
        "holdout_policy_name",
        "holdout_policy_selection_rate",
        "holdout_policy_lift_vs_base",
        "holdout_policy_forward_return",
        "holdout_policy_pass",
        "holdout_policy_consistency_pass",
        "holdout_policy_consistency_reject_reason",
        "mtf_leakage_passed",
        "holdout_signal_pass",
        "holdout_signal_reject_reason",
        "holdout_threshold_pass",
        "holdout_threshold_reject_reason",
        "holdout_soft_pass",
        "holdout_reject_reason",
        "frozen_selection",
    ]
    visible = holdout_evaluation[[column for column in display_cols if column in holdout_evaluation.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
    return "\n".join(lines)

def _write_holdout_files(
    path: Path,
    *,
    holdout_evaluation: pd.DataFrame,
    holdout_score_bands: pd.DataFrame,
    holdout_thresholds: pd.DataFrame,
    holdout_decision: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    holdout_evaluation.to_csv(path / "holdout_evaluation.csv", index=False)
    holdout_score_bands.to_csv(path / "holdout_score_band_summary.csv", index=False)
    holdout_thresholds.to_csv(path / "holdout_threshold_summary.csv", index=False)
    policy_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_selection_rate",
            "cv_policy_precision",
            "cv_policy_f1",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_positive_forward_return_fold_rate",
            "cv_policy_pass",
            "cv_policy_reject_reason",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_source",
            "holdout_policy_selection_rate",
            "holdout_policy_precision",
            "holdout_policy_recall",
            "holdout_policy_f1",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_selection_rate_delta_vs_cv",
            "holdout_policy_precision_delta_vs_cv",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_policy_reject_reason",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
            "holdout_signal_pass",
            "holdout_signal_reject_reason",
            "holdout_threshold_pass",
            "holdout_threshold_reject_reason",
            "holdout_soft_pass",
            "holdout_reject_reason",
            "frozen_selection",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_evaluation = (
        holdout_evaluation[policy_columns].copy()
        if policy_columns
        else pd.DataFrame()
    )
    holdout_policy_evaluation.to_csv(path / "holdout_policy_evaluation.csv", index=False)
    consistency_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "frozen_selection",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_pass",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_signal_pass",
            "holdout_threshold_pass",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_consistency = (
        holdout_evaluation[consistency_columns].copy()
        if consistency_columns
        else pd.DataFrame()
    )
    holdout_policy_consistency.to_csv(path / "holdout_policy_consistency.csv", index=False)
    (path / "holdout_policy_consistency.md").write_text(
        _table_markdown("Holdout Policy Consistency", holdout_policy_consistency),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_policy_consistency.json",
        {"rows": holdout_policy_consistency.to_dict(orient="records")},
    )
    holdout_policy_decision = _holdout_policy_decision_frame(holdout_decision, config)
    holdout_policy_decision.to_csv(path / "holdout_policy_decision.csv", index=False)
    (path / "holdout_policy_decision.md").write_text(
        _table_markdown("Holdout Policy Decision", holdout_policy_decision),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_policy_decision.json",
        {"rows": holdout_policy_decision.to_dict(orient="records")},
    )
    (path / "holdout_evaluation.md").write_text(
        _holdout_markdown(holdout_evaluation, holdout_decision),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_evaluation.json",
        {
            "decision": holdout_decision,
            "rows": holdout_evaluation.to_dict(orient="records"),
            "score_bands": holdout_score_bands.to_dict(orient="records"),
            "thresholds": holdout_thresholds.to_dict(orient="records"),
            "policy_evaluation": holdout_policy_evaluation.to_dict(orient="records"),
            "policy_consistency": holdout_policy_consistency.to_dict(orient="records"),
            "policy_decision": holdout_policy_decision.to_dict(orient="records"),
        },
    )

def _holdout_policy_decision_frame(
    holdout_decision: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = [
        "available",
        "frozen_selection",
        "frozen_selection_source",
        "score_policy_recommendation",
        "policy_action",
        "holdout_boundary_passed",
        "configured_frozen_candidate",
        "configured_policy_type",
        "configured_policy_name",
        "configured_status",
        "configured_threshold_deployment_allowed",
        "configured_future_oos_candidates",
        "configured_frozen_candidate_available",
        "configured_policy_match",
        "threshold_deployment_blocked_by_policy",
        "frozen_candidate",
        "frozen_cv_policy_name",
        "frozen_holdout_policy_name",
        "frozen_policy_consistency_pass",
        "frozen_signal_pass",
        "frozen_threshold_pass",
        "frozen_soft_pass",
        "frozen_holdout_policy_lift_vs_base",
        "frozen_holdout_policy_forward_return",
        "observed_best_policy_candidate",
        "observed_best_policy_is_frozen",
        "observed_best_policy_lift_vs_base",
        "observed_best_policy_forward_return",
        "do_not_promote_observed_best_from_same_holdout",
        "warning",
    ]
    if not bool(holdout_decision.get("available", False)):
        return pd.DataFrame(columns=columns)

    frozen = holdout_decision.get("frozen_policy_validation") or {}
    observed = holdout_decision.get("observed_best_policy_candidate") or {}
    frozen_selection = str(holdout_decision.get("frozen_selection", ""))
    frozen_selection_source = str(holdout_decision.get("frozen_selection_source", ""))
    observed_name = str(observed.get("candidate", ""))
    holdout_boundary_passed = bool(holdout_decision.get("holdout_boundary_passed", True))
    policy_review = _cfg(config or {}, ["experiments", "policy_review"], {}) or {}
    configured_candidate = str(policy_review.get("frozen_candidate", ""))
    configured_policy_type = str(policy_review.get("policy_type", ""))
    configured_policy_name = str(policy_review.get("policy_name", ""))
    configured_status = str(policy_review.get("status", ""))
    configured_threshold_allowed = bool(policy_review.get("threshold_deployment_allowed", False))
    future_candidates = ",".join(str(item) for item in policy_review.get("future_oos_candidates", []) or [])
    configured_candidate_available = bool(holdout_decision.get("configured_frozen_candidate_available", False))
    configured_policy_match = bool(
        configured_candidate
        and configured_candidate_available
        and configured_policy_name
        and frozen_selection == configured_candidate
        and str(frozen.get("cv_policy_name", "")) == configured_policy_name
        and str(frozen.get("holdout_policy_name", "")) == configured_policy_name
        and (not configured_policy_type or str(frozen.get("cv_policy_type", "")) == configured_policy_type)
        and (not configured_policy_type or str(frozen.get("holdout_policy_type", "")) == configured_policy_type)
    )
    action = _holdout_policy_action(
        frozen=frozen,
        observed_policy=observed,
        frozen_selection=frozen_selection,
        config=config,
        holdout_boundary_passed=holdout_boundary_passed,
    )
    row = {
        "available": True,
        "frozen_selection": frozen_selection,
        "frozen_selection_source": frozen_selection_source,
        "score_policy_recommendation": str(holdout_decision.get("score_policy_recommendation", "")),
        "policy_action": action,
        "holdout_boundary_passed": holdout_boundary_passed,
        "configured_frozen_candidate": configured_candidate,
        "configured_policy_type": configured_policy_type,
        "configured_policy_name": configured_policy_name,
        "configured_status": configured_status,
        "configured_threshold_deployment_allowed": configured_threshold_allowed,
        "configured_future_oos_candidates": future_candidates,
        "configured_frozen_candidate_available": configured_candidate_available,
        "configured_policy_match": configured_policy_match,
        "threshold_deployment_blocked_by_policy": not configured_threshold_allowed,
        "frozen_candidate": str(frozen.get("candidate", "")),
        "frozen_cv_policy_name": str(frozen.get("cv_policy_name", "")),
        "frozen_holdout_policy_name": str(frozen.get("holdout_policy_name", "")),
        "frozen_policy_consistency_pass": bool(frozen.get("holdout_policy_consistency_pass", False)),
        "frozen_signal_pass": bool(frozen.get("holdout_signal_pass", False)),
        "frozen_threshold_pass": bool(frozen.get("holdout_threshold_pass", False)),
        "frozen_soft_pass": bool(frozen.get("holdout_soft_pass", False)),
        "frozen_holdout_policy_lift_vs_base": _float(frozen, "holdout_policy_lift_vs_base"),
        "frozen_holdout_policy_forward_return": _float(frozen, "holdout_policy_forward_return"),
        "observed_best_policy_candidate": observed_name,
        "observed_best_policy_is_frozen": bool(observed_name and observed_name == frozen_selection),
        "observed_best_policy_lift_vs_base": _float(observed, "holdout_policy_lift_vs_base"),
        "observed_best_policy_forward_return": _float(observed, "holdout_policy_forward_return"),
        "do_not_promote_observed_best_from_same_holdout": bool(observed_name and observed_name != frozen_selection),
        "warning": str(holdout_decision.get("observed_best_policy_warning", "")),
    }
    return pd.DataFrame([row], columns=columns)

def _binary_metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = pd.to_numeric(scores, errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    y_pred = (y_score >= float(threshold)).astype(int)
    true_positive = float(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = float(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
    }

def _attach_holdout_cv_threshold_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    threshold, source = _cv_selected_threshold(cv_entry)
    metrics = _binary_metrics_at_threshold(predictions["label"], predictions["prob_long"], threshold)
    row["holdout_cv_threshold"] = float(threshold)
    row["holdout_cv_threshold_source"] = source
    row["holdout_cv_threshold_f1"] = metrics["f1"]
    row["holdout_cv_threshold_precision"] = metrics["precision"]
    row["holdout_cv_threshold_recall"] = metrics["recall"]
    row["holdout_cv_threshold_pred_long_rate"] = metrics["pred_long_rate"]
    return row

def _policy_metrics_from_mask(predictions: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    selected = predictions.loc[mask].copy()
    base_long_rate = float(predictions["label"].mean()) if len(predictions) else np.nan
    if selected.empty:
        return {
            "selection_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "lift_vs_base": np.nan,
            "forward_return": np.nan,
        }
    true_positive = float(selected["label"].astype(int).sum())
    total_positive = float(predictions["label"].astype(int).sum())
    precision = float(selected["label"].mean())
    recall = true_positive / max(total_positive, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "selection_rate": float(len(selected) / len(predictions)) if len(predictions) else np.nan,
        "precision": precision,
        "recall": float(recall),
        "f1": float(f1),
        "lift_vs_base": float(precision / base_long_rate) if base_long_rate and base_long_rate > 0 else np.nan,
        "forward_return": float(selected["forward_return"].mean()) if "forward_return" in selected.columns else np.nan,
    }

def _evaluate_score_policy_on_holdout(
    predictions: pd.DataFrame,
    policy: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not policy:
        return {"source": "missing_cv_policy", "reject_reason": "missing_cv_policy"}
    policy_type = str(policy.get("policy_type", ""))
    policy_name = str(policy.get("policy_name", ""))
    if policy_type == "score_band":
        score_bins = int(_cfg(config, ["validation", "score_lift_bins"], _cfg(config, ["validation", "calibration_bins"], 10)))
        score_bands = _cfg(config, ["validation", "score_bands"], None)
        band_rows = score_band_diagnostics(predictions, bins=score_bins, bands=score_bands)
        matched = band_rows.loc[band_rows["band"].astype(str) == policy_name]
        if matched.empty:
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_holdout_band",
            }
        item = matched.iloc[0].to_dict()
        metrics = {
            "selection_rate": _float(item, "selection_rate"),
            "precision": _float(item, "actual_long_rate"),
            "recall": _float(item, "recall"),
            "f1": _float(item, "f1"),
            "lift_vs_base": _float(item, "lift_vs_base"),
            "forward_return": _float(item, "mean_forward_return"),
        }
    elif policy_type == "threshold_cap":
        threshold = _float(policy, "threshold_mean", np.nan)
        if not np.isfinite(threshold):
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_cv_threshold_mean",
            }
        mask = pd.to_numeric(predictions["prob_long"], errors="coerce") >= threshold
        metrics = _policy_metrics_from_mask(predictions, mask)
    else:
        return {
            "name": policy_name,
            "type": policy_type,
            "source": "cv_score_policy_selection",
            "reject_reason": "unknown_policy_type",
        }

    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    policy_cfg = _cfg(config, ["validation", "policy_selection"], {}) or {}
    max_selection_rate = float(policy_cfg.get("max_selection_rate", threshold_cfg.get("max_pred_long_rate", 0.70)))
    min_precision = float(policy_cfg.get("min_precision", threshold_cfg.get("min_precision", 0.30)))
    min_lift = float(policy_cfg.get("min_lift_vs_base", 1.0))
    min_forward_return = float(policy_cfg.get("min_forward_return", 0.0))
    reasons = []
    if metrics["selection_rate"] > max_selection_rate:
        reasons.append("selection_rate")
    if metrics["precision"] < min_precision:
        reasons.append("precision")
    if not np.isfinite(metrics["lift_vs_base"]) or metrics["lift_vs_base"] <= min_lift:
        reasons.append("lift_vs_base")
    if not np.isfinite(metrics["forward_return"]) or metrics["forward_return"] <= min_forward_return:
        reasons.append("forward_return")
    return {
        "name": policy_name,
        "type": policy_type,
        "source": "cv_score_policy_selection",
        **metrics,
        "pass": len(reasons) == 0,
        "reject_reason": ";".join(reasons),
    }

def _attach_holdout_policy_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = _cv_score_policy(cv_entry)
    row["cv_policy_name"] = str(policy.get("policy_name", ""))
    row["cv_policy_type"] = str(policy.get("policy_type", ""))
    row["cv_policy_selection_rate"] = _float(policy, "selection_rate")
    row["cv_policy_precision"] = _float(policy, "precision")
    row["cv_policy_recall"] = _float(policy, "recall")
    row["cv_policy_f1"] = _float(policy, "f1")
    row["cv_policy_lift_vs_base"] = _float(policy, "lift_vs_base")
    row["cv_policy_forward_return"] = _float(policy, "forward_return")
    row["cv_policy_positive_lift_fold_rate"] = _float(policy, "positive_lift_fold_rate")
    row["cv_policy_positive_forward_return_fold_rate"] = _float(policy, "positive_forward_return_fold_rate")
    row["cv_policy_pass"] = bool(policy.get("policy_pass", False))
    row["cv_policy_reject_reason"] = str(policy.get("policy_reject_reason", ""))
    metrics = _evaluate_score_policy_on_holdout(predictions, policy, config)
    row["holdout_policy_name"] = metrics.get("name", "")
    row["holdout_policy_type"] = metrics.get("type", "")
    row["holdout_policy_source"] = metrics.get("source", "")
    row["holdout_policy_selection_rate"] = metrics.get("selection_rate", np.nan)
    row["holdout_policy_precision"] = metrics.get("precision", np.nan)
    row["holdout_policy_recall"] = metrics.get("recall", np.nan)
    row["holdout_policy_f1"] = metrics.get("f1", np.nan)
    row["holdout_policy_lift_vs_base"] = metrics.get("lift_vs_base", np.nan)
    row["holdout_policy_forward_return"] = metrics.get("forward_return", np.nan)
    row["holdout_policy_pass"] = bool(metrics.get("pass", False))
    row["holdout_policy_reject_reason"] = metrics.get("reject_reason", "")
    return row

def _attach_holdout_policy_consistency(row: dict[str, Any]) -> dict[str, Any]:
    row["holdout_policy_selection_rate_delta_vs_cv"] = (
        _float(row, "holdout_policy_selection_rate") - _float(row, "cv_policy_selection_rate")
    )
    row["holdout_policy_precision_delta_vs_cv"] = (
        _float(row, "holdout_policy_precision") - _float(row, "cv_policy_precision")
    )
    row["holdout_policy_lift_delta_vs_cv"] = (
        _float(row, "holdout_policy_lift_vs_base") - _float(row, "cv_policy_lift_vs_base")
    )
    row["holdout_policy_forward_return_delta_vs_cv"] = (
        _float(row, "holdout_policy_forward_return") - _float(row, "cv_policy_forward_return")
    )

    reasons = []
    if not str(row.get("cv_policy_name", "")).strip():
        reasons.append("missing_cv_policy")
    if not bool(row.get("cv_policy_pass", False)):
        reasons.append("cv_policy")
    if str(row.get("cv_policy_name", "")) != str(row.get("holdout_policy_name", "")):
        reasons.append("policy_name_mismatch")
    if str(row.get("cv_policy_type", "")) != str(row.get("holdout_policy_type", "")):
        reasons.append("policy_type_mismatch")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if not bool(row.get("holdout_signal_pass", False)):
        reasons.append("holdout_signal")
    row["holdout_policy_consistency_pass"] = len(reasons) == 0
    row["holdout_policy_consistency_reject_reason"] = ";".join(reasons)
    return row

def _holdout_signal_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    reasons = []
    if float(row.get("mean_rank_ic", 0.0)) <= target_rank_ic:
        reasons.append("mean_rank_ic")
    if float(row.get("top_10_lift_global", 0.0)) <= 1.0:
        reasons.append("top_10_lift_global")
    if float(row.get("top_10_forward_return_global", 0.0)) <= 0.0:
        reasons.append("top_10_forward_return_global")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if float(row.get("calibration_separation", 0.0)) <= 0.0:
        reasons.append("calibration_separation")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", True)):
        reasons.append("stationarity_policy")
    return reasons

def _holdout_threshold_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    reasons = []
    if float(row.get("holdout_cv_threshold_f1", 0.0)) <= min_long_f1:
        reasons.append("holdout_cv_threshold_f1")
    if float(row.get("holdout_cv_threshold_pred_long_rate", 1.0)) > max_pred_long_rate:
        reasons.append("holdout_cv_threshold_pred_long_rate")
    return reasons

def _attach_holdout_soft_pass(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    signal_reasons = _holdout_signal_pass_reasons(row, config)
    threshold_reasons = _holdout_threshold_pass_reasons(row, config)
    reasons = [*signal_reasons, *threshold_reasons]
    row["holdout_signal_pass"] = len(signal_reasons) == 0
    row["holdout_signal_reject_reason"] = ";".join(signal_reasons)
    row["holdout_threshold_pass"] = len(threshold_reasons) == 0
    row["holdout_threshold_reject_reason"] = ";".join(threshold_reasons)
    row["holdout_soft_pass"] = len(reasons) == 0
    row["holdout_reject_reason"] = ";".join(reasons)
    return row

def _frozen_policy_monitoring_plan_frame(config: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    monitor = policy_review.get("future_oos_monitor", {}) or {}
    holdout = settings.get("holdout", {}) or {}
    latest_data_end = _holdout_latest_available_data_end(holdout)
    monitor_state = _future_oos_monitor_state(config, latest_data_end)
    ready_at = _future_oos_ready_at_fields(monitor_state)
    row = {
        "enabled": bool(monitor.get("enabled", False)),
        "frozen_candidate": str(policy_review.get("frozen_candidate", "")),
        "policy_type": str(policy_review.get("policy_type", "")),
        "policy_name": str(policy_review.get("policy_name", "")),
        "status": str(policy_review.get("status", "")),
        "threshold_deployment_allowed": bool(policy_review.get("threshold_deployment_allowed", False)),
        "future_oos_candidates": ",".join(str(item) for item in policy_review.get("future_oos_candidates", []) or []),
        "anchor_run_id": monitor_state["anchor_run_id"],
        "anchor_data_end": monitor_state["anchor_data_end"],
        "latest_available_data_end": monitor_state["latest_available_data_end"],
        "new_bars_since_anchor": monitor_state["new_bars_since_anchor"],
        "min_new_bars": monitor_state["min_new_bars"],
        "preferred_new_bars": monitor_state["preferred_new_bars"],
        "min_new_bars_remaining": monitor_state["min_new_bars_remaining"],
        "preferred_new_bars_remaining": monitor_state["preferred_new_bars_remaining"],
        "min_ready_at": ready_at["min_ready_at"],
        "preferred_ready_at": ready_at["preferred_ready_at"],
        "future_oos_ready": monitor_state["future_oos_ready"],
        "future_oos_preferred_ready": monitor_state["future_oos_preferred_ready"],
        "allow_holdout_roll_forward": monitor_state["allow_holdout_roll_forward"],
        "holdout_roll_forward_locked": monitor_state["holdout_roll_forward_locked"],
        "current_holdout_data_end": str(holdout.get("holdout_data_end", "") or ""),
        "frozen_holdout_data_end": str(holdout.get("holdout_data_end", "") or ""),
        "next_action": monitor_state["next_action"],
        "policy": str(monitor.get("policy", "")),
    }
    return pd.DataFrame([row])

def _write_frozen_policy_monitoring_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "frozen_policy_monitoring_plan.csv", index=False)
    (path / "frozen_policy_monitoring_plan.md").write_text(
        _table_markdown("Frozen Policy Monitoring Plan", frame),
        encoding="utf-8",
    )
    _write_json(path / "frozen_policy_monitoring_plan.json", {"rows": frame.to_dict(orient="records")})

def _experiment_policy_guard_frame(settings: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    guard = copy.deepcopy(settings.get("experiment_policy_guard") or _experiment_policy_guard(settings, config))
    ready_at = _future_oos_ready_at_fields(guard)
    row = {
        "enabled": bool(guard.get("enabled", False)),
        "status": str(guard.get("status", "")),
        "profile_search_locked": bool(guard.get("profile_search_locked", False)),
        "action": str(guard.get("action", "")),
        "reason": str(guard.get("reason", "")),
        "allowed_benchmark_profiles": ",".join(str(item) for item in guard.get("allowed_benchmark_profiles", []) or []),
        "blocked_candidate_profiles": ",".join(str(item) for item in guard.get("blocked_candidate_profiles", []) or []),
        "blocked_full_profiles": ",".join(str(item) for item in guard.get("blocked_full_profiles", []) or []),
        "blocked_seed_profiles": ",".join(str(item) for item in guard.get("blocked_seed_profiles", []) or []),
        "future_oos_ready": bool(guard.get("future_oos_ready", False)),
        "future_oos_preferred_ready": bool(guard.get("future_oos_preferred_ready", False)),
        "new_bars_since_anchor": int(guard.get("new_bars_since_anchor", 0) or 0),
        "min_new_bars_remaining": int(guard.get("min_new_bars_remaining", 0) or 0),
        "preferred_new_bars_remaining": int(guard.get("preferred_new_bars_remaining", 0) or 0),
        "min_ready_at": ready_at["min_ready_at"],
        "preferred_ready_at": ready_at["preferred_ready_at"],
        "holdout_roll_forward_locked": bool(guard.get("holdout_roll_forward_locked", False)),
        "next_action": str(guard.get("next_action", "")),
        "anchor_run_id": str(guard.get("anchor_run_id", "")),
        "anchor_data_end": str(guard.get("anchor_data_end", "")),
        "latest_available_data_end": str(guard.get("latest_available_data_end", "")),
    }
    return pd.DataFrame([row])

def _write_experiment_policy_guard(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "experiment_policy_guard.csv", index=False)
    (path / "experiment_policy_guard.md").write_text(
        _table_markdown("Experiment Policy Guard", frame),
        encoding="utf-8",
    )
    _write_json(path / "experiment_policy_guard.json", {"rows": frame.to_dict(orient="records")})

def _recommendation_with_policy_guard(recommendation: str, settings: dict[str, Any]) -> str:
    guard = settings.get("experiment_policy_guard", {}) or {}
    if bool(guard.get("profile_search_locked", False)) and recommendation not in {
        "fix_missing_selected_profiles",
        "rerun_training_with_holdout_split",
    }:
        return str(guard.get("action") or "wait_for_new_unseen_bars_keep_control_profile")
    return recommendation

def _future_oos_candidate_plan_frame(
    settings: dict[str, Any],
    config: dict[str, Any],
    payoff_policy_robustness_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    ready_at = _future_oos_ready_at_fields(guard)
    profiles_cfg = _cfg(config, ["features", "profiles"], {}) or {}
    weighted_blends = _cfg(config, ["experiments", "profile_blends", "weighted"], []) or []
    rows: list[dict[str, Any]] = []

    def weighted_blend_profiles(candidate: str) -> list[str]:
        for blend in weighted_blends:
            if not isinstance(blend, dict):
                continue
            name = str(blend.get("name", ""))
            if candidate not in {name, f"blend_{name}"}:
                continue
            return [str(profile) for profile in blend.get("profiles", []) or [] if str(profile)]
        return []

    def add_row(
        *,
        candidate: str,
        candidate_type: str,
        stage: str,
        required_profiles: list[str] | None = None,
        note: str = "",
        policy_name: str = "",
        policy_type: str = "",
        selection_source: str = "",
        cv_mean_label_lift_vs_base: float | None = None,
        cv_mean_forward_return: float | None = None,
        cv_payoff_alignment_fold_rate: float | None = None,
        current_holdout_mean_label_lift_vs_base: float | None = None,
        current_holdout_mean_forward_return: float | None = None,
        current_holdout_mean_tb_return: float | None = None,
        current_holdout_payoff_alignment_fold_rate: float | None = None,
        current_holdout_reject_reason: str = "",
    ) -> None:
        required = [str(profile) for profile in required_profiles or [] if str(profile)]
        missing_profiles = [profile for profile in required if profile not in profiles_cfg]
        allowed = set(str(item) for item in guard.get("allowed_benchmark_profiles", []) or [])
        all_required_allowed = all(profile in allowed for profile in required) if required else candidate in allowed
        is_retired = stage == "retired_frozen_policy"
        candidate_status = {
            "control_profile": "active_control",
            "future_oos_candidate": "pre_registered_future_oos_candidate",
            "future_oos_score_band_policy": "pre_registered_future_oos_policy",
            "retired_frozen_policy": "historical_retired_policy_do_not_promote",
        }.get(stage, "diagnostic_candidate")
        candidate_id = candidate if not policy_name else f"{candidate}::{policy_name}"
        candidate_label = candidate if not policy_name else f"{candidate} [{policy_name}]"
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": candidate_label,
                "candidate": candidate,
                "candidate_type": candidate_type,
                "stage": stage,
                "required_profiles": ",".join(required),
                "missing_required_profiles": ",".join(missing_profiles),
                "all_required_profiles_allowed": all_required_allowed,
                "candidate_status": candidate_status,
                "profile_search_locked": bool(guard.get("profile_search_locked", False)),
                "future_oos_ready": bool(guard.get("future_oos_ready", False)),
                "min_new_bars_remaining": int(guard.get("min_new_bars_remaining", 0) or 0),
                "min_ready_at": ready_at["min_ready_at"],
                "preferred_ready_at": ready_at["preferred_ready_at"],
                "action": str(guard.get("action", "")),
                "evaluation_status": "wait_for_future_oos" if not bool(guard.get("future_oos_ready", False)) else "ready_for_future_oos_review",
                "promotion_allowed_now": (
                    bool(guard.get("future_oos_ready", False))
                    and bool(all_required_allowed)
                    and not is_retired
                ),
                "note": note,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "selection_source": selection_source,
                "cv_mean_label_lift_vs_base": cv_mean_label_lift_vs_base,
                "cv_mean_forward_return": cv_mean_forward_return,
                "cv_payoff_alignment_fold_rate": cv_payoff_alignment_fold_rate,
                "current_holdout_diagnostic_only": bool(policy_name),
                "current_holdout_mean_label_lift_vs_base": current_holdout_mean_label_lift_vs_base,
                "current_holdout_mean_forward_return": current_holdout_mean_forward_return,
                "current_holdout_mean_tb_return": current_holdout_mean_tb_return,
                "current_holdout_payoff_alignment_fold_rate": current_holdout_payoff_alignment_fold_rate,
                "current_holdout_reject_reason": current_holdout_reject_reason,
            }
        )

    control = str(settings.get("control_profile", ""))
    if control:
        add_row(
            candidate=control,
            candidate_type="profile",
            stage="control_profile",
            required_profiles=[control],
            note="Current control profile remains the safe baseline.",
        )

    frozen = str(policy_review.get("frozen_candidate", "")).strip()
    if frozen:
        add_row(
            candidate=frozen,
            candidate_type=str(policy_review.get("policy_type", "score_policy")),
            stage="retired_frozen_policy",
            note=str(policy_review.get("note", "")),
        )

    future_items = [str(item) for item in policy_review.get("future_oos_candidates", []) or []]
    for item in future_items:
        matched = False
        for blend in weighted_blends:
            if not isinstance(blend, dict):
                continue
            name = str(blend.get("name", ""))
            if item not in {name, f"blend_{name}"}:
                continue
            add_row(
                candidate=item,
                candidate_type="weighted_blend",
                stage="future_oos_candidate",
                required_profiles=[str(profile) for profile in blend.get("profiles", []) or []],
                note=str(blend.get("description", "")),
            )
            matched = True
            break
        if matched:
            continue
        add_row(
            candidate=item,
            candidate_type="profile" if item in profiles_cfg else "unknown",
            stage="future_oos_candidate",
            required_profiles=[item] if item in profiles_cfg else [],
            note="" if item in profiles_cfg else "Candidate is not a known feature profile or configured weighted blend.",
        )

    if payoff_policy_robustness_summary is not None and not payoff_policy_robustness_summary.empty:
        policy_rows = payoff_policy_robustness_summary.copy()
        holdout_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if {"candidate", "band", "evaluation_scope"}.issubset(policy_rows.columns):
            holdout_rows = policy_rows[policy_rows["evaluation_scope"].astype(str) == "holdout"]
            holdout_lookup = {
                (str(row.get("candidate", "")).strip(), str(row.get("band", "")).strip()): row.to_dict()
                for _, row in holdout_rows.iterrows()
            }
        if "future_oos_policy_candidate" in policy_rows.columns:
            candidate_mask = policy_rows["future_oos_policy_candidate"].map(
                lambda value: bool(value) if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"1", "true", "yes"}
            )
            policy_rows = policy_rows[
                (policy_rows.get("evaluation_scope", "").astype(str) == "cv_test")
                & candidate_mask
            ]
        else:
            policy_rows = policy_rows.iloc[0:0]
        existing_policy_keys = {
            (str(row.get("candidate", "")), str(row.get("stage", "")), str(row.get("policy_name", "")))
            for row in rows
        }
        for _, policy_row in policy_rows.iterrows():
            candidate = str(policy_row.get("candidate", "")).strip()
            band = str(policy_row.get("band", "")).strip()
            if not candidate or not band:
                continue
            key = (candidate, "future_oos_score_band_policy", band)
            if key in existing_policy_keys:
                continue
            if candidate in profiles_cfg:
                required_profiles = [candidate]
                candidate_type = "profile_score_band"
            else:
                required_profiles = weighted_blend_profiles(candidate)
                candidate_type = "weighted_blend_score_band" if required_profiles else "score_band_policy"
            current_holdout = holdout_lookup.get((candidate, band), {})
            add_row(
                candidate=candidate,
                candidate_type=candidate_type,
                stage="future_oos_score_band_policy",
                required_profiles=required_profiles,
                note=(
                    "CV payoff-policy robustness pre-registered this score band for future unseen OOS review. "
                    "Current holdout remains diagnostic-only and must not be used for promotion."
                ),
                policy_name=band,
                policy_type="score_band",
                selection_source="cv_payoff_policy_robustness",
                cv_mean_label_lift_vs_base=_optional_float(policy_row.get("mean_label_lift_vs_base")),
                cv_mean_forward_return=_optional_float(policy_row.get("mean_forward_return")),
                cv_payoff_alignment_fold_rate=_optional_float(policy_row.get("payoff_alignment_fold_rate")),
                current_holdout_mean_label_lift_vs_base=_optional_float(current_holdout.get("mean_label_lift_vs_base")),
                current_holdout_mean_forward_return=_optional_float(current_holdout.get("mean_forward_return")),
                current_holdout_mean_tb_return=_optional_float(current_holdout.get("mean_tb_return")),
                current_holdout_payoff_alignment_fold_rate=_optional_float(current_holdout.get("payoff_alignment_fold_rate")),
                current_holdout_reject_reason=str(current_holdout.get("reject_reason", "")),
            )
            existing_policy_keys.add(key)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    stage_order = {
        "control_profile": 0,
        "future_oos_candidate": 20,
        "future_oos_score_band_policy": 30,
        "retired_frozen_policy": 80,
    }
    policy_order = {
        "top_10": 10,
        "top_20": 20,
        "top_30": 30,
        "upper_half": 50,
        "mid_upper_40_90": 60,
    }
    out = frame.copy()
    out["_stage_order"] = out["stage"].map(stage_order).fillna(99).astype(int)
    out["_policy_order"] = out["policy_name"].map(policy_order).fillna(999).astype(int)
    out = out.sort_values(
        ["_stage_order", "candidate_type", "candidate_label", "_policy_order", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "plan_rank", np.arange(1, len(out) + 1, dtype=int))
    return out.drop(columns=["_stage_order", "_policy_order"])

def _write_future_oos_candidate_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "future_oos_candidate_plan.csv", index=False)
    (path / "future_oos_candidate_plan.md").write_text(
        _table_markdown("Future OOS Candidate Plan", frame),
        encoding="utf-8",
    )
    _write_json(path / "future_oos_candidate_plan.json", {"rows": frame.to_dict(orient="records")})

def _performance_gap_reasons(row: dict[str, Any], config: dict[str, Any]) -> str:
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    max_rank_ic_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    min_positive_ic_fraction = float(_cfg(config, ["validation", "min_positive_ic_fraction"], 0.75))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    reasons = []
    if _float(row, "mean_rank_ic") < target_rank_ic:
        reasons.append("cv_rank_ic_below_target")
    if _float(row, "std_rank_ic") > max_rank_ic_std:
        reasons.append("cv_rank_ic_std_above_phase1_target")
    if _float(row, "positive_ic_fraction") < min_positive_ic_fraction:
        reasons.append("cv_positive_ic_fraction_below_target")
    selected_f1 = _float(row, "test_f1_at_selected_threshold", np.nan)
    constrained_f1 = _float(row, "test_f1_at_constrained_threshold", np.nan)
    guarded_f1 = _float(row, "test_f1_at_guarded_threshold", np.nan)
    official_f1 = _float(row, "test_f1_at_official_threshold", guarded_f1)
    fixed_f1 = _float(row, "mean_long_f1", np.nan)
    if selected_f1 < min_long_f1:
        reasons.append("cv_selected_threshold_f1_below_target")
    if constrained_f1 < min_long_f1:
        reasons.append("cv_constrained_threshold_f1_below_target")
    if np.isfinite(guarded_f1) and guarded_f1 < min_long_f1:
        reasons.append("cv_guarded_threshold_f1_below_target")
    if np.isfinite(official_f1) and official_f1 < min_long_f1:
        reasons.append("cv_official_threshold_f1_below_target")
    if not np.isfinite(selected_f1) and not np.isfinite(constrained_f1) and not np.isfinite(guarded_f1) and not np.isfinite(official_f1) and fixed_f1 < min_long_f1:
        reasons.append("cv_fixed_0_50_f1_below_target")
    if _float(row, "test_pred_long_rate_at_selected_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_selected_threshold_pred_long_rate_above_guardrail")
    if _float(row, "test_pred_long_rate_at_constrained_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_constrained_threshold_pred_long_rate_above_guardrail")
    if _float(row, "test_pred_long_rate_at_official_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_official_threshold_pred_long_rate_above_guardrail")
    if _float(row, "top_10_lift_global") < 1.0:
        reasons.append("cv_top_10_lift_below_base")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return ";".join(reasons)

def _holdout_gap_reasons(holdout_row: dict[str, Any], config: dict[str, Any]) -> str:
    if not holdout_row:
        return "missing_holdout_evaluation"
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    reasons = []
    if _float(holdout_row, "mean_rank_ic") < target_rank_ic:
        reasons.append("holdout_rank_ic_below_target")
    if _float(holdout_row, "top_10_lift_global") <= 1.0:
        reasons.append("holdout_top_10_lift_not_above_base")
    if _float(holdout_row, "top_10_forward_return_global") <= 0.0:
        reasons.append("holdout_top_10_forward_return_not_positive")
    if _float(holdout_row, "holdout_cv_threshold_f1") < min_long_f1:
        reasons.append("holdout_cv_threshold_f1_below_target")
    if not bool(holdout_row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if not bool(holdout_row.get("holdout_signal_pass", False)):
        signal_reason = str(holdout_row.get("holdout_signal_reject_reason", "holdout_signal")).strip(";")
        reasons.append(signal_reason or "holdout_signal")
    if not bool(holdout_row.get("holdout_threshold_pass", False)):
        threshold_reason = str(holdout_row.get("holdout_threshold_reject_reason", "holdout_threshold")).strip(";")
        reasons.append(threshold_reason or "holdout_threshold")
    if not bool(holdout_row.get("mtf_leakage_passed", False)):
        reasons.append("holdout_mtf_leakage")
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))

def _performance_gap_action(
    *,
    cv_reasons: str,
    holdout_reasons: str,
    guard: dict[str, Any],
    candidate_type: str,
) -> str:
    if bool(guard.get("profile_search_locked", False)):
        return "wait_for_future_oos_do_not_tune_current_holdout"
    if holdout_reasons and holdout_reasons != "missing_holdout_evaluation":
        return "do_not_promote_investigate_holdout_failure"
    if cv_reasons:
        return "improve_cv_stability_before_promotion"
    if candidate_type == "blend":
        return "candidate_blend_ready_for_predefined_future_oos_review"
    return "candidate_profile_ready_for_predefined_future_oos_review"

def _performance_gap_analysis_frame(
    entries: list[dict[str, Any]],
    holdout_evaluation: pd.DataFrame,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature_count",
        "fold_count",
        "cv_mean_rank_ic",
        "cv_std_rank_ic",
        "cv_positive_ic_fraction",
        "cv_worst_5_rank_ic_mean",
        "cv_top_10_lift_global",
        "cv_top_10_forward_return_global",
        "cv_selected_threshold_f1",
        "cv_constrained_threshold_f1",
        "cv_guarded_threshold_f1",
        "cv_guarded_threshold_source",
        "cv_guarded_threshold_pred_long_rate",
        "cv_official_threshold_f1",
        "cv_official_threshold_source",
        "cv_official_threshold_pred_long_rate",
        "cv_calibrated_guarded_threshold_f1",
        "cv_calibrated_guarded_threshold_pred_long_rate",
        "holdout_available",
        "holdout_mean_rank_ic",
        "holdout_top_10_lift_global",
        "holdout_top_10_forward_return_global",
        "holdout_cv_threshold_f1",
        "holdout_policy_lift_vs_base",
        "holdout_policy_forward_return",
        "holdout_soft_pass",
        "cv_to_holdout_rank_ic_delta",
        "cv_to_holdout_top_10_lift_delta",
        "cv_to_holdout_top_10_forward_return_delta",
        "cv_phase1_blockers",
        "holdout_blockers",
        "profile_search_locked",
        "future_oos_ready",
        "next_action",
        "research_track",
        "note",
    ]
    rows: list[dict[str, Any]] = []
    holdout_by_candidate = {}
    if not holdout_evaluation.empty and "candidate" in holdout_evaluation.columns:
        holdout_by_candidate = {
            str(row["candidate"]): row.to_dict()
            for _, row in holdout_evaluation.iterrows()
        }
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope != "full" and not fold_scope.startswith("blend_"):
            continue
        row = dict(entry["diagnostics"]["row"])
        candidate = str(row.get("profile", entry.get("profile", "")))
        key = (candidate, fold_scope)
        if key in seen:
            continue
        seen.add(key)
        candidate_type = "blend" if fold_scope.startswith("blend_") else "profile"
        holdout_row = holdout_by_candidate.get(candidate, {})
        cv_reasons = _performance_gap_reasons(row, config)
        holdout_reasons = _holdout_gap_reasons(holdout_row, config) if holdout_by_candidate else "missing_holdout_evaluation"
        tracks = []
        if "std" in cv_reasons or "positive_ic_fraction" in cv_reasons:
            tracks.append("fold_stability")
        if "f1" in cv_reasons or "threshold" in holdout_reasons:
            tracks.append("threshold_calibration")
        if "top_10" in holdout_reasons or "policy" in holdout_reasons:
            tracks.append("score_band_policy")
        if "forward_return" in holdout_reasons:
            tracks.append("feature_regime_mismatch")
        if not tracks:
            tracks.append("future_oos_validation")
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "feature_count": int(row.get("feature_count", 0) or 0),
                "fold_count": int(row.get("fold_count", 0) or 0),
                "cv_mean_rank_ic": _float(row, "mean_rank_ic"),
                "cv_std_rank_ic": _float(row, "std_rank_ic"),
                "cv_positive_ic_fraction": _float(row, "positive_ic_fraction"),
                "cv_worst_5_rank_ic_mean": _float(row, "worst_5_rank_ic_mean"),
                "cv_top_10_lift_global": _float(row, "top_10_lift_global"),
                "cv_top_10_forward_return_global": _float(row, "top_10_forward_return_global"),
                "cv_selected_threshold_f1": _float(row, "test_f1_at_selected_threshold"),
                "cv_constrained_threshold_f1": _float(row, "test_f1_at_constrained_threshold"),
                "cv_guarded_threshold_f1": _float(row, "test_f1_at_guarded_threshold"),
                "cv_guarded_threshold_source": str(row.get("guarded_threshold_source", "")),
                "cv_guarded_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_guarded_threshold"),
                "cv_official_threshold_f1": _float(row, "test_f1_at_official_threshold"),
                "cv_official_threshold_source": str(row.get("official_threshold_source", "")),
                "cv_official_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_official_threshold"),
                "cv_calibrated_guarded_threshold_f1": _float(row, "test_f1_at_calibrated_guarded_threshold"),
                "cv_calibrated_guarded_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_calibrated_guarded_threshold"),
                "holdout_available": bool(holdout_row),
                "holdout_mean_rank_ic": _float(holdout_row, "mean_rank_ic") if holdout_row else np.nan,
                "holdout_top_10_lift_global": _float(holdout_row, "top_10_lift_global") if holdout_row else np.nan,
                "holdout_top_10_forward_return_global": _float(holdout_row, "top_10_forward_return_global") if holdout_row else np.nan,
                "holdout_cv_threshold_f1": _float(holdout_row, "holdout_cv_threshold_f1") if holdout_row else np.nan,
                "holdout_policy_lift_vs_base": _float(holdout_row, "holdout_policy_lift_vs_base") if holdout_row else np.nan,
                "holdout_policy_forward_return": _float(holdout_row, "holdout_policy_forward_return") if holdout_row else np.nan,
                "holdout_soft_pass": bool(holdout_row.get("holdout_soft_pass", False)) if holdout_row else False,
                "cv_to_holdout_rank_ic_delta": (
                    _float(holdout_row, "mean_rank_ic") - _float(row, "mean_rank_ic") if holdout_row else np.nan
                ),
                "cv_to_holdout_top_10_lift_delta": (
                    _float(holdout_row, "top_10_lift_global") - _float(row, "top_10_lift_global") if holdout_row else np.nan
                ),
                "cv_to_holdout_top_10_forward_return_delta": (
                    _float(holdout_row, "top_10_forward_return_global") - _float(row, "top_10_forward_return_global")
                    if holdout_row
                    else np.nan
                ),
                "cv_phase1_blockers": cv_reasons,
                "holdout_blockers": holdout_reasons,
                "profile_search_locked": bool(guard.get("profile_search_locked", False)),
                "future_oos_ready": bool(guard.get("future_oos_ready", False)),
                "next_action": _performance_gap_action(
                    cv_reasons=cv_reasons,
                    holdout_reasons=holdout_reasons,
                    guard=guard,
                    candidate_type=candidate_type,
                ),
                "research_track": ";".join(dict.fromkeys(tracks)),
                "note": "Diagnostics only; do not tune profiles or weights against the current frozen holdout.",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(
            ["candidate_type", "cv_mean_rank_ic", "cv_top_10_lift_global"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

def _performance_gap_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Performance Gap Analysis", ""]
    if frame.empty:
        lines.append("No full-profile or blend candidates were available for performance gap analysis.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "cv_mean_rank_ic",
        "cv_std_rank_ic",
        "cv_positive_ic_fraction",
        "cv_top_10_lift_global",
        "cv_guarded_threshold_f1",
        "cv_guarded_threshold_source",
        "cv_official_threshold_f1",
        "cv_official_threshold_source",
        "holdout_mean_rank_ic",
        "holdout_top_10_lift_global",
        "holdout_top_10_forward_return_global",
        "cv_phase1_blockers",
        "holdout_blockers",
        "next_action",
        "research_track",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)

def _write_performance_gap_analysis(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "performance_gap_analysis.csv", index=False)
    (path / "performance_gap_analysis.md").write_text(_performance_gap_markdown(frame), encoding="utf-8")
    _write_json(path / "performance_gap_analysis.json", {"rows": frame.to_dict(orient="records")})
