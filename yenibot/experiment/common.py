"""Shared deterministic utilities and metric primitives for experiment workflows."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

__all__ = [
    '_cfg',
    '_set_cfg',
    '_json_ready',
    '_hash_payload',
    '_slug',
    '_table_markdown',
    '_holdout_policy_action',
    '_deep_update',
    '_read_json',
    '_write_json',
    '_float',
    '_optional_float',
    '_optional_gate_float',
    '_metric_or',
    '_rank_ic_for_frame',
    '_fmt_metric',
    '_first_frame_row',
    '_diagnostic_candidate_type',
    '_is_stability_scope',
    '_score_ks_statistic',
    '_mean_for_mask',
    '_clean_probability_inputs',
    '_safe_average_precision',
    '_numeric_mean',
]

def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
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

def _set_cfg(config: dict[str, Any], path: list[str], value: Any) -> None:
    current: dict[str, Any] = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value

def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value

def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")

def _table_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No rows were produced.")
        return "\n".join(lines)
    lines.append("| " + " | ".join(frame.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)

def _holdout_policy_action(
    *,
    frozen: dict[str, Any],
    observed_policy: dict[str, Any],
    frozen_selection: str,
    config: dict[str, Any] | None = None,
    holdout_boundary_passed: bool = True,
) -> str:
    if not holdout_boundary_passed:
        return "invalid_holdout_training_boundary_rerun_04"
    policy_status = str(_cfg(config or {}, ["experiments", "policy_review", "status"], "")).lower()
    if any(token in policy_status for token in ("failed", "invalidated", "retired")):
        return "retired_frozen_policy_keep_control_profile"
    frozen_consistent = bool(frozen.get("holdout_policy_consistency_pass", False))
    frozen_signal = bool(frozen.get("holdout_signal_pass", False))
    frozen_threshold = bool(frozen.get("holdout_threshold_pass", False))
    observed_consistent = bool(observed_policy.get("holdout_policy_consistency_pass", False))
    observed_name = str(observed_policy.get("candidate", ""))
    threshold_allowed = bool(_cfg(config or {}, ["experiments", "policy_review", "threshold_deployment_allowed"], False))
    if frozen_consistent and frozen_signal and frozen_threshold and threshold_allowed:
        return "review_frozen_threshold_and_score_policy"
    if frozen_consistent and frozen_signal:
        return "review_frozen_score_band_policy_only_no_threshold_deployment"
    if observed_consistent and observed_name and observed_name != frozen_selection:
        return "holdout_only_candidate_do_not_promote_without_future_oos"
    return "keep_control_profile"

def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")

def _float(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

def _optional_gate_float(gates: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = gates.get(key, default)
    if value is None:
        return None
    return float(value)

def _metric_or(row: dict[str, Any], key: str, fallback: float) -> float:
    value = _float(row, key, np.nan)
    if np.isnan(value):
        return fallback
    return value

def _rank_ic_for_frame(predictions: pd.DataFrame) -> float:
    if predictions.empty or "prob_long" not in predictions.columns or "forward_return" not in predictions.columns:
        return np.nan
    frame = predictions[["prob_long", "forward_return"]].copy()
    frame["prob_long"] = pd.to_numeric(frame["prob_long"], errors="coerce")
    frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
    frame = frame.dropna()
    if len(frame) < 3 or frame["prob_long"].nunique() < 2 or frame["forward_return"].nunique() < 2:
        return np.nan
    return float(frame["prob_long"].corr(frame["forward_return"], method="spearman"))

def _fmt_metric(value: Any, digits: int = 4) -> str:
    number = _optional_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"

def _first_frame_row(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    if frame.empty:
        return {}
    selected = frame.loc[mask] if mask is not None else frame
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()

def _diagnostic_candidate_type(fold_scope: str) -> str:
    return "blend" if str(fold_scope).startswith("blend_") else "profile"

def _is_stability_scope(fold_scope: str) -> bool:
    fold_scope = str(fold_scope)
    return fold_scope == "full" or fold_scope.startswith("blend_")

def _score_ks_statistic(positive_scores: pd.Series, negative_scores: pd.Series) -> float:
    pos = pd.to_numeric(positive_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    neg = pd.to_numeric(negative_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    values = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    pos_cdf = np.searchsorted(pos_sorted, values, side="right") / len(pos_sorted)
    neg_cdf = np.searchsorted(neg_sorted, values, side="right") / len(neg_sorted)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))

def _mean_for_mask(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan

def _clean_probability_inputs(labels: pd.Series, scores: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "label": pd.to_numeric(labels, errors="coerce"),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return frame
    frame["label"] = frame["label"].astype(int)
    frame["score"] = frame["score"].clip(1e-6, 1.0 - 1e-6)
    return frame

def _safe_average_precision(labels: pd.Series, scores: pd.Series) -> float:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty or frame["label"].nunique(dropna=True) < 2:
        return np.nan
    try:
        return float(average_precision_score(frame["label"], frame["score"]))
    except ValueError:
        return np.nan

def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan
