"""Causal validation-adaptive ensemble primitives.

The helpers in this module intentionally know nothing about Future-OOS rows.
They operate on one historical walk-forward validation window at a time and
return an equal-weight subset of already-fitted, causally eligible models.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from yenibot.experiment.common import _rank_ic_for_frame

__all__ = [
    "aggregate_fixed_model_predictions",
    "select_models_by_validation_lcb",
    "split_validation_predictions",
]


def split_validation_predictions(
    raw_predictions: pd.DataFrame,
    *,
    selector_rows: int,
    purge_rows: int,
    min_calibration_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Chronologically separate model selection from threshold calibration."""

    required = {"timestamp", "model_fold", "prob_long", "label", "forward_return"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError(f"Validation predictions are missing columns: {missing}")
    frame = raw_predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    timestamps = pd.Index(frame["timestamp"].drop_duplicates().sort_values())
    selector_count = int(selector_rows)
    purge_count = int(purge_rows)
    minimum_calibration = int(min_calibration_rows)
    if selector_count <= 0 or purge_count < 0 or minimum_calibration <= 0:
        raise ValueError("Validation split row counts must be positive")
    required_rows = selector_count + purge_count + minimum_calibration
    if len(timestamps) < required_rows:
        raise ValueError(
            "Validation window is too short for the preregistered split: "
            f"observed={len(timestamps)} required={required_rows}"
        )

    selector_end = selector_count
    calibration_start = selector_end + purge_count
    selector_timestamps = set(timestamps[:selector_end])
    calibration_timestamps = set(timestamps[calibration_start:])
    selector = frame.loc[frame["timestamp"].isin(selector_timestamps)].copy()
    calibration = frame.loc[frame["timestamp"].isin(calibration_timestamps)].copy()
    metadata = {
        "validation_timestamp_count": int(len(timestamps)),
        "selector_rows": selector_count,
        "purge_rows": purge_count,
        "calibration_rows": int(len(calibration_timestamps)),
        "selector_start": timestamps[0].isoformat(),
        "selector_end": timestamps[selector_end - 1].isoformat(),
        "calibration_start": timestamps[calibration_start].isoformat(),
        "calibration_end": timestamps[-1].isoformat(),
        "selection_and_calibration_disjoint": bool(
            selector_timestamps.isdisjoint(calibration_timestamps)
        ),
    }
    return selector, calibration, metadata


def _moving_block_indices(
    n_rows: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block = max(1, min(int(block_length), int(n_rows)))
    starts = rng.integers(
        0,
        max(1, int(n_rows) - block + 1),
        size=int(np.ceil(int(n_rows) / block)),
    )
    return np.concatenate(
        [np.arange(int(start), int(start) + block) for start in starts]
    )[: int(n_rows)]


def _bootstrap_rank_ic_interval(
    frame: pd.DataFrame,
    *,
    block_length: int,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    clean = frame[["prob_long", "forward_return"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    observed = _rank_ic_for_frame(clean)
    if len(clean) < 3 or int(repeats) <= 0:
        return {
            "rank_ic": float(observed),
            "rank_ic_ci_low": float(observed),
            "rank_ic_ci_high": float(observed),
        }
    rng = np.random.default_rng(int(random_seed))
    values: list[float] = []
    for _ in range(int(repeats)):
        sampled = clean.iloc[
            _moving_block_indices(
                len(clean),
                block_length=int(block_length),
                rng=rng,
            )
        ]
        value = _rank_ic_for_frame(sampled)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        low = high = np.nan
    else:
        alpha = (1.0 - float(confidence_level)) / 2.0
        low = float(np.quantile(values, alpha))
        high = float(np.quantile(values, 1.0 - alpha))
    return {
        "rank_ic": float(observed),
        "rank_ic_ci_low": low,
        "rank_ic_ci_high": high,
    }


def select_models_by_validation_lcb(
    selector_predictions: pd.DataFrame,
    *,
    target_fold: int,
    pool_recent_k: int,
    select_top_k: int,
    block_length: int,
    bootstrap_repeats: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[list[int], pd.DataFrame]:
    """Select recent experts by downside-adjusted validation Rank IC.

    The pool is fixed by recency before labels are inspected. Every expert is
    scored on the same historical validation selector window. Ties prefer the
    more recent model; selected experts are later aggregated at equal weight.
    """

    required = {"timestamp", "model_fold", "prob_long", "forward_return"}
    missing = sorted(required.difference(selector_predictions.columns))
    if missing:
        raise ValueError(f"Selector predictions are missing columns: {missing}")
    frame = selector_predictions.copy()
    frame["model_fold"] = pd.to_numeric(
        frame["model_fold"], errors="raise"
    ).astype(int)
    eligible = sorted(
        int(item)
        for item in frame["model_fold"].unique().tolist()
        if int(item) <= int(target_fold)
    )
    if not eligible:
        raise ValueError(f"No models are eligible for target fold {target_fold}")
    pool_size = max(1, int(pool_recent_k))
    pool = eligible[-pool_size:]
    selected_count = min(max(1, int(select_top_k)), len(pool))
    rows: list[dict[str, Any]] = []
    for model_fold in pool:
        part = frame.loc[frame["model_fold"].eq(model_fold)].sort_values(
            "timestamp"
        )
        interval = _bootstrap_rank_ic_interval(
            part,
            block_length=int(block_length),
            repeats=int(bootstrap_repeats),
            confidence_level=float(confidence_level),
            random_seed=(
                int(random_seed) + int(target_fold) * 1009 + int(model_fold) * 9176
            ),
        )
        rows.append(
            {
                "target_fold": int(target_fold),
                "model_fold": int(model_fold),
                "pool_recent_k": pool_size,
                "select_top_k": int(select_top_k),
                "selector_rows": int(part["timestamp"].nunique()),
                "validation_rank_ic": interval["rank_ic"],
                "validation_rank_ic_ci_low": interval["rank_ic_ci_low"],
                "validation_rank_ic_ci_high": interval["rank_ic_ci_high"],
                "bootstrap_block_length": int(block_length),
                "bootstrap_repeats": int(bootstrap_repeats),
                "confidence_level": float(confidence_level),
                "selection_data_role": "historical_validation_selector_only",
            }
        )
    audit = pd.DataFrame(rows)
    audit = audit.sort_values(
        ["validation_rank_ic_ci_low", "validation_rank_ic", "model_fold"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected = [int(item) for item in audit.head(selected_count)["model_fold"]]
    audit["selected"] = audit["model_fold"].isin(selected)
    audit["selection_rank"] = np.arange(1, len(audit) + 1)
    return selected, audit


def aggregate_fixed_model_predictions(
    raw_predictions: pd.DataFrame,
    *,
    selected_model_folds: Iterable[int],
    policy_name: str,
) -> pd.DataFrame:
    """Equal-weight an explicitly selected causal model subset."""

    selected = sorted({int(item) for item in selected_model_folds})
    if not selected:
        raise ValueError("At least one selected model fold is required")
    frame = raw_predictions.copy()
    frame["model_fold"] = pd.to_numeric(
        frame["model_fold"], errors="raise"
    ).astype(int)
    frame = frame.loc[frame["model_fold"].isin(selected)].copy()
    observed = sorted(frame["model_fold"].unique().tolist())
    if observed != selected:
        raise ValueError(
            "Selected model predictions are incomplete: "
            f"selected={selected} observed={observed}"
        )
    first_columns = [
        column
        for column in ("label", "forward_return", "tb_return", "hit_type")
        if column in frame.columns
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        column: (column, "first") for column in first_columns
    }
    aggregations["prob_long"] = ("prob_long", "mean")
    aggregations["model_count"] = ("model_fold", "nunique")
    out = frame.groupby("timestamp", as_index=False).agg(**aggregations)
    if out.empty or not out["model_count"].eq(len(selected)).all():
        raise ValueError("Every timestamp must contain every selected model")
    out["policy"] = str(policy_name)
    out["selected_model_folds"] = ",".join(str(item) for item in selected)
    return out.sort_values("timestamp").reset_index(drop=True)
