"""Explicit two-channel open-interest input, with backward matching and audit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_oi_source(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", "sum_open_interest", "sum_open_interest_value"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Open-interest source missing columns: {missing}")
    source = frame[columns].copy()
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
    for column in columns[1:]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.timestamp.isna().any() or not np.isfinite(source[columns[1:]].to_numpy()).all() or (source[columns[1:]] <= 0).any().any():
        raise ValueError("Open-interest source has non-finite/non-positive values")
    source = source.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    if source.timestamp.duplicated().any():
        raise ValueError("Conflicting open-interest snapshots at the same timestamp")
    if len(source) < 2:
        raise ValueError("Need at least two open-interest snapshots")
    return source


def append_oi_features(frame: pd.DataFrame, metrics: pd.DataFrame, *, tolerance_minutes: int = 90,
                       max_diff_gap_minutes: int = 15, min_coverage: float = .99) -> tuple[pd.DataFrame, dict]:
    if not 0 < min_coverage <= 1 or min(tolerance_minutes, max_diff_gap_minutes) <= 0:
        raise ValueError("Invalid open-interest coverage/timing policy")
    source = normalize_oi_source(metrics)
    # Same snapshot log-difference definitions as compute_futures_metrics_features.
    # A long archive gap is not misrepresented as an ordinary snapshot change.
    usable = source.timestamp.diff().le(pd.Timedelta(minutes=max_diff_gap_minutes))
    source["fut_oi_log_return"] = np.log(source.sum_open_interest).diff().where(usable)
    source["fut_oi_value_log_return"] = np.log(source.sum_open_interest_value).diff().where(usable)
    source = source.rename(columns={"timestamp": "oi_source_timestamp"})
    matched = pd.merge_asof(frame, source[["oi_source_timestamp", "fut_oi_log_return", "fut_oi_value_log_return"]],
                            left_on="timestamp", right_on="oi_source_timestamp", direction="backward",
                            tolerance=pd.Timedelta(minutes=tolerance_minutes))
    columns = ["fut_oi_log_return", "fut_oi_value_log_return"]
    unavailable = matched[columns].isna().any(axis=1)
    coverage = float(1-unavailable.mean())
    if coverage < min_coverage:
        raise ValueError(f"Open-interest coverage {coverage:.4%} below required {min_coverage:.2%}; "
                         f"missing examples: {matched.loc[unavailable, 'timestamp'].head(8).astype(str).tolist()}")
    present = matched.oi_source_timestamp.notna()
    if (matched.loc[present, "oi_source_timestamp"] > matched.loc[present, "timestamp"]).any():
        raise ValueError("Future open-interest snapshot used")
    # Explicitly audited neutral fallback, consistent with the existing context
    # builder. Entire missing source or poor coverage cannot pass this policy.
    matched.loc[unavailable, columns] = 0.0
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), matched[frame.columns].reset_index(drop=True))
    audit = {"source_rows": len(source), "coverage": coverage, "minimum_coverage": min_coverage,
             "tolerance_minutes": tolerance_minutes, "max_diff_gap_minutes": max_diff_gap_minutes,
             "neutral_filled_rows": int(unavailable.sum()), "long_source_gaps": int((~usable.iloc[1:]).sum()),
             "neutral_timestamps": matched.loc[unavailable, "timestamp"].astype(str).tolist(),
             "future_matches": 0, "base_columns_unchanged": True}
    return matched, audit
