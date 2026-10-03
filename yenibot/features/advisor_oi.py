"""Explicit two-channel open-interest input, with backward matching and audit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_oi_source(frame: pd.DataFrame, *, invalid_policy: str = "error") -> pd.DataFrame:
    """Keep unavailable measurements in place when preserving an archive.

    Preserved zero/negative/non-finite measurements are never usable features;
    retaining their timestamp prevents asof from hiding them with an older row.
    """
    if invalid_policy not in {"error", "preserve_unavailable"}:
        raise ValueError("Unknown open-interest invalid-value policy")
    columns = ["timestamp", "sum_open_interest", "sum_open_interest_value"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Open-interest source missing columns: {missing}")
    source = frame[columns].copy()
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
    for column in columns[1:]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.timestamp.isna().any():
        raise ValueError("Open-interest source has invalid timestamps")
    invalid = (~np.isfinite(source[columns[1:]].to_numpy())).any(axis=1) | (source[columns[1:]] <= 0).any(axis=1)
    if invalid.any() and invalid_policy == "error":
        examples = source.loc[invalid, columns].head(8).astype(str).to_dict("records")
        raise ValueError(f"Open-interest source has non-finite/non-positive values: {examples}")
    source = source.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    if source.timestamp.duplicated().any():
        raise ValueError("Conflicting open-interest snapshots at the same timestamp")
    if len(source) < 2:
        raise ValueError("Need at least two open-interest snapshots")
    return source


def audit_oi_source(source: pd.DataFrame) -> dict:
    columns = ["sum_open_interest", "sum_open_interest_value"]
    invalid = (~np.isfinite(source[columns].to_numpy())).any(axis=1) | (source[columns] <= 0).any(axis=1)
    return {"policy": "preserve_unavailable", "rows": len(source),
            "invalid_source_rows": int(invalid.sum()), "dropped_rows": 0,
            "invalid_timestamps": source.loc[invalid, "timestamp"].astype(str).tolist(),
            "invalid_examples": source.loc[invalid, ["timestamp"]+columns].head(20).astype(str).to_dict("records")}


def append_oi_features(frame: pd.DataFrame, metrics: pd.DataFrame, *, tolerance_minutes: int = 90,
                       max_diff_gap_minutes: int = 15, min_coverage: float = .99) -> tuple[pd.DataFrame, dict]:
    if not 0 < min_coverage <= 1 or min(tolerance_minutes, max_diff_gap_minutes) <= 0:
        raise ValueError("Invalid open-interest coverage/timing policy")
    source = normalize_oi_source(metrics, invalid_policy="preserve_unavailable")
    quality = audit_oi_source(source)
    valid = np.isfinite(source[["sum_open_interest", "sum_open_interest_value"]].to_numpy()).all(axis=1)
    valid &= (source[["sum_open_interest", "sum_open_interest_value"]] > 0).all(axis=1)
    # Same snapshot log-difference definitions as compute_futures_metrics_features.
    # A long archive gap is not misrepresented as an ordinary snapshot change.
    usable = source.timestamp.diff().le(pd.Timedelta(minutes=max_diff_gap_minutes))
    # No log(0), fabricated replacement level or difference across an invalid row.
    source["fut_oi_log_return"] = np.log(source.sum_open_interest.where(valid)).diff().where(usable)
    source["fut_oi_value_log_return"] = np.log(source.sum_open_interest_value.where(valid)).diff().where(usable)
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
             "future_matches": 0, "base_columns_unchanged": True, "source_quality": quality}
    return matched, audit
