from __future__ import annotations

import numpy as np
import pandas as pd

from yenibot.data.binance import KLINE_COLUMNS, NUMERIC_COLUMNS, interval_to_milliseconds


def validate_full_kline_frame(
    frame: pd.DataFrame,
    interval: str,
    *,
    max_gap_multiplier: int = 2,
    require_taker_nonzero: bool = True,
    zero_volume_policy: str = "error",
) -> pd.DataFrame:
    """Validate full Binance kline data and return a sorted copy."""

    if zero_volume_policy not in {"error", "drop"}:
        raise ValueError("zero_volume_policy must be one of: error, drop")
    missing = [column for column in KLINE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Binance full-kline columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Kline frame is empty")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df[["timestamp", "close_time"]].isna().any().any():
        raise ValueError("Kline timestamps and close times must not be missing")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"Kline column {column} must contain only finite values")
    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("Kline OHLC prices must be positive")
    if (
        (df["low"] > prices[["open", "close"]].min(axis=1))
        | (df["high"] < prices[["open", "close"]].max(axis=1))
    ).any():
        raise ValueError("Kline OHLC prices must satisfy low <= open/close <= high")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        dupes = df.loc[df["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Duplicate kline timestamps detected: {dupes}")

    bad_activity = (df["volume"] <= 0) | (df["num_trades"] <= 0)
    dropped_zero_volume_rows = int(bad_activity.sum())
    if dropped_zero_volume_rows:
        first = df.loc[bad_activity, "timestamp"].iloc[0]
        if zero_volume_policy == "error":
            raise ValueError(f"Zero or negative volume/trade activity detected at {first}")
        df = df.loc[~bad_activity].reset_index(drop=True)
        if df.empty:
            raise ValueError("All kline rows were removed by zero_volume_policy=drop")

    expected = pd.Timedelta(milliseconds=interval_to_milliseconds(interval))
    max_allowed_gap = expected * max_gap_multiplier
    gaps = df["timestamp"].diff().dropna()
    bad_gaps = gaps[gaps > max_allowed_gap]
    if not bad_gaps.empty:
        first_idx = bad_gaps.index[0]
        raise ValueError(
            "Kline gap exceeds allowed threshold: "
            f"{df.loc[first_idx - 1, 'timestamp']} -> {df.loc[first_idx, 'timestamp']} "
            f"({bad_gaps.iloc[0]})"
        )

    max_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)
    df.attrs["gap_count_gt_expected"] = int((gaps > expected).sum())
    df.attrs["max_gap"] = str(max_gap)
    df.attrs["max_gap_seconds"] = float(max_gap.total_seconds())
    if require_taker_nonzero and df["taker_buy_base_vol"].abs().sum() == 0:
        raise ValueError("taker_buy_base_vol is all zero; this is not usable full-kline data")
    if (df["taker_buy_base_vol"] < 0).any() or (df["taker_buy_base_vol"] > df["volume"]).any():
        raise ValueError("taker_buy_base_vol must be within [0, volume]")
    if (df["taker_buy_quote_vol"] < 0).any() or (df["taker_buy_quote_vol"] > df["quote_volume"]).any():
        raise ValueError("taker_buy_quote_vol must be within [0, quote_volume]")

    df.attrs["dropped_zero_volume_rows"] = dropped_zero_volume_rows
    return df
