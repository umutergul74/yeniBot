from __future__ import annotations

import pandas as pd

from yenibot.data.binance import KLINE_COLUMNS, interval_to_milliseconds


def validate_full_kline_frame(
    frame: pd.DataFrame,
    interval: str,
    *,
    max_gap_multiplier: int = 2,
    require_taker_nonzero: bool = True,
) -> pd.DataFrame:
    """Validate full Binance kline data and return a sorted copy."""

    missing = [column for column in KLINE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Binance full-kline columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Kline frame is empty")

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        dupes = df.loc[df["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Duplicate kline timestamps detected: {dupes}")

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

    if (df["volume"] <= 0).any():
        first = df.loc[df["volume"] <= 0, "timestamp"].iloc[0]
        raise ValueError(f"Zero or negative volume detected at {first}")
    if (df["num_trades"] <= 0).any():
        first = df.loc[df["num_trades"] <= 0, "timestamp"].iloc[0]
        raise ValueError(f"Zero or negative num_trades detected at {first}")
    if require_taker_nonzero and df["taker_buy_base_vol"].abs().sum() == 0:
        raise ValueError("taker_buy_base_vol is all zero; this is not usable full-kline data")
    if (df["taker_buy_base_vol"] < 0).any() or (df["taker_buy_base_vol"] > df["volume"]).any():
        raise ValueError("taker_buy_base_vol must be within [0, volume]")
    if (df["taker_buy_quote_vol"] < 0).any() or (df["taker_buy_quote_vol"] > df["quote_volume"]).any():
        raise ValueError("taker_buy_quote_vol must be within [0, quote_volume]")

    return df
