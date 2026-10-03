"""Strict hourly research contract with an explicit, audited empty-bar policy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_advisor_klines(frame: pd.DataFrame, interval: str, *,
                            zero_activity_policy: str = "error") -> tuple[pd.DataFrame, dict]:
    if zero_activity_policy not in {"error", "preserve_verified_empty"}:
        raise ValueError("Unsupported advisor zero_activity_policy")
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades",
               "taker_buy_base_vol", "taker_buy_quote_vol"]
    required = ["timestamp", *numeric]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing raw columns for {interval}: {missing}")
    part = frame.copy()
    part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
    if len(part) < 2 or not part.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=int(interval[:-1]))).all():
        raise ValueError(f"{interval}: expected unique, ordered, contiguous raw timestamps")
    for column in numeric:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    finite = np.isfinite(part[numeric].to_numpy()).all(axis=1)
    positive_price = (part[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_range = ((part.low <= part[["open", "close"]].min(axis=1)) &
                   (part.high >= part[["open", "close"]].max(axis=1)))
    nonnegative = (part[["volume", "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]] >= 0).all(axis=1)
    integer_trades = part.num_trades.eq(np.floor(part.num_trades))
    taker_valid = ((part.taker_buy_base_vol <= part.volume) &
                   (part.taker_buy_quote_vol <= part.quote_volume))
    no_activity = (part.volume == 0) | (part.num_trades == 0)
    all_activity_zero = part[["volume", "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]].eq(0).all(axis=1)
    # Never fabricate trades, prices or a synthetic filled row. Confirm that the
    # source's own empty bar is flat and agrees with the previous observed close.
    flat = part[["open", "high", "low"]].eq(part.close, axis=0).all(axis=1)
    unchanged = part.close.eq(part.close.shift(1))
    verified_empty = no_activity & all_activity_zero & flat & unchanged
    active = (part.volume > 0) & (part.num_trades > 0) & (part.quote_volume > 0)
    accepted_activity = active | (verified_empty if zero_activity_policy == "preserve_verified_empty" else False)
    valid = finite & positive_price & valid_range & nonnegative & integer_trades & taker_valid & accepted_activity
    audit = {"interval": interval, "policy": zero_activity_policy, "rows": len(part),
             "verified_empty_rows": int(verified_empty.sum()), "invalid_rows": int((~valid).sum()),
             "dropped_rows": 0, "filled_rows": 0,
             "verified_empty_timestamps": [timestamp.isoformat() for timestamp in part.loc[verified_empty, "timestamp"]]}
    if not valid.all():
        sample = part.loc[~valid, required].head(8).copy()
        sample["timestamp"] = sample.timestamp.astype(str)
        raise ValueError(f"{interval}: invalid raw bars under policy {zero_activity_policy!r}; "
                         f"invalid_count={int((~valid).sum())}; examples={sample.to_dict('records')}")
    return part, audit
