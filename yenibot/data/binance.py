from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

KLINE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
]

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_to_milliseconds(interval: str) -> int:
    try:
        return INTERVAL_TO_MS[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported Binance interval: {interval}") from exc


def to_milliseconds(value: str | int | float | datetime | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return int(ts.timestamp() * 1000)


def klines_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert Binance full kline rows into a typed DataFrame."""

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    if any(len(row) < len(KLINE_COLUMNS) for row in rows):
        raise ValueError("Binance kline rows must contain the full 12-column schema")

    frame = pd.DataFrame([row[: len(KLINE_COLUMNS)] for row in rows], columns=KLINE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["num_trades"] = frame["num_trades"].astype("int64")
    return frame.sort_values("timestamp").reset_index(drop=True)


def download_full_klines(
    symbol: str,
    interval: str,
    start: str | int | float | datetime | pd.Timestamp,
    end: str | int | float | datetime | pd.Timestamp | None = None,
    *,
    base_url: str = "https://fapi.binance.com",
    limit: int = 1500,
    request_sleep_seconds: float = 0.15,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download full Binance USDT-M futures klines with microstructure columns."""

    if limit > 1500:
        raise ValueError("Binance futures kline limit cannot exceed 1500")
    interval_ms = interval_to_milliseconds(interval)
    start_ms = to_milliseconds(start)
    end_ms = to_milliseconds(end)
    if start_ms is None:
        raise ValueError("start is required")
    if end_ms is not None and end_ms <= start_ms:
        raise ValueError("end must be after start")

    http = session or requests.Session()
    url = f"{base_url.rstrip('/')}/fapi/v1/klines"
    rows: list[list[Any]] = []
    cursor = start_ms

    while True:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": limit,
        }
        if end_ms is not None:
            params["endTime"] = end_ms - 1

        response = http.get(url, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected Binance response: {batch}")
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance")
        cursor = next_cursor
        if end_ms is not None and cursor >= end_ms:
            break
        if len(batch) < limit:
            break
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    return klines_to_dataframe(rows)
