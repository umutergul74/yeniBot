"""Automatic Binance funding history, with explicit completeness checks.

The public website endpoint is not a versioned API contract. Schema changes or
inaccessible sources must fail closed, never produce a stale successful dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from yenibot.data.binance import (
    BINANCE_VISION_BASE_URL, _download_vision_csv_zip, _month_iter,
    download_funding_rates, funding_rates_to_dataframe, to_milliseconds,
)

WEBSITE_URL = "https://www.binance.com/bapi/futures/v1/public/future/common/get-funding-rate-history"


def validate_funding_history(frame, symbol, start, end):
    """Require the requested history, allowing settlement timestamp millisecond jitter."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if end <= start:
        raise ValueError("Funding end must be after start")
    if frame.empty or not frame["symbol"].eq(symbol).all():
        raise ValueError("Empty funding history or unexpected symbol")
    times = frame["timestamp"]
    if (times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing
            or (times < start).any() or (times >= end).any()
            or not np.isfinite(frame["funding_rate"].to_numpy(dtype=float)).all()):
        raise ValueError("Invalid funding timestamps or rates")
    # Binance BTCUSDT regular settlements are at most eight hours apart.
    allowance = pd.Timedelta(hours=8, seconds=1)
    if (times.iloc[0] - start >= allowance or end - times.iloc[-1] > allowance
            or (times.diff().dropna() > allowance).any()):
        raise ValueError("Funding history incomplete for requested cutoff; retry when published")
    return frame


def _website_history(http, symbol, start_ms, end_ms):
    rows = {}
    oldest = None
    for page in range(1, 1001):
        response = http.post(WEBSITE_URL, json={"symbol": symbol, "page": page, "rows": 100}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if (not isinstance(payload, dict) or payload.get("success") is not True
                or payload.get("code") != "000000" or not isinstance(payload.get("data"), list)):
            raise ValueError("Unexpected Binance website funding response")
        batch = payload["data"]
        if not batch:
            break
        times = [int(row["calcTime"]) for row in batch]
        if times != sorted(times, reverse=True) or (oldest is not None and min(times) >= oldest):
            raise ValueError("Binance website funding pagination did not advance")
        for row, timestamp in zip(batch, times):
            if row["symbol"] != symbol:
                raise ValueError("Unexpected funding symbol")
            if row.get("rateType", "Regular") != "Regular":
                raise ValueError("Unsupported Binance funding rate type")
            if start_ms <= timestamp < end_ms:
                normalized = {"symbol": symbol, "fundingTime": timestamp,
                              "fundingRate": row["lastFundingRate"], "markPrice": row.get("markPrice")}
                if timestamp in rows and rows[timestamp] != normalized:
                    raise ValueError("Conflicting funding records across website pages")
                rows[timestamp] = normalized
        oldest = min(times)
        if oldest <= start_ms:
            break
    else:
        raise ValueError("Binance funding pagination exceeded its safety limit")
    return list(rows.values())


def download_funding_history(symbol, start, end, *, base_url="https://fapi.binance.com",
                             vision_base_url=BINANCE_VISION_BASE_URL, session=None):
    """REST first; on access/network failure use monthly archives plus website history."""
    http = session or requests.Session()
    start_ms, end_ms = to_milliseconds(start), to_milliseconds(end)
    if start_ms is None or end_ms is None or end_ms <= start_ms:
        raise ValueError("Explicit funding start and end are required")
    try:
        result = download_funding_rates(symbol, start, end, base_url=base_url, session=http)
    except requests.RequestException as exc:
        print(f"Funding REST unavailable ({type(exc).__name__}); using Binance public history")
    else:
        validate_funding_history(result, symbol, start, end)
        result.attrs["funding_sources"] = [base_url.rstrip("/") + "/fapi/v1/fundingRate"]
        return result

    rows, missing, sources = [], [], []
    for year, month in _month_iter(start_ms, end_ms):
        begin = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        finish = begin + pd.offsets.MonthBegin(1)
        lo = max(start_ms, int(begin.timestamp() * 1000))
        hi = min(end_ms, int(finish.timestamp() * 1000))
        url = (f"{vision_base_url.rstrip('/')}/data/futures/um/monthly/fundingRate/"
               f"{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip")
        archive = _download_vision_csv_zip(url, session=http)
        if archive is None:
            missing.append((lo, hi))
            continue
        if not {"calc_time", "last_funding_rate", "funding_interval_hours"} <= set(archive.columns):
            raise ValueError("Unexpected Binance funding archive schema")
        sources.append(url)
        for row in archive.to_dict("records"):
            timestamp = int(row["calc_time"])
            if lo <= timestamp < hi:
                rows.append({"symbol": symbol, "fundingTime": timestamp,
                             "fundingRate": row["last_funding_rate"]})
    if missing:
        recent = _website_history(http, symbol, min(lo for lo, _ in missing), max(hi for _, hi in missing))
        rows.extend(row for row in recent if any(lo <= row["fundingTime"] < hi for lo, hi in missing))
        sources.append(WEBSITE_URL)
    times = [row["fundingTime"] for row in rows]
    if len(times) != len(set(times)):
        raise ValueError("Duplicate Binance funding archive timestamps")
    result = funding_rates_to_dataframe(rows)
    validate_funding_history(result, symbol, start, end)
    result.attrs["funding_sources"] = sources
    return result
