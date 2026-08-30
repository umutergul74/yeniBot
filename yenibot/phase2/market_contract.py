"""Explicit UTC bar intervals and fail-closed execution inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import Phase2StrategyContract


def normalize_execution_inputs(bars, signals, contract: Phase2StrategyContract):
    bars, signals = bars.copy(), signals.copy()
    interval = pd.Timedelta(hours=contract.expected_bar_interval_hours)
    legacy = "bar_open_time" not in bars.columns
    if legacy:
        # v1 bundles mislabeled Binance OPEN timestamps as bar_close_time and
        # decision_time. Convert both together, preserving the next-row fill.
        bars["bar_open_time"] = pd.to_datetime(bars[contract.bar_time_column], utc=True)
        signals[contract.decision_time_column] = (
            pd.to_datetime(signals[contract.decision_time_column], utc=True) + interval
        )
    else:
        bars["bar_open_time"] = pd.to_datetime(bars["bar_open_time"], utc=True)
        declared = pd.to_datetime(bars[contract.bar_time_column], utc=True)
        expected = bars["bar_open_time"] + interval
        if not (
            declared.eq(expected) | declared.eq(expected - pd.Timedelta(milliseconds=1))
        ).all():
            raise ValueError(
                "Declared bar close does not match the open/interval contract"
            )
        signals[contract.decision_time_column] = pd.to_datetime(
            signals[contract.decision_time_column], utc=True
        )
    bars[contract.bar_time_column] = bars["bar_open_time"] + interval
    numeric = ["open", "high", "low", "close", contract.atr_column]
    for column in numeric:
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    if bars["bar_open_time"].isna().any() or bars["bar_open_time"].duplicated().any():
        raise ValueError("Execution bars require finite, unique timestamps")
    if not np.isfinite(bars[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Execution bars contain non-finite price/ATR")
    prices = bars[["open", "high", "low", "close"]]
    if (
        (prices <= 0).any().any()
        or (bars["high"] < prices.max(axis=1)).any()
        or (bars["low"] > prices.min(axis=1)).any()
    ):
        raise ValueError("Invalid execution OHLC geometry")
    score = pd.to_numeric(signals[contract.score_column], errors="raise")
    if not np.isfinite(score).all() or not score.between(0, 1).all():
        raise ValueError("Signal scores must be finite and within [0, 1]")
    signals[contract.score_column] = score
    if (
        signals[contract.decision_time_column].isna().any()
        or signals[contract.decision_time_column].duplicated().any()
    ):
        raise ValueError("Signals require finite, unique decision timestamps")
    return bars, signals, legacy
