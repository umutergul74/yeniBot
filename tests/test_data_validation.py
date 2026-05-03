from __future__ import annotations

import pandas as pd
import pytest

from yenibot.data.validation import validate_full_kline_frame


def test_full_kline_validation_rejects_ohlcv_only() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 11, 12],
        }
    )
    with pytest.raises(ValueError, match="Missing Binance full-kline columns"):
        validate_full_kline_frame(frame, "1h")


def test_full_kline_validation_accepts_microstructure_columns(synthetic_klines) -> None:
    frame = synthetic_klines(12, "1h")
    validated = validate_full_kline_frame(frame, "1h")
    assert len(validated) == 12
    assert validated["taker_buy_base_vol"].sum() > 0
