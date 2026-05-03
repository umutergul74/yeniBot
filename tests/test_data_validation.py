from __future__ import annotations

import zipfile
from io import BytesIO

import pandas as pd
import pytest
import requests

from yenibot.data import download_full_klines
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


def test_full_kline_validation_can_drop_zero_volume_rows(synthetic_klines) -> None:
    frame = synthetic_klines(6, "1h")
    frame.loc[2, ["volume", "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]] = 0

    with pytest.raises(ValueError, match="Zero or negative volume/trade activity"):
        validate_full_kline_frame(frame, "1h")

    validated = validate_full_kline_frame(frame, "1h", zero_volume_policy="drop")

    assert len(validated) == 5
    assert validated.attrs["dropped_zero_volume_rows"] == 1
    assert pd.Timestamp("2022-01-01 02:00", tz="UTC") not in set(validated["timestamp"])


def test_download_full_klines_falls_back_to_vision_on_451() -> None:
    session = _FakeSession()
    df = download_full_klines(
        "BTCUSDT",
        "1h",
        "2022-01-01",
        "2022-01-01 03:00",
        data_source="auto",
        session=session,
    )

    assert len(df) == 3
    assert df["taker_buy_base_vol"].sum() > 0
    assert any("data.binance.vision" in call for call in session.calls)


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", payload: object | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, **kwargs) -> _FakeResponse:
        self.calls.append(url)
        if "fapi/v1/klines" in url:
            return _FakeResponse(451)
        if "monthly/klines" in url:
            return _FakeResponse(200, _vision_zip_bytes())
        return _FakeResponse(404)


def _vision_zip_bytes() -> bytes:
    rows = []
    for i in range(3):
        open_time = 1640995200000 + i * 3_600_000
        close_time = open_time + 3_600_000 - 1
        rows.append(
            [
                open_time,
                "100.0",
                "101.0",
                "99.0",
                "100.5",
                "10.0",
                close_time,
                "1005.0",
                20,
                "5.5",
                "552.75",
                "0",
            ]
        )
    csv = "\n".join(",".join(map(str, row)) for row in rows)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1h-2022-01.csv", csv)
    return buffer.getvalue()
