from __future__ import annotations

import pandas as pd

from yenibot.features import build_feature_matrix
from yenibot.features.wavelet import causal_wavelet_denoise


def test_4h_alignment_delays_bar_until_complete(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(36, "1h")
    htf = synthetic_klines(10, "4h")
    result = build_feature_matrix(primary, htf, tiny_config)
    frame = result.frame

    row_23 = frame.loc[frame["timestamp"] == pd.Timestamp("2022-01-01 23:00", tz="UTC")].iloc[0]
    row_24 = frame.loc[frame["timestamp"] == pd.Timestamp("2022-01-02 00:00", tz="UTC")].iloc[0]

    assert row_23["4h_source_timestamp"] == pd.Timestamp("2022-01-01 16:00", tz="UTC")
    assert row_24["4h_source_timestamp"] == pd.Timestamp("2022-01-01 20:00", tz="UTC")
    assert row_24["4h_available_timestamp"] == pd.Timestamp("2022-01-02 00:00", tz="UTC")


def test_causal_wavelet_value_unchanged_when_future_appended() -> None:
    series = pd.Series(range(320), dtype=float)
    extended = pd.Series(range(380), dtype=float)
    base = causal_wavelet_denoise(series, window=64)
    future = causal_wavelet_denoise(extended, window=64)
    pd.testing.assert_series_equal(base.dropna(), future.iloc[: len(base)].dropna(), check_names=False)
