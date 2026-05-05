from __future__ import annotations

import copy

import pandas as pd
import pytest

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
    pytest.importorskip("pywt")
    series = pd.Series(range(320), dtype=float)
    extended = pd.Series(range(380), dtype=float)
    base = causal_wavelet_denoise(series, window=64)
    future = causal_wavelet_denoise(extended, window=64)
    pd.testing.assert_series_equal(base.dropna(), future.iloc[: len(base)].dropna(), check_names=False)


def test_stationary_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(80, "1h")
    htf = synthetic_klines(24, "4h")
    extended_primary = synthetic_klines(96, "1h")
    extended_htf = synthetic_klines(28, "4h")

    base = build_feature_matrix(primary, htf, tiny_config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, tiny_config).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "volume_log_zscore",
        "true_cvd_delta_norm",
        "cvd_cumulative_rate_norm",
        "vol_per_trade_log_zscore",
        "atr_14_pct",
        "4h_true_cvd_delta_norm",
        "4h_cvd_cumulative_rate_norm",
        "4h_atr_14_pct",
    ]

    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)


def test_structure_stability_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["structure_stability"] = {
        "enabled": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "source_columns": [
            "log_return",
            "realized_vol_14",
            "gk_vol_14",
            "atr_14_pct",
            "adx_14",
            "vwap_dist_atr",
            "volume_log_zscore",
        ],
    }
    primary = synthetic_klines(112, "1h")
    htf = synthetic_klines(34, "4h")
    extended_primary = synthetic_klines(128, "1h")
    extended_htf = synthetic_klines(38, "4h")

    base = build_feature_matrix(primary, htf, config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, config).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "log_return_stable_zscore",
        "gk_vol_14_stable_rank",
        "atr_14_pct_stable_zscore",
        "vwap_dist_atr_stable_rank",
        "volume_log_zscore_stable_zscore",
        "4h_gk_vol_14_stable_rank",
        "4h_volume_log_zscore_stable_zscore",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)


def test_inactive_stable_features_do_not_change_feature_matrix_rows(synthetic_klines, tiny_config) -> None:
    base_config = copy.deepcopy(tiny_config)
    base_config["features"]["profiles"] = {
        "baseline": {
            "include_patterns": [
                "*log_return",
                "*gk_vol_14",
                "*adx_14",
                "*true_cvd_zscore",
                "*vwap_dist_atr",
                "*atr_14_pct",
            ],
            "exclude_patterns": ["*_stable_*"],
        }
    }
    base_config["features"]["active_profile"] = "baseline"
    base_config["labeling"]["atr_column"] = "atr_14"
    stable_config = copy.deepcopy(base_config)
    stable_config["features"]["structure_stability"] = {
        "enabled": True,
        "stable_window": 24,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "source_columns": ["gk_vol_14", "atr_14_pct", "vwap_dist_atr"],
    }
    primary = synthetic_klines(96, "1h")
    htf = synthetic_klines(30, "4h")

    base = build_feature_matrix(primary, htf, base_config)
    stable = build_feature_matrix(primary, htf, stable_config)

    assert base.feature_columns == stable.feature_columns
    assert len(base.frame) == len(stable.frame)
    pd.testing.assert_series_equal(base.frame["timestamp"], stable.frame["timestamp"])
    assert stable.frame["gk_vol_14_stable_zscore"].isna().any()
    assert not stable.frame[stable.feature_columns].isna().any().any()


def test_order_flow_v2_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["order_flow_v2"] = {
        "enabled": True,
        "ratio_zscore_window": 4,
        "ratio_delta_periods": 1,
        "imbalance_slope_window": 4,
        "pressure_windows": [3, 4],
        "efficiency_epsilon": 0.001,
        "stable_only": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
    }
    primary = synthetic_klines(96, "1h")
    htf = synthetic_klines(30, "4h")
    extended_primary = synthetic_klines(112, "1h")
    extended_htf = synthetic_klines(34, "4h")

    base = build_feature_matrix(primary, htf, config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, config).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "taker_imbalance",
        "taker_buy_ratio_zscore",
        "taker_imbalance_slope",
        "signed_large_trade_pressure",
        "signed_large_trade_pressure_stable_zscore",
        "signed_large_trade_pressure_stable_rank",
        "orderflow_efficiency",
        "absorption_pressure_3",
        "absorption_pressure_3_stable_rank",
        "cvd_price_divergence_4",
        "cvd_price_divergence_4_stable_zscore",
        "4h_taker_imbalance",
        "4h_cvd_pressure_3",
        "4h_cvd_pressure_3_stable_rank",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)

    assert "signed_large_trade_pressure" not in build_feature_matrix(primary, htf, config).feature_columns
    assert "signed_large_trade_pressure_stable_zscore" in build_feature_matrix(primary, htf, config).feature_columns
