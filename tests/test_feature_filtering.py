from __future__ import annotations

import pytest

from yenibot.features import filter_feature_columns


def test_filter_feature_columns_uses_configured_names_and_patterns() -> None:
    columns = ["true_cvd_zscore", "realized_vol_14", "4h_log_return", "4h_true_cvd_zscore"]
    config = {
        "features": {
            "exclude_columns": ["realized_vol_14"],
            "exclude_patterns": ["4h_log_return"],
        }
    }

    assert filter_feature_columns(columns, config) == ["true_cvd_zscore", "4h_true_cvd_zscore"]


def test_filter_feature_columns_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="removed every feature"):
        filter_feature_columns(["a"], {"features": {"exclude_patterns": ["*"]}})


def test_filter_feature_columns_applies_stationarity_policy() -> None:
    columns = [
        "close_denoised",
        "close_denoised_log_return",
        "4h_atr_14",
        "4h_atr_14_pct",
        "true_cvd_delta",
        "true_cvd_delta_norm",
    ]
    config = {
        "features": {
            "stationarity": {
                "exclude_nonstationary": True,
                "exclude_patterns": ["*close_denoised", "*atr_14", "*true_cvd_delta"],
            }
        }
    }

    assert filter_feature_columns(columns, config) == [
        "close_denoised_log_return",
        "4h_atr_14_pct",
        "true_cvd_delta_norm",
    ]
