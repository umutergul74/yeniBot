from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from yenibot.data.binance import NUMERIC_COLUMNS
from yenibot.data.validation import validate_full_kline_frame
from yenibot.experiment.common import _hash_payload
from yenibot.experiment.configuration import _is_complete, _training_signature


@pytest.mark.parametrize("policy", ["error", "drop"])
@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_kline_validation_rejects_nonfinite_values(synthetic_klines, policy, column, value):
    frame = synthetic_klines(6)
    frame[column] = frame[column].astype(float)
    frame.loc[2, column] = value

    with pytest.raises(ValueError, match=f"{column} must contain only finite values"):
        validate_full_kline_frame(frame, "1h", zero_volume_policy=policy)


@pytest.mark.parametrize("policy", ["error", "drop"])
@pytest.mark.parametrize("column", ["timestamp", "close_time"])
def test_kline_validation_rejects_missing_times(synthetic_klines, policy, column):
    frame = synthetic_klines(6)
    frame.loc[2, column] = pd.NaT

    with pytest.raises(ValueError, match="must not be missing"):
        validate_full_kline_frame(frame, "1h", zero_volume_policy=policy)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_kline_validation_rejects_nonpositive_prices(synthetic_klines, column, value):
    frame = synthetic_klines(6)
    frame.loc[2, column] = value

    with pytest.raises(ValueError, match="OHLC prices must be positive"):
        validate_full_kline_frame(frame, "1h")


@pytest.mark.parametrize("column,value", [("high", 1.0), ("low", 1000.0)])
def test_kline_validation_rejects_inverted_price_bounds(synthetic_klines, column, value):
    frame = synthetic_klines(6)
    frame.loc[2, column] = value

    with pytest.raises(ValueError, match="low <= open/close <= high"):
        validate_full_kline_frame(frame, "1h")


def test_valid_kline_values_and_source_are_preserved(synthetic_klines):
    original = synthetic_klines(6)
    shuffled = original.iloc[::-1].copy()
    source_copy = shuffled.copy(deep=True)

    actual = validate_full_kline_frame(shuffled, "1h")

    pd.testing.assert_frame_equal(actual, original)
    pd.testing.assert_frame_equal(shuffled, source_copy)


@pytest.fixture
def signature_inputs():
    config = {
        "features": {"active_profile": "control", "profiles": {"control": {
            "include_patterns": ["model_feature"],
        }}},
        "labeling": {"max_holding_bars": 3},
        "hmm": {"features": ["hmm_feature"]},
    }
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "model_feature": [1.0, 2.0, 3.0],
        "hmm_feature": [4.0, 5.0, 6.0],
        "label": [0, 1, 0],
        "fwd_return_3h": [0.1, 0.2, 0.3],
        "diagnostic_only": [7.0, 8.0, 9.0],
    })
    return frame, config


def _signature(frame, config):
    return _training_signature(
        frame=frame, config=config, profile="control",
        feature_columns=["model_feature"], fold_ids=None, fold_scope="full",
    )


@pytest.mark.parametrize("column", ["hmm_feature", "fwd_return_3h"])
def test_training_signature_covers_all_consumed_values(signature_inputs, column):
    frame, config = signature_inputs
    before = _signature(frame, config)
    frame.loc[0, column] = 99.0

    assert _signature(frame, config)["frame_fingerprint"] != before["frame_fingerprint"]


def test_training_signature_rejects_forward_return_substitution(signature_inputs):
    frame, config = signature_inputs
    frame = frame.rename(columns={"fwd_return_3h": "fwd_return_10h"})
    with pytest.raises(ValueError, match="configured return target"):
        _signature(frame, config)


def test_training_signature_uses_profile_overridden_inputs(signature_inputs):
    frame, config = signature_inputs
    config["labeling"]["max_holding_bars"] = 10
    config["hmm"]["features"] = []
    config["features"]["profiles"]["control"]["config_overrides"] = {
        "labeling": {"max_holding_bars": 3}, "hmm": {"features": ["hmm_feature"]},
    }
    original_config = copy.deepcopy(config)
    before = _signature(frame, config)
    for column in ("hmm_feature", "fwd_return_3h"):
        changed = frame.copy()
        changed.loc[0, column] = 99.0
        assert _signature(changed, config)["frame_fingerprint"] != before["frame_fingerprint"]
    assert config == original_config


def test_training_signature_ignores_diagnostic_changes(signature_inputs):
    frame, config = signature_inputs
    before = _signature(frame, config)
    frame.loc[0, "diagnostic_only"] = 99.0
    config["validation"] = {"calibration_bins": 17}

    assert _signature(frame, config) == before


def test_old_training_signature_is_not_reused_or_rewritten(signature_inputs, tmp_path):
    frame, config = signature_inputs
    current = _signature(frame, config)
    previous = {**current, "signature_version": "profile_training_v2"}
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps({
        "completed": True, "signature_hash": _hash_payload(previous),
    }), encoding="utf-8")
    predictions_path = tmp_path / "predictions_all.parquet"
    pd.DataFrame({"probability": [0.5]}).to_parquet(predictions_path)
    before_manifest = manifest_path.read_bytes()
    before_predictions = predictions_path.read_bytes()

    assert current["signature_version"] == "profile_training_v3"
    assert not _is_complete(tmp_path, _hash_payload(current))
    assert manifest_path.read_bytes() == before_manifest
    assert predictions_path.read_bytes() == before_predictions
