import numpy as np
import pandas as pd
import pytest

from yenibot.features.builder import filter_feature_columns, select_feature_columns
from yenibot.training.trainer import _assert_training_inputs_available, _make_dataset, train_one_fold
from yenibot.training.walk_forward import PurgedWalkForwardCV


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_forward_targets_never_enter_auto_or_explicit_features(horizon):
    name = f"fwd_return_{horizon}h"
    frame = pd.DataFrame({"signal": [1., 2.], name: [.1, .2], "label": [0, 1], "forward_return": [.1, .2]})
    assert select_feature_columns(frame) == ["signal"]
    with pytest.raises(ValueError, match="Target columns"):
        filter_feature_columns(["signal", name], {})
    config = {"hmm": {"features": [name]}, "labeling": {"max_holding_bars": horizon}}
    with pytest.raises(ValueError, match="model/HMM"):
        _assert_training_inputs_available(frame, ["signal"], config)


def test_missing_horizon_is_rejected_before_any_fit():
    frame = pd.DataFrame({"signal": [1., 2.], "label": [0, 1], "fwd_return_10h": [.1, .2]})
    config = {"model": {"seq_len": 1}, "labeling": {"max_holding_bars": 24}}
    with pytest.raises(ValueError, match="fwd_return_24h"):
        _make_dataset(frame, ["signal"], config)
    with pytest.raises(ValueError, match="fwd_return_24h"):
        train_one_fold(frame, None, ["signal"], config, device="cpu")
    frame["fwd_return_24h"] = [.3, .4]
    dataset = _make_dataset(frame, ["signal"], config)
    np.testing.assert_allclose(dataset.forward_returns, [.3, .4])


PARAMS = dict(train_bars=8, val_bars=4, test_bars=4, step_bars=4, purge_bars=2, embargo_bars=2)


@pytest.mark.parametrize("key", ["train_bars", "val_bars", "test_bars", "step_bars"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, float("nan")])
def test_invalid_windows_fail_before_split(key, bad):
    with pytest.raises(ValueError, match=key):
        PurgedWalkForwardCV(**{**PARAMS, key: bad})


@pytest.mark.parametrize("key", ["purge_bars", "embargo_bars"])
@pytest.mark.parametrize("bad", [-1, False, .5])
def test_invalid_gaps_fail(key, bad):
    with pytest.raises(ValueError, match=key):
        PurgedWalkForwardCV(**{**PARAMS, key: bad})


@pytest.mark.parametrize("bad", [-1, True, 4.5])
def test_invalid_row_count_fails(bad):
    with pytest.raises(ValueError, match="n_rows"):
        list(PurgedWalkForwardCV(**PARAMS).split(bad))


def test_existing_valid_fold_indices_are_unchanged():
    folds = list(PurgedWalkForwardCV(**PARAMS).split(24))
    assert len(folds) == 2
    np.testing.assert_array_equal(folds[0].train, np.arange(8))
    np.testing.assert_array_equal(folds[0].val, np.arange(10, 14))
    np.testing.assert_array_equal(folds[0].test, np.arange(16, 20))
    np.testing.assert_array_equal(folds[1].test, np.arange(20, 24))
