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


@pytest.mark.parametrize("gap", ["purge_bars", "embargo_bars"])
def test_label_horizon_requires_both_boundary_gaps(gap):
    params = {**PARAMS, "purge_bars": 10, "embargo_bars": 10, "label_horizon": 10}
    with pytest.raises(ValueError, match=gap):
        PurgedWalkForwardCV(**{**params, gap: 6})
    for fold in PurgedWalkForwardCV(**params).split(80):
        assert fold.train[-1] + 10 < fold.val[0]
        assert fold.val[-1] + 10 < fold.test[0]


def test_direct_fold_rejects_negative_index_and_unsorted_timestamps():
    from yenibot.training.walk_forward import FoldIndices
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC"),
                          "signal": 1., "label": 0, "fwd_return_10h": .01})
    fold = FoldIndices(0, np.array([-1]), np.array([12]), np.array([24]))
    with pytest.raises(ValueError, match="in-range"):
        train_one_fold(frame, fold, ["signal"], {})
    with pytest.raises(ValueError, match="chronological"):
        _assert_training_inputs_available(frame.iloc[::-1], ["signal"], {})


def test_holdout_purges_targets_without_moving_reserved_window():
    from yenibot.experiment.holdout import prepare_training_holdout_split, _selection_frame_before_holdout
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")})
    config = {"labeling": {"max_holding_bars": 10}, "experiments": {"holdout": {"holdout_bars": 20}}}
    selection, holdout, meta = prepare_training_holdout_split(frame, config)
    assert len(selection) == 70
    assert holdout.timestamp.iloc[0] == frame.timestamp.iloc[80]
    assert meta["selection_purge_bars"] == 10
    assert selection.timestamp.iloc[-1] + pd.Timedelta(hours=10) < holdout.timestamp.iloc[0]
    settings = {"holdout": meta}
    pd.testing.assert_frame_equal(_selection_frame_before_holdout(frame, settings), selection)
    pd.testing.assert_frame_equal(_selection_frame_before_holdout(selection, settings), selection)
    with pytest.raises(ValueError, match="boundary missing"):
        _selection_frame_before_holdout(frame.iloc[:80], {"holdout": {
            "enabled": True, "holdout_data_start": str(holdout.timestamp.iloc[0])}})


def test_threshold_choice_cannot_respond_to_test_results():
    from yenibot.diagnostics.reporting import attach_threshold_summary_to_phase1_report
    from yenibot.experiment.training import _select_official_threshold_candidate
    def summary(test_value, val_score):
        values = {"source_is_validation": 1., "selected_threshold": .4, "constrained_threshold": .6,
                  "source_best_f1": val_score, "source_best_precision": .8, "source_best_pred_long_rate": .4}
        for source in ("selected", "constrained"):
            for metric in ("f1", "precision", "recall", "pred_long_rate"):
                values[f"test_{metric}_at_{source}_threshold"] = test_value
        return pd.DataFrame([{"metric": k, "mean": v} for k, v in values.items()])
    choices = []
    for test_value in (.01, .99):
        raw = summary(test_value, .6)
        calibrated = summary(1-test_value, .8)
        raw_report = attach_threshold_summary_to_phase1_report({}, raw, {})
        cal_report = attach_threshold_summary_to_phase1_report({}, calibrated, {})
        assert raw_report["threshold_guarded"]["threshold_source"] == "validation_selected_threshold"
        selected = _select_official_threshold_candidate(raw_report=raw_report, raw_threshold_summary=raw,
            calibrated_threshold_report=cal_report, calibrated_threshold_summary=calibrated, config={})
        choices.append((selected["threshold_source"], selected["threshold_mean"]))
    assert choices[0] == choices[1] == ("calibrated_validation_selected_threshold", .4)
    oracle = summary(.99, .99)
    oracle.loc[oracle.metric == "source_is_validation", "mean"] = 0.
    rejected = attach_threshold_summary_to_phase1_report({}, oracle, {})
    assert rejected["threshold_guarded"]["threshold_source"] == "unavailable_validation_threshold"
