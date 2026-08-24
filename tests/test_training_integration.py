from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from yenibot.features import build_feature_matrix
from yenibot.training.sample_weights import build_sample_weights, label_uniqueness_weights
from yenibot.training import PurgedWalkForwardCV, run_walk_forward_training, train_one_fold


def test_small_pipeline_runs_one_training_step(synthetic_klines, tiny_config, tmp_path) -> None:
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)

    # Deterministic synthetic labels for integration wiring only.
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)

    cv = PurgedWalkForwardCV(**tiny_config["walk_forward"])
    fold = next(cv.split(len(frame)))
    result = train_one_fold(
        frame,
        fold,
        features.feature_columns,
        tiny_config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert not result["predictions"].empty
    assert (tmp_path / "scaler_fold_000.pkl").exists()
    assert (tmp_path / "model_fold_000.pt").exists()
    assert (tmp_path / "hmm_fold_000.pkl").exists()
    assert (tmp_path / "predictions_fold_000.parquet").exists()


def test_train_one_fold_repeats_with_same_seed_on_cpu(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["project"] = {"random_seed": 123, "deterministic": True}
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    first = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")
    second = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    np.testing.assert_allclose(
        first["predictions"]["prob_long"].to_numpy(),
        second["predictions"]["prob_long"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_one_fold_supports_val_loss_early_stopping_as_explicit_experiment(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["early_stop_metric"] = "val_loss"
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    history = result["history"]
    assert set(history["early_stop_metric"]) == {"val_loss"}
    assert history["early_stop_value"].notna().all()
    assert history["val_loss"].notna().all()


def test_train_one_fold_supports_pairwise_label_margin_loss(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["loss"]["label_margin_weight"] = 0.05
    config["training"]["loss"]["label_margin"] = 0.25
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    assert not result["predictions"].empty
    assert result["history"]["train_loss"].notna().all()


def test_train_one_fold_supports_pairwise_return_order_loss(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["loss"]["return_pairwise_weight"] = 0.03
    config["training"]["loss"]["return_pairwise_margin"] = 0.05
    config["training"]["loss"]["return_pairwise_min_return_diff"] = 0.0005
    config["training"]["loss"]["return_pairwise_return_scale"] = 0.005
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    assert not result["predictions"].empty
    assert result["history"]["train_loss"].notna().all()


def test_train_one_fold_supports_conflict_projected_auxiliary_return_head(
    synthetic_klines,
    tiny_config,
    tmp_path,
) -> None:
    config = copy.deepcopy(tiny_config)
    config["model"]["auxiliary_return_head"] = True
    config["model"]["auxiliary_return_scale"] = 0.01
    config["training"]["auxiliary_return"] = {
        "enabled": True,
        "weight": 0.10,
        "target_clip": 5.0,
        "huber_beta": 1.0,
        "gradient_strategy": "primary_preserving_projection",
    }
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(
        frame,
        fold,
        features.feature_columns,
        config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert "auxiliary_return_prediction" in result["predictions"]
    assert result["predictions"]["auxiliary_return_prediction"].notna().all()
    assert result["auxiliary_task_audit"]["enabled"].all()
    assert result["auxiliary_task_audit"][
        "test_auxiliary_return_rank_ic"
    ].notna().all()
    assert result["multitask_gradient_audit"]["enabled"].all()
    assert set(result["multitask_gradient_audit"]["strategy"]) == {
        "primary_preserving_projection"
    }
    assert result["multitask_gradient_audit"]["batch_count"].gt(0).all()
    assert (tmp_path / "auxiliary_task_audit_fold_000.csv").exists()
    assert (tmp_path / "multitask_gradient_audit_fold_000.csv").exists()


def test_train_one_fold_uses_single_preregistered_swa_checkpoint(
    synthetic_klines,
    tiny_config,
    tmp_path,
) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["epochs"] = 3
    config["training"]["early_stop_patience"] = 2
    config["training"]["weight_averaging"] = {
        "enabled": True,
        "strategy": "swa",
        "start_epoch": 1,
        "update_interval_epochs": 1,
        "min_snapshots": 2,
    }
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(
        frame,
        fold,
        features.feature_columns,
        config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    audit = result["weight_averaging_audit"].iloc[0]
    assert bool(audit["enabled"])
    assert audit["strategy"] == "swa"
    assert audit["snapshots_collected"] >= 2
    assert audit["selected_epoch"] >= 2
    assert audit["selected_mean_abs_parameter_delta"] > 0.0
    assert bool(audit["single_checkpoint_output"])
    assert set(result["history"].loc[1:, "selection_model_source"]) == {"swa"}
    assert (tmp_path / "weight_averaging_audit_fold_000.csv").exists()
    checkpoint = torch.load(
        tmp_path / "model_fold_000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["config_training"]["weight_averaging"]["enabled"] is True


def test_label_uniqueness_weights_downweight_overlapping_events() -> None:
    frame = pd.DataFrame({"label": np.zeros(12, dtype=int)})
    weights = label_uniqueness_weights(
        frame=frame,
        horizon_bars=4,
    )

    assert len(weights) == 12
    assert weights[4] < weights[0]


def test_build_sample_weights_uses_train_fold_only_event_columns() -> None:
    frame = np.arange(20, dtype=float)
    train = {
        "timestamp": np.arange(20),
        "event_feature": frame,
        "label": (frame % 3 == 0).astype(int),
        "fwd_return_10h": frame / 1000.0,
    }
    config = {
        "labeling": {"max_holding_bars": 4},
        "training": {
            "sample_weighting": {
                "enabled": True,
                "normalize_mean": True,
                "min_weight": 0.25,
                "max_weight": 2.5,
                "components": {
                    "uniqueness": {"enabled": False},
                    "event": {
                        "enabled": True,
                        "include_patterns": ["event_*"],
                        "aggregation": "mean_abs_then_rank",
                        "quantile": 0.80,
                        "strength": 0.5,
                        "effectiveness_guard": {
                            "enabled": True,
                            "min_active_fraction": 0.15,
                            "max_active_fraction": 0.25,
                            "min_p90_p10_spread": 0.05,
                            "max_dominant_weight_fraction": 0.85,
                        },
                    },
                },
            }
        },
    }

    weights, audit = build_sample_weights(
        train_frame=pd.DataFrame(train),
        feature_columns=["event_feature"],
        config=config,
    )

    assert len(weights) == 20
    assert np.isclose(weights.mean(), 1.0)
    assert {"uniqueness", "event", "combined"}.issubset(set(audit["component"]))
    event_row = audit.loc[audit["component"].eq("event")].iloc[0]
    assert event_row["selected_column_count"] == 1
    assert 0.15 <= event_row["active_fraction"] <= 0.25
    assert event_row["p90_p10_spread"] >= 0.05
    assert event_row["event_aggregation"] == "mean_abs_then_rank"


def test_build_sample_weights_rejects_inert_event_aggregation() -> None:
    row_count = 100
    frame = pd.DataFrame(
        {
            f"event_{idx:02d}": (
                np.arange(row_count, dtype=float)
                if idx % 2 == 0
                else np.arange(row_count - 1, -1, -1, dtype=float)
            )
            for idx in range(20)
        }
    )
    config = {
        "training": {
            "sample_weighting": {
                "enabled": True,
                "components": {
                    "uniqueness": {"enabled": False},
                    "event": {
                        "enabled": True,
                        "include_patterns": ["event_*"],
                        "aggregation": "mean_feature_rank",
                        "quantile": 0.80,
                        "strength": 0.35,
                        "effectiveness_guard": {
                            "enabled": True,
                            "min_active_fraction": 0.15,
                            "max_active_fraction": 0.25,
                            "min_p90_p10_spread": 0.05,
                            "max_dominant_weight_fraction": 0.85,
                        },
                    },
                },
            }
        }
    }

    with pytest.raises(ValueError, match="effectively a no-op"):
        build_sample_weights(
            train_frame=frame,
            feature_columns=list(frame.columns),
            config=config,
        )


def test_train_one_fold_rejects_unknown_early_stop_metric(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["early_stop_metric"] = "not_a_real_metric"
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    with pytest.raises(ValueError, match="Unsupported early_stop_metric"):
        train_one_fold(frame, fold, features.feature_columns, config, device="cpu")


def test_run_walk_forward_training_fails_fast_on_active_feature_nans(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    frame.loc[0, features.feature_columns[0]] = np.nan

    with pytest.raises(ValueError, match="Re-run notebooks 02 and 03"):
        run_walk_forward_training(frame, tiny_config, feature_columns=features.feature_columns, max_folds=1, device="cpu")


def test_run_walk_forward_training_honors_selected_fold_ids(synthetic_klines, tiny_config, tmp_path) -> None:
    primary = synthetic_klines(260, "1h")
    htf = synthetic_klines(80, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)

    result = run_walk_forward_training(
        frame,
        tiny_config,
        feature_columns=features.feature_columns,
        checkpoint_dir=tmp_path,
        fold_ids=[1],
        device="cpu",
    )

    assert sorted(result["predictions"]["fold"].unique().tolist()) == [1]
    assert not (tmp_path / "model_fold_000.pt").exists()
    assert (tmp_path / "model_fold_001.pt").exists()
    assert (tmp_path / "preprocessing_audit.csv").exists()


def test_train_one_fold_writes_sample_weight_audit_when_enabled(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["sample_weighting"] = {
        "enabled": True,
        "normalize_mean": True,
        "min_weight": 0.25,
        "max_weight": 2.5,
        "components": {
            "uniqueness": {"enabled": True, "power": 1.0},
            "event": {"enabled": False},
        },
    }
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(
        frame,
        fold,
        features.feature_columns,
        config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert not result["sample_weight_audit"].empty
    assert (tmp_path / "sample_weight_audit_fold_000.csv").exists()
