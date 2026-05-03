from __future__ import annotations

import numpy as np

from yenibot.features import build_feature_matrix
from yenibot.training import PurgedWalkForwardCV, train_one_fold


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
