from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from yenibot.experiment.preprocessing_audit import write_preprocessing_audit
from yenibot.training.preprocessing import CausalFoldPreprocessor


def test_disabled_preprocessing_matches_robust_scaler() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 5.0, 8.0], "b": [10.0, 12.0, 14.0, 20.0]})
    config = {"training": {"preprocessing": {}}}

    actual = CausalFoldPreprocessor(["a", "b"], config).fit(frame).transform(frame)
    expected = RobustScaler().fit_transform(frame)

    np.testing.assert_allclose(actual, expected)


def test_quantile_clip_is_fit_on_train_only_and_survives_joblib(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "4h_large_trade_ratio": np.r_[np.linspace(0.0, 1.0, 100), 20.0],
            "other": np.linspace(-1.0, 1.0, 101),
        }
    )
    config = {
        "training": {
            "preprocessing": {
                "quantile_clip": {
                    "enabled": True,
                    "columns": ["4h_large_trade_ratio"],
                    "lower_quantile": 0.01,
                    "upper_quantile": 0.99,
                }
            }
        }
    }
    processor = CausalFoldPreprocessor(list(train.columns), config).fit(train)
    original_bounds = dict(processor.clip_bounds)
    future = pd.DataFrame({"4h_large_trade_ratio": [1_000_000.0], "other": [5.0]})

    path = tmp_path / "processor.pkl"
    joblib.dump(processor, path)
    restored = joblib.load(path)

    assert restored.clip_bounds == original_bounds
    np.testing.assert_allclose(restored.transform(future), processor.transform(future))
    assert restored.audit_frame().iloc[0]["train_clip_fraction"] > 0.0


def test_stability_mask_detects_train_block_return_and_label_reversal() -> None:
    block_rows = 100
    feature = np.tile(np.linspace(-1.0, 1.0, block_rows), 6)
    returns = feature.copy()
    labels = (feature > 0).astype(int)
    returns[-2 * block_rows :] *= -1.0
    labels[-2 * block_rows :] = (feature[-2 * block_rows :] < 0).astype(int)
    frame = pd.DataFrame(
        {
            "4h_taker_imbalance_mean_12": feature,
            "stable_feature": np.sin(np.arange(len(feature)) / 20.0),
        }
    )
    config = {
        "training": {
            "preprocessing": {
                "stability_mask": {
                    "enabled": True,
                    "patterns": ["4h_taker_imbalance_mean_*"],
                    "block_rows": block_rows,
                    "min_blocks": 5,
                    "recent_blocks": 2,
                    "min_abs_rank_ic": 0.01,
                    "min_abs_label_gap": 0.05,
                    "min_rank_ic_sign_agreement": 0.60,
                }
            }
        }
    }

    processor = CausalFoldPreprocessor(list(frame.columns), config).fit(
        frame,
        forward_returns=pd.Series(returns),
        labels=pd.Series(labels),
    )
    transformed = processor.transform(frame)
    audit = processor.audit_frame().set_index("feature")

    assert processor.masked_columns == ["4h_taker_imbalance_mean_12"]
    assert bool(audit.loc["4h_taker_imbalance_mean_12", "rank_ic_reversal"])
    assert bool(audit.loc["4h_taker_imbalance_mean_12", "label_gap_reversal"])
    assert np.allclose(transformed[:, 0], 0.0)
    assert not np.allclose(transformed[:, 1], 0.0)


def test_preprocessing_audit_collects_profile_scope_files(tmp_path) -> None:
    scope = tmp_path / "run" / "profile" / "triage"
    scope.mkdir(parents=True)
    pd.DataFrame(
        [{"fold": 7, "feature": "4h_taker_imbalance_mean_12", "masked": True}]
    ).to_csv(scope / "preprocessing_audit.csv", index=False)

    report = write_preprocessing_audit(
        [{"profile": "candidate", "fold_scope": "triage", "output_dir": scope}],
        tmp_path / "reports",
    )

    assert report.iloc[0]["profile"] == "candidate"
    assert report.iloc[0]["fold_scope"] == "triage"
    assert (tmp_path / "reports" / "preprocessing_audit.csv").exists()
    summary = pd.read_csv(tmp_path / "reports" / "preprocessing_audit_summary.csv")
    assert summary.iloc[0]["masked_decision_count"] == 1
    assert summary.iloc[0]["masked_fold_count"] == 1
