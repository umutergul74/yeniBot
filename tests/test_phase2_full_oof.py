from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yenibot.phase2.full_oof import (
    build_full_oof_inputs,
    file_sha256,
    validation_cdf_test_scores,
)


def _source():
    parts = []
    for fold in range(2):
        for split, day in (("val", 1), ("test", 3)):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            f"2025-0{fold + 1}-{day:02d}", periods=5, freq="h", tz="UTC"
                        ),
                        "fold": fold,
                        "split": split,
                        "prob_long": [0.1, 0.2, 0.4, 0.6, 0.8]
                        if split == "val"
                        else [0.05, 0.2, 0.3, 0.7, 0.9],
                        "open": 100.0,
                        "high": 102.0,
                        "low": 98.0,
                        "close": 101.0,
                        "atr_14": 1.0,
                        "label": 0,
                        "forward_return": 0.01,
                        "tb_return": 0.01,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True), {
        "fold_count": 2,
        "validation_rows_per_fold": 5,
        "test_rows_per_fold": 5,
        "label_maturity_hours": 10,
        "historical_cutoff": "2025-03-01T00:00:00Z",
        "profile": "control",
        "source_run_id": "historical",
        "source_timestamp_semantics": "binance_bar_open_1h",
        "score_transform": "same_fold_validation_empirical_cdf_right",
    }


def test_cdf_uses_only_same_fold_validation_scores():
    frame, spec = _source()
    first, _ = validation_cdf_test_scores(frame, spec=spec)
    np.testing.assert_allclose(first.prob_long, [0, 0.4, 0.4, 0.8, 1] * 2)
    altered = frame.copy()
    altered["label"] = 1
    altered["forward_return"] = -99
    altered.loc[altered.fold.eq(1), "prob_long"] = 0.999
    second, _ = validation_cdf_test_scores(altered, spec=spec)
    np.testing.assert_array_equal(first.prob_long[:5], second.prob_long[:5])
    # Later test scores cannot retroactively alter earlier percentile decisions.
    altered = frame.copy()
    altered.loc[9, "prob_long"] = 0.01
    third, _ = validation_cdf_test_scores(altered, spec=spec)
    np.testing.assert_array_equal(first.prob_long[:4], third.prob_long[:4])


@pytest.mark.parametrize(
    "issue",
    ["missing_split", "cutoff", "count", "overlap", "maturity", "gap", "score", "fold"],
)
def test_cdf_rejects_bad_temporal_or_identity_contract(issue):
    frame, spec = _source()
    if issue == "missing_split":
        frame.loc[0, "split"] = None
    elif issue == "cutoff":
        spec["historical_cutoff"] = "2025-01-01T00:00:00Z"
    elif issue == "count":
        frame = frame.iloc[1:]
    elif issue == "overlap":
        frame.loc[frame.fold.eq(1), "timestamp"] -= pd.Timedelta(days=31)
    elif issue == "maturity":
        frame.loc[frame.split.eq("test"), "timestamp"] -= pd.Timedelta(hours=44)
    elif issue == "gap":
        frame.loc[9, "timestamp"] += pd.Timedelta(hours=1)
    elif issue == "score":
        frame.loc[0, "prob_long"] = np.inf
    else:
        frame["fold"] = frame.fold.astype(float)
        frame.loc[0, "fold"] = 0.5
    with pytest.raises(ValueError):
        validation_cdf_test_scores(frame, spec=spec)


def test_builder_pins_sources_and_declares_true_bar_times(tmp_path: Path):
    frame, spec = _source()
    frame.to_parquet(tmp_path / "predictions_all.parquet")
    (tmp_path / "training_manifest.json").write_text(
        json.dumps(
            {
                "completed": True,
                "fold_scope": "full",
                "profile": "control",
                "data_end": spec["historical_cutoff"],
                "prediction_rows": len(frame),
            }
        ),
        encoding="utf-8",
    )
    spec["predictions_sha256"] = file_sha256(tmp_path / "predictions_all.parquet")
    spec["training_manifest_sha256"] = file_sha256(tmp_path / "training_manifest.json")
    bars, signals, metadata = build_full_oof_inputs(tmp_path, spec=spec)
    assert len(signals) == 10
    assert (bars.bar_close_time - bars.bar_open_time).eq(pd.Timedelta(hours=1)).all()
    assert signals.decision_time.equals(bars.bar_close_time)
    assert metadata["test_labels_used_for_score_transform"] is False
    assert metadata["promotion_allowed"] is False
    spec["predictions_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_full_oof_inputs(tmp_path, spec=spec)
