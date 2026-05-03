from __future__ import annotations

import json
import zipfile

import pandas as pd

from yenibot.diagnostics import (
    calibration_table,
    fold_diagnostics,
    regime_diagnostics,
    write_phase1_diagnostic_bundle,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC"),
            "fold": [0] * 6 + [1] * 6,
            "label": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "prob_long": [0.1, 0.7, 0.2, 0.8, 0.4, 0.6, 0.3, 0.2, 0.5, 0.9, 0.1, 0.7],
            "forward_return": [-0.01, 0.03, -0.02, 0.02, 0.0, 0.01, -0.01, -0.02, 0.01, 0.04, 0.0, 0.02],
            "regime_prob_0": [0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4],
            "regime_prob_1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "regime_prob_2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3],
        }
    )


def test_calibration_bins_are_readable_numbers() -> None:
    table = calibration_table(_predictions()["label"], _predictions()["prob_long"], bins=4)

    assert table["bin"].tolist() == [0, 1, 2, 3]
    assert set(table.columns) == {"bin", "count", "mean_prob_long", "actual_long_rate"}


def test_diagnostic_bundle_contains_shareable_outputs(tmp_path) -> None:
    predictions = _predictions()
    calibration = calibration_table(predictions["label"], predictions["prob_long"], bins=4)
    fold_metrics = fold_diagnostics(predictions)
    regime_metrics = regime_diagnostics(predictions)
    report = {
        "passed": False,
        "mean_rank_ic": 0.01,
        "std_rank_ic": 0.10,
        "positive_ic_fraction": 0.50,
        "mean_long_f1": 0.20,
        "mean_prauc": 0.35,
        "calibration_separation": 0.01,
        "checks": {"rank_ic_mean": False},
        "alerts": [],
    }

    zip_path = write_phase1_diagnostic_bundle(
        output_dir=tmp_path,
        report=report,
        predictions=predictions,
        calibration=calibration,
        fold_metrics=fold_metrics,
        regime_metrics=regime_metrics,
        config={"project": {"name": "test"}},
    )

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "phase1_report.json" in names
        assert "summary.md" in names
        assert "test_predictions.parquet" in names
        assert "calibration.csv" in names
        assert "fold_metrics.csv" in names
        assert "regime_metrics.csv" in names
        payload = json.loads(archive.read("phase1_report.json"))
        assert payload["passed"] is False
