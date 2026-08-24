from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yenibot.experiment.adaptive_ensemble import (
    aggregate_fixed_model_predictions,
    select_models_by_validation_lcb,
    split_validation_predictions,
)
from yenibot.experiment.cached_policy_research import (
    run_cached_adaptive_ensemble_research,
)


def _raw_predictions(
    timestamps: pd.DatetimeIndex,
    model_folds: list[int],
) -> pd.DataFrame:
    labels = (np.arange(len(timestamps)) % 3 == 0).astype(int)
    returns = np.linspace(-0.02, 0.02, len(timestamps))
    rows = []
    for model_fold in model_folds:
        direction = 1.0 if model_fold % 2 == 0 else -1.0
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "model_fold": model_fold,
                    "prob_long": 0.5 + direction * returns * 5,
                    "label": labels,
                    "forward_return": returns,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_validation_selection_and_threshold_calibration_are_disjoint() -> None:
    timestamps = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    selector, calibration, metadata = split_validation_predictions(
        _raw_predictions(timestamps, [0, 1, 2]),
        selector_rows=18,
        purge_rows=4,
        min_calibration_rows=8,
    )

    assert selector["timestamp"].nunique() == 18
    assert calibration["timestamp"].nunique() == 8
    assert metadata["selection_and_calibration_disjoint"] is True
    assert set(selector["timestamp"]).isdisjoint(set(calibration["timestamp"]))


def test_validation_lcb_selects_from_fixed_recent_pool_and_equal_aggregates() -> None:
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    raw = _raw_predictions(timestamps, list(range(8)))

    selected, audit = select_models_by_validation_lcb(
        raw,
        target_fold=7,
        pool_recent_k=6,
        select_top_k=3,
        block_length=6,
        bootstrap_repeats=40,
        confidence_level=0.90,
        random_seed=42,
    )

    assert len(selected) == 3
    assert set(audit["model_fold"]) == {2, 3, 4, 5, 6, 7}
    assert set(selected).issubset({2, 4, 6})
    assert audit["selected"].sum() == 3
    aggregated = aggregate_fixed_model_predictions(
        raw,
        selected_model_folds=selected,
        policy_name="candidate",
    )
    assert aggregated["model_count"].eq(3).all()
    assert aggregated["policy"].eq("candidate").all()


def test_cached_adaptive_research_enforces_historical_contract(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    signature = "a" * 64
    (source / "recency_ensemble_manifest.json").write_text(
        json.dumps(
            {
                "signature_hash": "source",
                "prediction_signature_hash": signature,
                "failed_future_oos_used_for_policy_selection": False,
            }
        ),
        encoding="utf-8",
    )
    schedule_rows = []
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    for fold in range(3):
        validation_start = origin + pd.Timedelta(days=fold * 2)
        validation_end = validation_start + pd.Timedelta(hours=15)
        test_start = validation_end + pd.Timedelta(hours=2)
        test_end = test_start + pd.Timedelta(hours=7)
        schedule_rows.append(
            {
                "fold": fold,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        timestamps = pd.date_range(
            validation_start,
            test_end,
            freq="h",
            tz="UTC",
        )
        raw = _raw_predictions(timestamps, list(range(fold + 1)))
        raw.to_parquet(
            source / f"cross_predictions_{signature[:12]}_fold_{fold:03d}.parquet",
            index=False,
        )
    pd.DataFrame(schedule_rows).to_csv(
        source / "recency_ensemble_schedule.csv", index=False
    )
    config = {
        "validation": {
            "threshold_checks": {
                "max_pred_long_rate": 0.70,
                "min_precision": 0.0,
            }
        },
        "experiments": {
            "next_research_cycle": {
                "adaptive_ensemble": {
                    "enabled": True,
                    "preregistered": True,
                    "hypothesis_id": "adaptive_v1",
                    "source_run_id": "source",
                    "source_prediction_signature_hash": signature,
                    "historical_selection_end": "2024-01-31T00:00:00+00:00",
                    "excluded_evaluation_windows": [
                        {
                            "candidate_id": "failed",
                            "start": "2024-02-01T00:00:00+00:00",
                            "end": "2024-03-01T00:00:00+00:00",
                        }
                    ],
                    "control": {"name": "control", "recent_k": 1},
                    "policy": {
                        "name": "candidate",
                        "pool_recent_k": 3,
                        "select_top_k": 1,
                        "selector_rows": 8,
                        "purge_rows": 2,
                        "min_calibration_rows": 6,
                        "bootstrap_block_length": 2,
                        "bootstrap_repeats": 20,
                        "confidence_level": 0.90,
                        "random_seed": 42,
                    },
                    "comparison": {
                        "control_policy": "control",
                        "bootstrap_repeats": 20,
                        "block_length_folds": 2,
                        "confidence_level": 0.95,
                        "random_seed": 42,
                        "gates": {
                            "min_mean_rank_ic_delta": -1.0,
                            "max_std_rank_ic_delta": 1.0,
                            "min_positive_ic_fraction_delta": -1.0,
                            "min_worst_5_rank_ic_delta": -1.0,
                            "min_mean_top_10_lift_delta": -1.0,
                            "min_positive_selected_return_fraction_delta": -1.0,
                            "min_rank_ic_delta_probability": 0.0,
                            "min_rank_ic_win_rate": 0.0,
                        },
                    },
                }
            }
        },
    }

    result = run_cached_adaptive_ensemble_research(
        source_research_dir=source,
        output_dir=output,
        config=config,
        code_commit="preregistered-commit",
    )

    assert result["status"] == "completed"
    assert result["manifest"]["fit_operations_performed"] == 0
    assert result["manifest"]["failed_future_oos_used_for_selection"] is False
    assert result["manifest"]["selection_and_threshold_rows_disjoint"] is True
    assert result["by_fold"]["selection_and_threshold_rows_disjoint"].all()
    assert (output / "adaptive_ensemble_decision.json").exists()
