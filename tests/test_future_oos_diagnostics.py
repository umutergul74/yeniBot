from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import yenibot.experiment.rolling_research as rolling_module
from yenibot.experiment.future_oos_diagnostics import (
    future_oos_diagnostic_frames,
    future_oos_failure_summary,
    future_oos_model_metrics,
)
from yenibot.experiment.rolling_research import (
    aggregate_recency_predictions,
    recency_weights,
    rolling_origin_schedule,
    run_recency_ensemble_research,
)


def _predictions() -> pd.DataFrame:
    rows = 40
    labels = np.asarray(([0, 1] * 20), dtype=int)
    scores = np.linspace(0.2, 0.8, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "candidate_id": "candidate",
            "prob_long": scores,
            "prob_long_model_std": np.linspace(0.01, 0.12, rows),
            "prob_long_model_min": scores - 0.1,
            "prob_long_model_max": scores + 0.1,
            "model_fold_count": 4,
            "label": labels,
            "forward_return": np.where(labels == 1, 0.01, -0.01),
            "tb_return": np.where(labels == 1, 0.02, -0.01),
            "regime_prob_0": np.linspace(0.8, 0.1, rows),
            "regime_prob_1": np.linspace(0.1, 0.8, rows),
            "regime_prob_2": 0.1,
        }
    )


def test_future_oos_diagnostics_cover_time_bands_regimes_and_disagreement() -> None:
    frames = future_oos_diagnostic_frames(
        _predictions(),
        threshold=0.5,
        block_hours=12,
    )

    assert not frames["temporal_blocks"].empty
    assert frames["score_bands"]["score_decile"].nunique() == 10
    assert set(frames["regime_metrics"]["regime"]) == {"0", "1"}
    assert frames["ensemble_disagreement"].loc[0, "model_count_min"] == 4


def test_future_oos_failure_summary_distinguishes_ranking_from_threshold_only() -> None:
    model_metrics = pd.DataFrame(
        {
            "model_fold": list(range(8)),
            "rank_ic": np.linspace(-0.2, 0.2, 8),
        }
    )
    summary = future_oos_failure_summary(
        {
            "candidate_id": "candidate",
            "evidence_passed": False,
            "failed_gates": (
                "rank_ic;rank_ic_lower_ci;prauc_lift;top_10_lift;"
                "top_10_forward_return;pred_long_rate"
            ),
        },
        temporal_blocks=pd.DataFrame({"rank_ic": [-0.1, 0.1]}),
        ensemble_disagreement=pd.DataFrame({"rows": [40]}),
        model_metrics=model_metrics,
    )

    assert summary["primary_failure_mechanism"] == (
        "ranking_and_payoff_breakdown_not_threshold_only"
    )
    assert summary["candidate_status"] == "retired_after_failed_future_oos"
    assert summary["same_window_tuning_allowed"] is False
    assert summary["recency_signal"] == (
        "newer_models_outperform_older_models_diagnostic_only"
    )


def test_model_metrics_preserve_per_fold_evidence() -> None:
    base = _predictions().drop(
        columns=[
            "candidate_id",
            "prob_long_model_std",
            "prob_long_model_min",
            "prob_long_model_max",
            "model_fold_count",
        ]
    )
    raw = pd.concat(
        [
            base.assign(model_fold=0),
            base.assign(model_fold=1, prob_long=1.0 - base["prob_long"]),
        ],
        ignore_index=True,
    )

    metrics = future_oos_model_metrics(
        raw,
        candidate_id="candidate",
        profile="control",
        threshold=0.5,
    )

    assert metrics["model_fold"].tolist() == [0, 1]
    assert metrics["rows"].tolist() == [40, 40]


def test_recency_weights_are_causal_and_normalized() -> None:
    weights = recency_weights(
        [0, 1, 2, 3],
        target_fold=3,
        policy="exponential_decay",
        half_life_folds=2,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[3] > weights[2] > weights[1] > weights[0]
    with pytest.raises(ValueError, match="Future model folds"):
        recency_weights(
            [0, 1, 4],
            target_fold=3,
            policy="latest_only",
        )


def test_recency_aggregation_uses_only_selected_models() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "model_fold": fold,
                    "prob_long": value,
                    "label": [0, 1, 0],
                    "forward_return": [-0.01, 0.01, -0.01],
                }
            )
            for fold, value in [(0, 0.2), (1, 0.4), (2, 0.8)]
        ],
        ignore_index=True,
    )

    aggregated = aggregate_recency_predictions(
        raw,
        target_fold=2,
        policy="equal_recent_k",
        recent_k=2,
    )

    assert aggregated["prob_long"].tolist() == pytest.approx([0.6, 0.6, 0.6])
    assert aggregated["model_count"].tolist() == [2, 2, 2]


def test_rolling_schedule_never_exposes_future_model_folds() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
        }
    )
    config = {
        "walk_forward": {
            "train_bars": 20,
            "val_bars": 8,
            "test_bars": 6,
            "step_bars": 6,
            "purge_bars": 2,
            "embargo_bars": 1,
        }
    }

    schedule = rolling_origin_schedule(frame, config)

    assert not schedule.empty
    assert (schedule["future_model_count"] == 0).all()
    assert (
        schedule["latest_eligible_model_fold"].astype(int)
        == schedule["fold"].astype(int)
    ).all()


def test_recency_research_selects_thresholds_on_validation_only(
    tmp_path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC"),
            "feature": np.linspace(-1, 1, 60),
            "label": np.arange(60) % 2,
            "fwd_return_10h": np.where(np.arange(60) % 2, 0.01, -0.01),
        }
    )
    scope = tmp_path / "control" / "full"
    scope.mkdir(parents=True)
    (scope / "training_manifest.json").write_text(
        (
            '{"profile":"control","feature_columns":["feature"],'
            '"signature_hash":"training"}'
        ),
        encoding="utf-8",
    )
    for fold in range(5):
        (scope / f"model_fold_{fold:03d}.pt").write_bytes(b"model")

    def fake_predict(
        *,
        holdout_context,
        holdout_start,
        holdout_end,
        model_folds,
        **_,
    ):
        part = holdout_context.copy()
        timestamps = pd.to_datetime(part["timestamp"], utc=True)
        part = part.loc[timestamps.between(holdout_start, holdout_end)].copy()
        rows = []
        for model_fold in sorted(model_folds):
            score = 0.2 + 0.1 * model_fold + 0.4 * part["label"].to_numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": part["timestamp"].to_numpy(),
                        "model_fold": model_fold,
                        "prob_long": np.clip(score, 0, 1),
                        "label": part["label"].to_numpy(),
                        "forward_return": part["fwd_return_10h"].to_numpy(),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    monkeypatch.setattr(
        rolling_module,
        "_predict_holdout_for_profile",
        fake_predict,
    )
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {
                    "include_patterns": ["feature"],
                    "exclude_patterns": [],
                }
            },
        },
        "model": {"seq_len": 4},
        "validation": {
            "threshold_checks": {
                "max_pred_long_rate": 0.70,
                "min_precision": 0.30,
            }
        },
        "walk_forward": {
            "train_bars": 20,
            "val_bars": 8,
            "test_bars": 6,
            "step_bars": 6,
            "purge_bars": 2,
            "embargo_bars": 1,
        },
        "experiments": {
            "next_research_cycle": {
                "status": "research_only",
                "same_window_selection_allowed": False,
                "new_future_oos_anchor_required": True,
                "recency_ensemble": {
                    "enabled": True,
                    "policies": [
                        {"name": "latest", "policy": "latest_only"},
                        {
                            "name": "recent_2",
                            "policy": "equal_recent_k",
                            "recent_k": 2,
                        },
                    ],
                },
            }
        },
    }

    result = run_recency_ensemble_research(
        frame=frame,
        scope_dir=scope,
        config=config,
        output_dir=tmp_path / "research",
    )

    assert result["status"] == "completed"
    assert set(result["summary"]["policy_name"]) == {"latest", "recent_2"}
    assert result["summary"]["failed_future_oos_used_for_selection"].eq(False).all()
    assert result["eligibility_audit"]["eligible"].all()
    assert (tmp_path / "research" / "recency_ensemble_manifest.json").exists()
