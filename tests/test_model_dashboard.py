from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yenibot.experiment.dashboard import (
    attach_active_charter_status,
    write_model_performance_dashboard,
)


CONTROL = "control"


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for fold in range(4):
        for idx in range(80):
            forward_return = float(rng.normal(0.0, 0.01))
            signal = forward_return * 12.0 + float(rng.normal(0.0, 0.45))
            probability = float(1.0 / (1.0 + np.exp(-signal)))
            rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=fold * 100 + idx),
                    "label": int(forward_return > 0.003),
                    "prob_long": probability,
                    "forward_return": forward_return,
                }
            )
    return pd.DataFrame(rows)


def _readiness(blockers: list[str] | None = None) -> dict[str, object]:
    return {
        "active_validation_charter": "v4_evidence",
        "ready_for_phase2": False,
        "blockers": blockers or ["future_unseen_oos_not_ready"],
        "checks": [
            {
                "check": "mean_rank_ic",
                "value": 0.061,
                "target": ">= 0.03",
                "status": "passed",
                "role": "gate",
                "source": "rank_ic_aggregate_evidence.csv",
            },
            {
                "check": "positive_fold_fraction",
                "value": 0.83,
                "target": ">= 0.75",
                "status": "passed",
                "role": "gate",
                "source": "rank_ic_aggregate_evidence.csv",
            },
            {
                "check": "rank_ic_std",
                "value": 0.071,
                "target": "< 0.03",
                "status": "monitor",
                "role": "monitor",
                "source": "fold_stability_summary.csv",
            },
            {
                "check": "prauc_lift_vs_prevalence",
                "value": 1.12,
                "target": ">= 1.05",
                "status": "passed",
                "role": "gate",
                "source": "classification_skill_summary.csv",
            },
            {
                "check": "precision_lift_vs_prevalence",
                "value": 1.10,
                "target": ">= 1.05",
                "status": "passed",
                "role": "gate",
                "source": "classification_skill_summary.csv",
            },
            {
                "check": "f1_skill_vs_rate_matched_random",
                "value": 0.027,
                "target": "> 0",
                "status": "passed",
                "role": "gate",
                "source": "classification_skill_summary.csv",
            },
            {
                "check": "positive_forward_return_fold_fraction",
                "value": 0.69,
                "target": ">= 0.60",
                "status": "passed",
                "role": "gate",
                "source": "classification_skill_summary.csv",
            },
            {
                "check": "raw_long_f1",
                "value": 0.431,
                "target": "> 0.45",
                "status": "monitor",
                "role": "monitor",
                "source": "classification_skill_summary.csv",
            },
        ],
    }


def test_active_charter_status_separates_model_evidence_from_phase2() -> None:
    comparison = pd.DataFrame(
        [{"profile": CONTROL, "fold_scope": "full", "passed_phase1": False}]
    )

    enriched = attach_active_charter_status(
        comparison,
        phase2_readiness=_readiness(),
        control_profile=CONTROL,
    )

    row = enriched.iloc[0]
    assert bool(row["passed_phase1_legacy_v3"]) is False
    assert bool(row["model_evidence_passed_active_charter"]) is True
    assert bool(row["phase2_ready"]) is False
    assert row["phase1_status"] == "model_evidence_passed_future_oos_pending"


def test_model_dashboard_writes_professional_tables_and_visuals(tmp_path: Path) -> None:
    predictions = _predictions()
    fold_rows = []
    for fold, rank_ic in enumerate([0.08, 0.05, -0.02, 0.09]):
        fold_rows.append(
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "fold": fold,
                "rank_ic": rank_ic,
            }
        )
    comparison = pd.DataFrame(
        [
            {
                "profile": CONTROL,
                "fold_scope": "full",
                "passed_phase1": False,
                "top_10_lift_global": 1.13,
            }
        ]
    )
    rank_evidence = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "positive_fold_fraction": 0.75,
                "positive_fold_sign_test_pvalue": 0.01,
                "random_effects_ci_low_min": 0.018,
            }
        ]
    )
    classification = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "policy_name": "official_threshold",
            }
        ]
    )
    probability = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "mean_brier_score": 0.22,
                "mean_log_loss": 0.64,
                "mean_ece_equal_count": 0.04,
            }
        ]
    )
    uncertainty = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "estimand": "macro_fold",
                "block_length": 24,
                "metric": "prauc_lift_vs_prevalence",
                "ci_low": 1.06,
                "gate": 1.05,
                "probability_above_gate": 0.96,
            },
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "estimand": "macro_fold",
                "block_length": 24,
                "metric": "f1_skill_vs_rate_matched_random",
                "ci_low": 0.01,
                "gate": 0.0,
                "probability_above_gate": 0.99,
            },
        ]
    )
    calibration_comparison = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "method": "raw",
                "mean_brier_skill_vs_climatology": -0.10,
                "pooled_brier_skill_vs_climatology": -0.08,
                "positive_brier_skill_fold_fraction": 0.20,
                "mean_ece_equal_count": 0.15,
                "recommended_use": "ranking_score_only_not_probability",
            },
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "method": "platt",
                "mean_brier_skill_vs_climatology": 0.01,
                "pooled_brier_skill_vs_climatology": 0.02,
                "positive_brier_skill_fold_fraction": 0.65,
                "mean_ece_equal_count": 0.04,
                "recommended_use": "diagnostic_only",
            },
        ]
    )
    stability_summary = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "fold_scope": "full",
                "worst_fold_rank_ic": -0.02,
                "top_5_variance_contribution": 0.51,
            }
        ]
    )
    payoff = pd.DataFrame(
        [
            {
                "candidate": CONTROL,
                "evaluation_scope": "cv_test",
                "band": "top_10",
                "label_lift_vs_base": 1.13,
                "mean_forward_return": 0.003,
            },
            {
                "candidate": CONTROL,
                "evaluation_scope": "holdout",
                "band": "top_10",
                "label_lift_vs_base": 1.01,
                "mean_forward_return": -0.0003,
            },
        ]
    )
    seed_stability = pd.DataFrame(
        [{"profile": CONTROL, "mean_rank_ic_seed_std": 0.012}]
    )
    future = {
        "new_labeled_rows": 253,
        "min_rows": 720,
        "min_rows_remaining": 467,
        "ready_for_evaluation": False,
    }

    result = write_model_performance_dashboard(
        tmp_path,
        entries=[
            {
                "profile": CONTROL,
                "fold_scope": "full",
                "predictions": predictions,
            }
        ],
        comparison=comparison,
        fold_stability_forensics=pd.DataFrame(fold_rows),
        fold_stability_summary=stability_summary,
        rank_ic_aggregate_evidence=rank_evidence,
        classification_skill_summary=classification,
        probability_quality_summary=probability,
        model_evidence_uncertainty=uncertainty,
        probability_calibration_comparison=calibration_comparison,
        payoff_alignment=payoff,
        seed_stability=seed_stability,
        phase2_readiness=_readiness(),
        future_oos_readiness=future,
        control_profile=CONTROL,
    )

    expected = {
        "model_performance_dashboard.md",
        "model_performance_summary.json",
        "model_performance_scorecard.csv",
        "model_metric_definitions.csv",
        "model_calibration_reliability.csv",
        "model_precision_recall_curve.csv",
        "model_scorecard.png",
        "rank_ic_stability.png",
        "classification_quality.png",
        "score_band_payoff.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
    assert not result["calibration"].empty
    assert not result["precision_recall"].empty
    assert {
        "random_effects_lower_ci_min",
        "worst_fold_rank_ic",
        "brier_score",
        "top_10_cv_forward_return",
        "future_oos_progress",
        "prauc_lift_macro_ci_low_min",
        "platt_macro_brier_skill",
    }.issubset(set(result["scorecard"]["metric"]))
    summary = json.loads(
        (tmp_path / "model_performance_summary.json").read_text(encoding="utf-8")
    )
    assert summary["model_evidence_passed"] is True
    assert summary["phase2_ready"] is False
    assert summary["blockers"] == ["future_unseen_oos_not_ready"]


def test_model_dashboard_fails_model_evidence_for_non_future_blocker(
    tmp_path: Path,
) -> None:
    readiness = _readiness(
        ["active_charter_rank_ic_mean_failed", "future_unseen_oos_not_ready"]
    )
    comparison = pd.DataFrame(
        [{"profile": CONTROL, "fold_scope": "full", "passed_phase1": False}]
    )

    enriched = attach_active_charter_status(
        comparison,
        phase2_readiness=readiness,
        control_profile=CONTROL,
    )

    assert bool(enriched.iloc[0]["model_evidence_passed_active_charter"]) is False
    assert enriched.iloc[0]["phase1_status"] == "active_charter_model_evidence_failed"


def test_future_oos_candidate_failure_is_model_evidence_failure() -> None:
    comparison = pd.DataFrame(
        [{"profile": CONTROL, "fold_scope": "full", "passed_phase1": False}]
    )
    readiness = _readiness(["future_unseen_oos_candidate_failed"])

    enriched = attach_active_charter_status(
        comparison,
        phase2_readiness=readiness,
        control_profile=CONTROL,
    )

    assert bool(enriched.iloc[0]["model_evidence_passed_active_charter"]) is False
    assert enriched.iloc[0]["phase1_status"] == "active_charter_model_evidence_failed"
