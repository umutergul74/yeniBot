from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yenibot.experiment.replacement as replacement
from yenibot.experiment.common import _hash_payload
from yenibot.experiment.frozen import freeze_candidate_manifests


def _config(anchor: pd.Timestamp) -> dict:
    return {
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
                "replacement_candidate": {
                    "enabled": True,
                    "candidate_id": "control_recent3_v2",
                    "profile": "control",
                    "policy_name": "recent_3_equal",
                    "selection_track": "balanced_noninferiority",
                    "recent_k": 3,
                    "anchor_data_end": anchor.isoformat(),
                    "fold_scope": "replacement_recent3",
                    "resume_existing": True,
                    "force_retrain": False,
                }
            }
        },
    }


def _frame() -> pd.DataFrame:
    rows = 60
    labels = np.asarray(([0, 1] * 30), dtype=int)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "feature": np.linspace(-1.0, 1.0, rows),
            "label": labels,
            "fwd_return_10h": np.where(labels == 1, 0.01, -0.01),
        }
    )


def test_replacement_fit_uses_fixed_policy_anchor_and_validation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    anchor = pd.to_datetime(frame["timestamp"], utc=True).max()
    config = _config(anchor)
    run_dir = tmp_path / "experiments" / "run"
    research_dir = run_dir / "recency_research"
    research_dir.mkdir(parents=True)
    (research_dir / "recency_ensemble_decision.json").write_text(
        json.dumps(
            {
                "status": "historical_policy_cleared_balanced_noninferiority_gates",
                "recommended_policy": "recent_3_equal",
                "recommended_selection_track": "balanced_noninferiority",
                "candidate_ready_for_preregistration": True,
                "failed_future_oos_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )

    observed: dict[str, object] = {}

    def fake_run_profile(frame_arg, config_arg, **kwargs):
        observed["fit_end"] = pd.to_datetime(frame_arg["timestamp"], utc=True).max()
        observed["fold_ids"] = kwargs["fold_ids"]
        output_dir = run_dir / "control" / "replacement_recent3"
        output_dir.mkdir(parents=True)
        return {
            "output_dir": output_dir,
            "feature_columns": ["feature"],
            "skipped": False,
            "summary": {},
        }

    def fake_predict(**kwargs):
        timestamps = pd.to_datetime(
            kwargs["holdout_context"]["timestamp"],
            utc=True,
        )
        timestamps = timestamps[timestamps >= kwargs["holdout_start"]]
        labels = np.asarray(([0, 1] * 20)[: len(timestamps)], dtype=int)
        rows = []
        for fold, offset in zip(sorted(kwargs["model_folds"]), (-0.03, 0.0, 0.03)):
            for timestamp, label in zip(timestamps, labels, strict=True):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "model_fold": fold,
                        "prob_long": 0.25 + 0.5 * label + offset,
                        "label": label,
                        "forward_return": 0.01 if label else -0.01,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(replacement, "run_profile_experiment", fake_run_profile)
    monkeypatch.setattr(replacement, "_predict_holdout_for_profile", fake_predict)

    result = replacement.run_replacement_candidate_fit(
        frame=frame,
        config=config,
        checkpoint_dir=tmp_path,
        run_id="run",
    )

    assert result["status"] == "fit_complete_manifest_pin_required"
    assert result["candidate_id"] == "control_recent3_v2"
    assert result["selected_model_folds"] == observed["fold_ids"]
    assert len(result["selected_model_folds"]) == 3
    assert observed["fit_end"] == anchor
    assert result["threshold"]["selected_from"] == (
        "pre_anchor_latest_fold_validation_only"
    )
    assert result["threshold"]["guarded"] is True
    assert result["failed_future_oos_used_for_policy_selection"] is False
    assert result["manifest_pin_required"] is True
    assert (run_dir / "replacement_candidate_fit.json").exists()


def test_replacement_fit_rejects_policy_not_selected_historically(
    tmp_path: Path,
) -> None:
    frame = _frame()
    anchor = pd.to_datetime(frame["timestamp"], utc=True).max()
    config = _config(anchor)
    run_dir = tmp_path / "experiments" / "run" / "recency_research"
    run_dir.mkdir(parents=True)
    (run_dir / "recency_ensemble_decision.json").write_text(
        json.dumps(
            {
                "recommended_policy": "latest_only",
                "recommended_selection_track": "balanced_noninferiority",
                "candidate_ready_for_preregistration": True,
                "failed_future_oos_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        replacement.run_replacement_candidate_fit(
            frame=frame,
            config=config,
            checkpoint_dir=tmp_path,
            run_id="run",
        )


def test_replacement_manifest_freezes_threshold_policy_and_three_models(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "experiments" / "run"
    scope = run_dir / "control" / "replacement_recent3"
    scope.mkdir(parents=True)
    anchor = pd.Timestamp("2026-06-13 01:00:00", tz="UTC")
    features = ["feature"]
    (scope / "training_manifest.json").write_text(
        json.dumps(
            {
                "data_start": "2025-01-01T00:00:00+00:00",
                "data_end": anchor.isoformat(),
                "feature_columns": features,
                "feature_columns_hash": _hash_payload(features),
                "signature_hash": "training-signature",
            }
        ),
        encoding="utf-8",
    )
    replacement_payload = {
        "candidate_id": "control_recent3_v2",
        "anchor_data_end": anchor.isoformat(),
        "selected_model_folds": [7, 8, 9],
        "selection_evidence_hash": "selection-evidence",
        "threshold": {
            "value": 0.42,
            "source": "replacement_recent3_common_validation_guarded_f1",
            "selected_from": "pre_anchor_latest_fold_validation_only",
        },
    }
    (scope / "replacement_candidate_fit.json").write_text(
        json.dumps(replacement_payload),
        encoding="utf-8",
    )
    for fold in (7, 8, 9):
        (scope / f"model_fold_{fold:03d}.pt").write_bytes(b"model")
        (scope / f"scaler_fold_{fold:03d}.pkl").write_bytes(b"scaler")
        (scope / f"hmm_fold_{fold:03d}.pkl").write_bytes(b"hmm")
    config = {
        "experiments": {
            "frozen_candidates": {
                "enabled": True,
                "protocol_version": "v1",
                "anchor_data_end": "2026-05-13 08:00:00+00:00",
                "primary_candidate_id": "old_primary",
                "candidates": [
                    {
                        "candidate_id": "control_recent3_v2",
                        "candidate_type": "recency_profile",
                        "profile": "control",
                        "source_run_id": "run",
                        "anchor_data_end": anchor.isoformat(),
                        "fold_scope": "replacement_recent3",
                        "recency_policy": "equal_recent_k",
                        "recent_k": 3,
                        "status": "preregistration_build_manifest_pin_pending",
                        "required_for_evaluation": False,
                    }
                ],
            }
        }
    }

    manifests, index = freeze_candidate_manifests(
        run_dir=run_dir,
        report_dir=tmp_path / "reports",
        entries=[],
        config=config,
    )

    manifest = manifests[0]
    assert manifest["available"] is True
    assert manifest["candidate_type"] == "recency_profile"
    assert manifest["selected_model_folds"] == [7, 8, 9]
    assert manifest["recency_policy"] == "equal_recent_k"
    assert manifest["threshold"]["value"] == pytest.approx(0.42)
    assert manifest["selection_evidence_hash"] == "selection-evidence"
    assert int(index.iloc[0]["model_count"]) == 3
