from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yenibot.experiment.future_oos as future_oos_module
from yenibot.experiment.charter import write_validation_charter_status
from yenibot.experiment.frozen import (
    freeze_candidate_manifests,
    verify_frozen_manifest_artifacts,
)
from yenibot.experiment.future_oos import evaluate_future_oos
from yenibot.experiment.registry import append_experiment_registry


def _config(tmp_path: Path, *, min_rows: int = 20) -> dict:
    return {
        "paths": {"data_dir": str(tmp_path / "data")},
        "model": {"seq_len": 4},
        "validation": {
            "charter": {
                "active_version": "v3_legacy",
                "activation_policy": "explicit_commit_only",
                "versions": {
                    "v3_legacy": {"status": "active"},
                    "v4_draft": {"status": "proposed_not_active"},
                },
            }
        },
        "experiments": {
            "frozen_candidates": {
                "enabled": True,
                "protocol_version": "v1",
                "anchor_run_id": "anchor",
                "anchor_data_end": "2024-01-02 00:00:00+00:00",
                "primary_candidate_id": "control_v1",
                "artifact_policy": "hash_existing",
                "candidates": [
                    {
                        "candidate_id": "control_v1",
                        "candidate_type": "profile",
                        "profile": "control",
                        "fold_scope": "full",
                        "status": "preregistered",
                    }
                ],
            },
            "future_oos_validation": {
                "enabled": True,
                "min_rows": min_rows,
                "preferred_rows": min_rows * 2,
                "block_length": 4,
                "bootstrap_repeats": 20,
                "confidence_level": 0.90,
                "random_seed": 7,
                "gates": {
                    "min_rank_ic": 0.0,
                    "min_rank_ic_lower_ci": -1.0,
                    "min_top_10_lift": 0.0,
                    "min_top_10_forward_return": -1.0,
                    "min_prauc_lift_vs_prevalence": 0.0,
                    "min_precision_lift_vs_prevalence": 0.0,
                    "min_f1_skill_vs_rate_random": -1.0,
                    "max_pred_long_rate": 1.0,
                },
            },
        },
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {
                    "include_patterns": ["feature"],
                    "exclude_patterns": [],
                }
            },
        },
        "hmm": {"features": []},
        "labeling": {"max_holding_bars": 10},
        "training": {"batch_size": 8},
    }


def _fake_scope(run_dir: Path) -> tuple[Path, dict]:
    scope = run_dir / "control" / "full"
    scope.mkdir(parents=True)
    manifest = {
        "profile": "control",
        "fold_scope": "full",
        "feature_columns": ["feature"],
        "feature_columns_hash": "features",
        "signature_hash": "training",
        "data_start": "2023-01-01 00:00:00+00:00",
        "data_end": "2024-01-01 23:00:00+00:00",
    }
    (scope / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("model_fold_000.pt", "scaler_fold_000.pkl", "hmm_fold_000.pkl"):
        (scope / name).write_bytes(name.encode("ascii"))
    entry = {
        "scope_dir": scope,
        "profile": "control",
        "fold_scope": "full",
        "diagnostics": {
            "row": {
                "official_threshold_mean": 0.55,
                "official_threshold_source": "validation_threshold",
            }
        },
    }
    return scope, entry


def test_validation_charter_keeps_draft_inactive(tmp_path: Path) -> None:
    frame = write_validation_charter_status(tmp_path, _config(tmp_path))

    active = frame.loc[frame["active_for_phase1_readiness"]]
    assert active["version"].tolist() == ["v3_legacy"]
    payload = json.loads((tmp_path / "validation_charter_status.json").read_text())
    assert payload["automatic_activation_allowed"] is False
    assert payload["official_gate_unchanged"] is True


def test_validation_charter_rejects_active_version_pointing_to_draft(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["validation"]["charter"]["active_version"] = "v4_draft"

    with pytest.raises(ValueError, match="draft charters cannot be activated"):
        write_validation_charter_status(tmp_path, config)


def test_frozen_manifest_hashes_artifacts_and_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    scope, entry = _fake_scope(run_dir)

    manifests, index = freeze_candidate_manifests(
        run_dir=run_dir,
        report_dir=tmp_path / "report",
        entries=[entry],
        config=_config(tmp_path),
    )

    assert bool(index.loc[0, "available"]) is True
    assert index.loc[0, "model_count"] == 1
    assert verify_frozen_manifest_artifacts(manifests[0], run_dir=run_dir) == []
    (scope / "model_fold_000.pt").write_bytes(b"tampered")
    errors = verify_frozen_manifest_artifacts(manifests[0], run_dir=run_dir)
    assert any("model_fold_000.pt" in error for error in errors)


def test_future_oos_waits_without_loading_models(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, min_rows=20)
    data_dir = Path(config["paths"]["data_dir"]) / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
            "feature": np.arange(30, dtype=float),
            "label": np.arange(30) % 2,
            "fwd_return_10h": np.linspace(-0.01, 0.01, 30),
        }
    ).to_parquet(data_dir / "labeled_1h.parquet", index=False)
    monkeypatch.setattr(
        future_oos_module,
        "_profile_predictions",
        lambda **_: (_ for _ in ()).throw(AssertionError("models must not load before readiness")),
    )

    evaluation, status = evaluate_future_oos(
        run_dir=tmp_path / "run",
        report_dir=tmp_path / "report",
        config=config,
        manifests=[],
    )

    assert evaluation.empty
    assert status["ready_for_evaluation"] is False
    assert status["fit_operations_performed"] == 0


def test_future_oos_scores_frozen_predictions_without_refit(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, min_rows=20)
    data_dir = Path(config["paths"]["data_dir"]) / "processed"
    data_dir.mkdir(parents=True)
    timestamps = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature": np.arange(80, dtype=float),
            "label": np.arange(80) % 2,
            "fwd_return_10h": np.where(np.arange(80) % 2, 0.01, -0.01),
        }
    ).to_parquet(data_dir / "labeled_1h.parquet", index=False)
    future = pd.DataFrame(
        {
            "timestamp": timestamps[timestamps > pd.Timestamp("2024-01-02", tz="UTC")],
            "prob_long": np.where(np.arange(len(timestamps[timestamps > pd.Timestamp("2024-01-02", tz="UTC")])) % 2, 0.9, 0.1),
            "label": np.arange(len(timestamps[timestamps > pd.Timestamp("2024-01-02", tz="UTC")])) % 2,
            "forward_return": np.where(
                np.arange(len(timestamps[timestamps > pd.Timestamp("2024-01-02", tz="UTC")])) % 2,
                0.01,
                -0.01,
            ),
        }
    )
    monkeypatch.setattr(future_oos_module, "verify_frozen_manifest_artifacts", lambda *_, **__: [])
    monkeypatch.setattr(
        future_oos_module,
        "_profile_predictions",
        lambda **_: {"control": future.copy()},
    )
    manifest = {
        "candidate_id": "control_v1",
        "candidate_type": "profile",
        "profiles": ["control"],
        "components": [{}],
        "available": True,
        "threshold": {"value": 0.55, "source": "validation_threshold"},
        "manifest_hash": "frozen",
    }

    evaluation, status = evaluate_future_oos(
        run_dir=tmp_path / "run",
        report_dir=tmp_path / "report",
        config=config,
        manifests=[manifest],
    )

    assert status["evaluation_completed"] is True
    assert status["fit_operations_performed"] == 0
    assert bool(evaluation.loc[0, "no_refit_verified"]) is True
    assert evaluation.loc[0, "rank_ic"] > 0.9


def test_registry_is_append_only_and_deduplicates_events(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    snapshot = tmp_path / "snapshot.jsonl"
    event = {"event_type": "diagnostics", "run_id": "run_1"}

    first = append_experiment_registry(
        registry_path=registry,
        snapshot_path=snapshot,
        event=event,
    )
    second = append_experiment_registry(
        registry_path=registry,
        snapshot_path=snapshot,
        event=event,
    )

    assert first["event_id"] == second["event_id"]
    assert len(registry.read_text(encoding="utf-8").splitlines()) == 1
    assert snapshot.read_text(encoding="utf-8") == registry.read_text(encoding="utf-8")
