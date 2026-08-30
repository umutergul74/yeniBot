from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yenibot.automation import phase2_forward_shadow_prepare as prepare
from yenibot.phase2.forward_shadow import load_shadow_manifest
from yenibot.phase2.full_oof import file_sha256


def _targets(path: Path) -> None:
    rows = 1200
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    percentile = np.linspace(0, 1, rows)
    atr = 0.01 + np.arange(rows) / 1_000_000
    pd.DataFrame(
        {
            "decision_time": decision,
            "outcome_time_conservative": decision + pd.Timedelta(hours=10),
            "source_split": "test",
            "fit_eligible": True,
            "adverse_net_target": 0.002 * percentile - 0.0005,
            "frozen_score_percentile": percentile,
            "decision_atr_close_fraction": atr,
            "score_atr_product": percentile * atr,
        }
    ).to_csv(path, index=False)


def test_prepare_publishes_only_after_saved_artifact_parity(tmp_path, monkeypatch):
    labeled_path = tmp_path / "labeled.parquet"
    rows = 30
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-01", periods=rows, freq="h", tz="UTC"),
            "label": np.arange(rows) % 2,
            "fwd_return_10h": np.linspace(-0.01, 0.01, rows),
            "f1": np.linspace(0, 1, rows),
        }
    ).to_parquet(labeled_path, index=False)
    targets_path = tmp_path / "targets.csv"
    _targets(targets_path)
    spec = {
        "version": "block_prequential_forward_shadow_v2",
        "profile": "profile",
        "source_evidence": {
            "initial_oof_targets_sha256": file_sha256(targets_path),
            "historical_confirmation_cutoff": "2025-12-31T00:00:00Z",
        },
        "model_schedule": {
            "block_hours": 720,
            "evidence_hours_per_block": 657,
            "sequence_burn_in_hours_per_block": 63,
            "train_bars": 10,
            "purge_bars": 2,
            "validation_bars": 5,
            "embargo_bars": 1,
            "post_fit_audit_bars": 3,
            "minimum_label_maturity_hours": 10,
            "minimum_registration_lead_hours": 72,
            "minimum_lock_to_context_hours": 24,
            "block_ordinal_anchor": "2022-11-15T01:00:00Z",
        },
        "score_transform": {"minimum_reference_rows": 2},
        "payoff_layer": {"ridge_alpha": 10.0, "minimum_fit_rows": 1000},
    }
    (tmp_path / "config.yaml").write_text("project: test\n", encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(prepare, "_canonical_spec", lambda _path: (spec, "c" * 64))
    monkeypatch.setattr(
        prepare,
        "_git_identity",
        lambda _repo: {
            "commit": "a" * 40,
            "branch": "codex/test",
            "tracked_tree_clean": True,
        },
    )
    monkeypatch.setattr(
        prepare,
        "load_config",
        lambda _path: {
            "project": {"random_seed": 42},
            "labeling": {"max_holding_bars": 10},
        },
    )
    monkeypatch.setattr(prepare, "profile_config", lambda config, _profile: config)
    monkeypatch.setattr(prepare, "select_feature_columns", lambda _frame: ["f1"])
    monkeypatch.setattr(
        prepare, "filter_feature_columns", lambda columns, _config: columns
    )

    def fake_train(frame, fold, feature_columns, config, *, checkpoint_dir, device):
        root = Path(checkpoint_dir)
        for pattern, content in (
            (f"model_fold_{fold.fold:03d}.pt", b"model"),
            (f"scaler_fold_{fold.fold:03d}.pkl", b"scaler"),
            (f"hmm_fold_{fold.fold:03d}.pkl", b"hmm"),
        ):
            (root / pattern).write_bytes(content)
        val_times = frame.iloc[fold.val].timestamp.iloc[-2:].reset_index(drop=True)
        test_time = frame.iloc[fold.test].timestamp.iloc[-1]
        predictions = pd.DataFrame(
            {
                "timestamp": [*val_times, test_time],
                "prob_long": [0.2, 0.8, 0.6],
                "split": ["val", "val", "test"],
            }
        )
        return {
            "predictions": predictions,
            "val_metrics": {"rank_ic": 0.1},
            "test_metrics": {"rank_ic": np.nan},
            "history": pd.DataFrame({"epoch": [1], "rank_ic": [0.1]}),
        }

    monkeypatch.setattr(prepare, "train_one_fold", fake_train)

    def fake_label_free(frame, **_kwargs):
        return pd.DataFrame(
            {"timestamp": [frame.timestamp.iloc[-1]], "raw_score": [0.6]}
        )

    monkeypatch.setattr(prepare, "predict_label_free_artifacts", fake_label_free)
    output = tmp_path / "published"
    report = prepare.prepare_forward_shadow(
        labeled_path=labeled_path,
        oof_targets_path=targets_path,
        output_dir=output,
        config_path=tmp_path / "config.yaml",
        spec_path=tmp_path / "spec.json",
        repo_dir=tmp_path,
        device="cpu",
    )
    assert report["confirmation_clock_started"] is False
    assert output.is_dir()
    assert not list(tmp_path.glob(".published.staging-*"))
    manifest = load_shadow_manifest(
        output / "forward_shadow_manifest.json", artifact_root=output
    )
    assert manifest["post_fit_parity"]["exact_score_match"] is True
    assert manifest["training_audit"]["post_fit_audit_metrics"]["rank_ic"] is None
    assert manifest["registration_status"] == (
        "sealed_manifest_awaiting_git_registration"
    )
    assert json.loads((output / "preparation_report.json").read_text())["status"] == (
        "sealed_manifest_awaiting_git_registration"
    )
