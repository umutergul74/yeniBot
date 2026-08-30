from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yenibot.models.hybrid import HybridEncoder
from yenibot.phase2.forward_shadow import (
    LabelFreeSequenceDataset,
    append_shadow_predictions,
    build_shadow_scores,
    canonical_shadow_manifest_hash,
    empirical_cdf_right,
    fit_initial_shadow_payoff,
    load_shadow_manifest,
    plan_shadow_block,
    predict_label_free_model,
    read_shadow_ledger,
    seal_shadow_manifest,
    select_shadow_training_window,
    validate_shadow_manifest,
)
from yenibot.training.dataset import SequenceDataset
from yenibot.training.trainer import _predict_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ridge(*, coefficients=(0.002, 0.0, 0.0), intercept=-0.001):
    return {
        "alpha": 10.0,
        "center": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "coefficients": list(coefficients),
        "intercept": intercept,
        "fit_rows": 2000,
        "fit_target_mean": intercept,
    }


def _manifest(tmp_path: Path):
    artifacts = {}
    for name in ("model", "scaler", "hmm", "validation_cdf"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = {"path": path.name, "sha256": _sha(path)}
    features = ["f1", "f2"]
    feature_hash = hashlib.sha256(
        json.dumps(
            features,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return seal_shadow_manifest(
        {
            "manifest_version": "forward_shadow_block_manifest_v2",
            "process_id": "block_prequential_forward_shadow_v2",
            "candidate_id": "shadow-v2",
            "block": {
                "block_id": "block_046",
                "ordinal": 46,
                "locked_at_utc": "2026-08-30T00:30:00Z",
                "context_block_hours": 657,
                "sequence_burn_in_hours": 0,
                "evidence_hours": 657,
                "context_start_inclusive": "2026-08-30T02:00:00Z",
                "evidence_start_inclusive": "2026-08-30T02:00:00Z",
                "evidence_end_inclusive": "2026-09-26T10:00:00Z",
            },
            "model": {
                "profile": "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility",
                "feature_columns": features,
                "feature_columns_sha256": feature_hash,
            },
            "payoff_layer": {
                "candidate_fit": _ridge(),
                "atr_only_fit": _ridge(coefficients=(0.0, 0.1, 0.0), intercept=-0.0005),
            },
            "artifacts": artifacts,
        }
    )


def _raw_rows():
    timestamps = pd.date_range("2026-08-30T01:00:00Z", periods=4, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "raw_score": [0.2, 0.6, 0.8, 0.4],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "atr_14": [1.0, 1.1, 1.2, 1.3],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [-1.0, -2.0, -3.0, -4.0],
            "label": [0, 1, 0, 1],
            "forward_return": [-99.0, 99.0, -99.0, 99.0],
        }
    )


def _spec():
    return {
        "source_evidence": {
            "historical_confirmation_cutoff": "2025-12-31T00:00:00Z"
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
            "minimum_registration_lead_hours": 72,
            "block_ordinal_anchor": "2022-11-15T01:00:00Z",
        },
        "payoff_layer": {"ridge_alpha": 10.0, "minimum_fit_rows": 1000},
    }


def test_block_plan_is_grid_aligned_and_preserves_burn_in():
    spec = _spec()
    block = plan_shadow_block(spec, prepared_at="2026-08-30T05:15:00Z")
    anchor = pd.Timestamp(spec["model_schedule"]["block_ordinal_anchor"])
    context = pd.Timestamp(block["context_start_inclusive"])
    evidence = pd.Timestamp(block["evidence_start_inclusive"])
    end = pd.Timestamp(block["evidence_end_inclusive"])
    assert context >= pd.Timestamp("2026-09-02T05:15:00Z")
    assert context == anchor + pd.Timedelta(hours=720 * block["ordinal"])
    assert evidence - context == pd.Timedelta(hours=63)
    assert end - context == pd.Timedelta(hours=719)
    assert block["block_id"].startswith(
        f"shadow_v2_block_{block['ordinal']:04d}_"
    )


def test_training_window_is_latest_contiguous_and_has_exact_boundaries():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC"),
            "value": np.arange(30),
        }
    )
    selected, fold = select_shadow_training_window(
        frame, spec=_spec(), block_ordinal=47
    )
    assert len(selected) == 21
    assert selected.value.tolist() == list(range(9, 30))
    assert fold.fold == 47
    assert fold.train.tolist() == list(range(10))
    assert fold.val.tolist() == list(range(12, 17))
    assert fold.test.tolist() == [18, 19, 20]
    broken = frame.copy()
    broken.loc[20, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="not hourly contiguous"):
        select_shadow_training_window(broken, spec=_spec(), block_ordinal=47)


def test_initial_payoff_fit_uses_only_frozen_mature_oof_rows():
    rows = 1200
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    percentile = np.linspace(0, 1, rows)
    atr = 0.01 + np.arange(rows) / 1_000_000
    targets = pd.DataFrame(
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
    )
    fitted = fit_initial_shadow_payoff(targets, spec=_spec())
    assert fitted["candidate_fit"]["fit_rows"] == rows
    assert fitted["atr_only_fit"]["fit_rows"] == rows
    assert fitted["source_audit"]["current_or_future_shadow_outcomes_used"] is False
    assert len(fitted["source_audit"]["training_membership_sha256"]) == 64
    changed = targets.copy()
    changed.loc[0, "outcome_time_conservative"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="beyond its frozen cutoff"):
        fit_initial_shadow_payoff(changed, spec=_spec())


def test_manifest_hash_and_artifacts_fail_closed(tmp_path):
    manifest = _manifest(tmp_path)
    checked = validate_shadow_manifest(manifest, artifact_root=tmp_path)
    assert checked["integrity_audit"]["verified_artifact_count"] == 4
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_shadow_manifest(path)["manifest_hash"] == canonical_shadow_manifest_hash(
        manifest
    )

    changed = json.loads(json.dumps(manifest))
    changed["block"]["ordinal"] = 47
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_shadow_manifest(changed)
    (tmp_path / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        validate_shadow_manifest(manifest, artifact_root=tmp_path)


def test_manifest_rejects_artifact_escape_even_with_matching_hash(tmp_path):
    outside = tmp_path.parent / "outside-forward-shadow.bin"
    outside.write_bytes(b"outside")
    manifest = _manifest(tmp_path)
    manifest.pop("manifest_hash")
    manifest["artifacts"]["model"] = {
        "path": "../outside-forward-shadow.bin",
        "sha256": _sha(outside),
    }
    manifest = seal_shadow_manifest(manifest)
    with pytest.raises(ValueError, match="escapes"):
        validate_shadow_manifest(manifest, artifact_root=tmp_path)


def test_empirical_cdf_is_right_sided_and_strictly_validated():
    reference = np.linspace(0, 1, 1000)
    result = empirical_cdf_right(reference, np.array([0.0, 0.5, 1.0]))
    assert result[0] == pytest.approx(0.001)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match=">=1000"):
        empirical_cdf_right(reference[:-1], np.array([0.5]))
    with pytest.raises(ValueError, match=">=1000"):
        empirical_cdf_right(reference, np.array([1.1]))


def test_label_free_dataset_excludes_sequences_crossing_gaps():
    times = pd.to_datetime(
        [
            "2026-01-01T00:00Z",
            "2026-01-01T01:00Z",
            "2026-01-01T02:00Z",
            "2026-01-01T04:00Z",
            "2026-01-01T05:00Z",
            "2026-01-01T06:00Z",
        ]
    )
    dataset = LabelFreeSequenceDataset(
        np.arange(12).reshape(6, 2), seq_len=3, timestamps=times
    )
    assert dataset.end_positions.tolist() == [2, 5]
    features, position = dataset[1]
    assert features.shape == (3, 2)
    assert position.item() == 5


def test_label_free_model_prediction_matches_labeled_trainer_path():
    torch.manual_seed(7)
    model = HybridEncoder(
        2,
        seq_len=4,
        tcn_channels=4,
        tcn_kernel_size=2,
        tcn_dilations=[1, 2],
        gru_hidden=3,
        gru_layers=1,
        dropout=0.0,
        fusion_hidden=4,
    )
    features = np.arange(20, dtype=float).reshape(10, 2) / 20
    times = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    source = pd.DataFrame(
        {"timestamp": times, "label": np.arange(10) % 2, "fwd_return_10h": 0.0}
    )
    label_free = LabelFreeSequenceDataset(features, seq_len=4, timestamps=times)
    observed = predict_label_free_model(
        model, label_free, source[["timestamp"]], batch_size=3, device="cpu"
    )
    labeled = SequenceDataset(
        features,
        source.label.to_numpy(),
        source.fwd_return_10h.to_numpy(),
        seq_len=4,
        timestamps=times,
    )
    expected = _predict_dataset(
        model, labeled, source, batch_size=3, device=torch.device("cpu")
    )
    assert observed.source_row_position.tolist() == expected.source_row_position.tolist()
    assert np.array_equal(observed.raw_score.to_numpy(), expected.prob_long.to_numpy())


def test_shadow_scores_ignore_outcomes_and_reconstruct_actions(tmp_path):
    manifest = _manifest(tmp_path)
    raw = _raw_rows()
    reference = np.linspace(0, 1, 1000)
    first = build_shadow_scores(
        raw,
        manifest=manifest,
        cdf_reference=reference,
        generated_at="2026-08-30T05:10:00Z",
    )
    changed = raw.copy()
    changed[["label", "forward_return"]] = [[1, -1e9], [0, 1e9], [1, -1e9], [0, 1e9]]
    second = build_shadow_scores(
        changed,
        manifest=manifest,
        cdf_reference=reference,
        generated_at="2026-08-30T05:10:00Z",
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.candidate_action.equals(first.predicted_candidate_utility.gt(0))
    assert first.atr_action.equals(first.predicted_atr_utility.gt(0))
    assert first.candidate_score.gt(0.5).equals(first.candidate_action)
    assert first.atr_score.gt(0.5).equals(first.atr_action)
    assert first.feature_snapshot_sha256.str.fullmatch(r"[0-9a-f]{64}").all()


def test_shadow_ledger_is_hash_chained_idempotent_and_marks_latency(tmp_path):
    manifest = _manifest(tmp_path)
    scores = build_shadow_scores(
        _raw_rows(),
        manifest=manifest,
        cdf_reference=np.linspace(0, 1, 1000),
        generated_at="2026-08-30T05:20:00Z",
    )
    scores.loc[0, "generated_at"] = scores.loc[0, "decision_time"] + pd.Timedelta(minutes=5)
    path = tmp_path / "ledger.jsonl"
    first = append_shadow_predictions(path, scores.iloc[:2], manifest=manifest)
    assert first["appended_rows"] == 2
    assert append_shadow_predictions(path, scores, manifest=manifest)["appended_rows"] == 2
    assert append_shadow_predictions(path, scores, manifest=manifest)["appended_rows"] == 0
    ledger, head = read_shadow_ledger(path, with_hash=True)
    assert len(ledger) == 4 and len(head) == 64
    assert ledger.evidence_role.tolist()[0] == "timely_shadow"
    assert ledger.evidence_role.tolist()[1:] == ["sealed_batch_replay"] * 3

    changed = scores.copy()
    changed.loc[0, "raw_score"] += 0.01
    with pytest.raises(ValueError, match="changed"):
        append_shadow_predictions(path, changed, manifest=manifest)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"raw_score": 0.2', '"raw_score": 0.21', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        read_shadow_ledger(path)


def test_shadow_ledger_rejects_prelock_gap_and_identity_changes(tmp_path):
    manifest = _manifest(tmp_path)
    scores = build_shadow_scores(
        _raw_rows(),
        manifest=manifest,
        cdf_reference=np.linspace(0, 1, 1000),
        generated_at="2026-08-30T06:00:00Z",
    )
    prelock = scores.iloc[[0]].copy()
    prelock["timestamp"] = pd.Timestamp("2026-08-30T00:00:00Z")
    prelock["decision_time"] = pd.Timestamp("2026-08-30T01:00:00Z")
    with pytest.raises(ValueError, match="Pre-lock"):
        append_shadow_predictions(tmp_path / "prelock.jsonl", prelock, manifest=manifest)

    gapped = scores.iloc[[0, 2]].copy()
    with pytest.raises(ValueError, match="contiguous"):
        append_shadow_predictions(tmp_path / "gap.jsonl", gapped, manifest=manifest)

    wrong = scores.copy()
    wrong["block_id"] = "another"
    with pytest.raises(ValueError, match="identity mismatch"):
        append_shadow_predictions(tmp_path / "wrong.jsonl", wrong, manifest=manifest)
