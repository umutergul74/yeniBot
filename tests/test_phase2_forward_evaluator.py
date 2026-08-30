from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from yenibot.phase2.forward_evaluator import (
    circular_block_mean_interval,
    evaluate_complete_shadow_block,
    rank_ic,
    shadow_block_preflight,
)
from yenibot.phase2.forward_shadow import (
    seal_shadow_manifest,
    seal_shadow_registration,
)


def _ridge():
    return {
        "alpha": 10.0,
        "center": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "coefficients": [0.01, 0.0, 0.0],
        "intercept": -0.005,
        "fit_rows": 1200,
        "fit_target_mean": 0.0,
    }


def _inputs():
    features = ["f1"]
    feature_hash = hashlib.sha256(
        json.dumps(
            features,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    manifest = seal_shadow_manifest(
        {
            "manifest_version": "forward_shadow_block_manifest_v2",
            "process_id": "block_prequential_forward_shadow_v2",
            "candidate_id": "candidate",
            "block": {
                "block_id": "block_1",
                "ordinal": 1,
                "locked_at_utc": "2026-01-01T00:00:00Z",
                "context_block_hours": 5,
                "sequence_burn_in_hours": 0,
                "evidence_hours": 5,
                "context_start_inclusive": "2026-01-01T02:00:00Z",
                "evidence_start_inclusive": "2026-01-01T02:00:00Z",
                "evidence_end_inclusive": "2026-01-01T06:00:00Z",
            },
            "model": {
                "profile": "profile",
                "feature_columns": features,
                "feature_columns_sha256": feature_hash,
            },
            "payoff_layer": {
                "candidate_fit": _ridge(),
                "atr_only_fit": _ridge(),
            },
            "artifacts": {
                key: {"path": f"{key}.bin", "sha256": "a" * 64}
                for key in ("model", "scaler", "hmm", "validation_cdf")
            },
        }
    )
    registration = seal_shadow_registration(
        {
            "registration_version": "forward_shadow_registration_v2",
            "process_id": manifest["process_id"],
            "candidate_id": manifest["candidate_id"],
            "block_id": manifest["block"]["block_id"],
            "manifest_hash": manifest["manifest_hash"],
            "manifest_git_commit": "b" * 40,
            "manifest_registry_path": "registry/block_1.json",
            "registered_at_utc": "2026-01-01T01:00:00Z",
        }
    )
    decisions = pd.date_range("2026-01-01T02:00:00Z", periods=5, freq="h")
    ledger = pd.DataFrame(
        {
            "decision_time": decisions,
            "evidence_role": "timely_shadow",
            "process_id": manifest["process_id"],
            "candidate_id": manifest["candidate_id"],
            "block_id": manifest["block"]["block_id"],
            "block_manifest_hash": manifest["manifest_hash"],
            "candidate_score": [0.7, 0.8, 0.4, 0.9, 0.6],
            "atr_score": [0.4, 0.6, 0.4, 0.6, 0.4],
            "score_percentile": [0.2, 0.4, 0.1, 0.8, 0.6],
        }
    )
    timestamps = pd.date_range("2026-01-01T01:00:00Z", periods=15, freq="h")
    opening = 100 + np.arange(15) * 0.5
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opening,
            "high": opening + 1.0,
            "low": opening - 0.5,
            "close": opening + 0.4,
            "atr_14": 1.0,
        }
    )
    return manifest, registration, ledger, market


def test_complete_block_preflight_rejects_even_one_replay_or_missing_outcome_bar():
    manifest, registration, ledger, market = _inputs()
    ready = shadow_block_preflight(
        ledger, market, manifest=manifest, registration=registration
    )
    assert ready["ready_for_complete_block_evaluation"] is True
    late = ledger.copy()
    late.loc[2, "evidence_role"] = "sealed_batch_replay"
    failed = shadow_block_preflight(
        late, market, manifest=manifest, registration=registration
    )
    assert failed["ready_for_complete_block_evaluation"] is False
    assert "late_or_replayed_rows_present" in failed["failed_checks"]
    missing = shadow_block_preflight(
        ledger, market.iloc[:-1], manifest=manifest, registration=registration
    )
    assert "mature_outcome_market_incomplete" in missing["failed_checks"]


def test_complete_block_evaluation_uses_same_cohort_and_mtm_engine():
    manifest, registration, ledger, market = _inputs()
    result = evaluate_complete_shadow_block(
        ledger, market, manifest=manifest, registration=registration
    )
    assert result.report["status"] == "complete_forward_shadow_block_evaluated"
    assert set(result.report["policies"]) == {
        "candidate_base",
        "candidate_adverse",
        "atr_only_base",
        "atr_only_adverse",
    }
    assert result.report["opportunity_rows"] == 5
    assert len(result.report["market_membership_sha256"]) == 64
    assert result.report["promotion_allowed"] is False
    assert result.report["live_trading_allowed"] is False


def test_rank_ic_and_circular_block_interval_are_deterministic():
    assert rank_ic(np.arange(10), np.arange(10)) == pytest.approx(1.0)
    first = circular_block_mean_interval(
        np.array([0.01, -0.01, 0.02, 0.03]),
        block_length=2,
        replicates=500,
        seed=7,
    )
    second = circular_block_mean_interval(
        np.array([0.01, -0.01, 0.02, 0.03]),
        block_length=2,
        replicates=500,
        seed=7,
    )
    assert first == second
    assert first["lower95"] <= first["actual_mean"] <= first["upper95"]
