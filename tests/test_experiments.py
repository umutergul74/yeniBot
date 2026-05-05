from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from yenibot.config import load_config
from yenibot.experiments import (
    experiment_settings,
    profile_config,
    resolve_experiment_run_id,
    run_experiment_matrix,
    run_profile_experiment,
    write_experiment_diagnostics,
)
from yenibot.features import build_feature_matrix, filter_feature_columns, select_feature_columns


def _labeled_frame(synthetic_klines, config: dict, *, periods: int = 220) -> tuple[pd.DataFrame, list[str]]:
    primary = synthetic_klines(periods, "1h")
    htf = synthetic_klines(max(70, periods // 3), "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    return frame, features.feature_columns


def test_profile_config_overrides_active_profile_without_mutating_source(tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["active_profile"] = "base"
    config["features"]["profiles"] = {
        "base": {"include_patterns": ["*gk_vol*"], "exclude_patterns": []},
        "candidate": {"include_patterns": ["*true_cvd*"], "exclude_patterns": []},
    }

    updated = profile_config(config, "candidate")

    assert config["features"]["active_profile"] == "base"
    assert updated["features"]["active_profile"] == "candidate"


def test_experiment_settings_resolves_control_and_candidates() -> None:
    config = {
        "features": {"active_profile": "fallback"},
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["candidate_a", "candidate_a", "candidate_b"],
        },
    }

    settings = experiment_settings(config)

    assert settings["control_profile"] == "control"
    assert settings["profiles"] == ["control", "candidate_a", "candidate_b"]
    assert settings["candidate_profiles"] == ["candidate_a", "candidate_b"]


def test_repo_experiment_profiles_keep_default_baseline_and_candidate_boundaries() -> None:
    config = load_config("config.yaml")
    assert config["features"]["active_profile"] == "baseline_plus_4h_bounded_whale_no_4h_tier1"
    assert config["experiments"]["full_cv_profiles"] == [
        "baseline_plus_4h_bounded_whale_no_4h_tier1",
        "baseline_no_4h_tier1_4h_large_trade_pressure_stable",
        "baseline_no_4h_tier1_flow_stable_combo",
        "baseline_no_4h_tier1_4h_cvd_pressure_stable",
    ]
    columns = [
        "4h_large_trade_ratio",
        "4h_vpt_zscore",
        "4h_vol_per_trade_log_zscore",
        "4h_cvd_pressure_3",
        "4h_cvd_pressure_3_stable_rank",
        "4h_cvd_pressure_24_stable_zscore",
        "4h_signed_large_trade_pressure_stable_rank",
        "4h_large_trade_pressure_12_stable_zscore",
        "4h_gk_vol_14_stable_rank",
        "4h_vwap_dist_atr_stable_zscore",
        "taker_buy_ratio",
    ]

    pruned = profile_config(config, "baseline_no_4h_tier1_pruned_whale")
    assert "4h_large_trade_ratio" not in filter_feature_columns(columns, pruned)
    assert "4h_vpt_zscore" in filter_feature_columns(columns, pruned)

    cvd = profile_config(config, "baseline_no_4h_tier1_4h_cvd_pressure_stable")
    cvd_columns = filter_feature_columns(columns, cvd)
    assert "4h_cvd_pressure_3" not in cvd_columns
    assert "4h_cvd_pressure_3_stable_rank" in cvd_columns
    assert "4h_gk_vol_14_stable_rank" not in cvd_columns

    large = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_stable")
    large_columns = filter_feature_columns(columns, large)
    assert "4h_signed_large_trade_pressure_stable_rank" in large_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in large_columns
    assert "4h_cvd_pressure_3_stable_rank" not in large_columns


def test_profile_experiment_writes_isolated_outputs_and_resumes(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["active_profile"] = "base"
    config["features"]["profiles"] = {"base": {"include_patterns": ["*"], "exclude_patterns": []}}
    frame, feature_columns = _labeled_frame(synthetic_klines, config, periods=220)
    assert feature_columns

    first = run_profile_experiment(
        frame,
        config,
        profile="base",
        checkpoint_dir=tmp_path,
        run_id="run_a",
        fold_scope="triage",
        fold_ids=[0],
        device="cpu",
    )
    second = run_profile_experiment(
        frame,
        config,
        profile="base",
        checkpoint_dir=tmp_path,
        run_id="run_a",
        fold_scope="triage",
        fold_ids=[0],
        device="cpu",
    )

    assert not first["skipped"]
    assert second["skipped"]
    assert first["output_dir"] != tmp_path
    assert (first["output_dir"] / "predictions_all.parquet").exists()
    assert (first["output_dir"] / "training_manifest.json").exists()


def test_experiment_matrix_and_diagnostics_write_profile_comparison(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_buy_ratio"]},
        "candidate": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_sell_ratio"]},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": ["candidate"],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    result = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="matrix", device="cpu")
    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="matrix",
    )

    assert set(result["comparison"]["profile"]) == {"control", "candidate"}
    assert (tmp_path / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert diagnostics["zip_paths"]
    assert (tmp_path / "reports" / "phase1_experiment_bundle_matrix.zip").exists()
    assert (tmp_path / "reports" / "phase1_latest_experiment_bundle.zip").exists()
    assert diagnostics["bundle_zip"].endswith("phase1_experiment_bundle_matrix.zip")
    assert diagnostics["latest_bundle_zip"].endswith("phase1_latest_experiment_bundle.zip")
    assert diagnostics["decision"]["recommendation"] in {"keep_control_profile", "promote_best_candidate"}


def test_experiment_run_id_reuses_latest_matching_signature(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": []},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    first = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="stable_run", device="cpu")
    run_id, source = resolve_experiment_run_id(tmp_path, config)

    assert first["run_id"] == "stable_run"
    assert run_id == "stable_run"
    assert source == "matching_existing"
