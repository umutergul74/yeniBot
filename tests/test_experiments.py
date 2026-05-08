from __future__ import annotations

import copy
import zipfile

import numpy as np
import pandas as pd
import pytest

from yenibot.config import load_config
from yenibot.experiments import (
    _auto_full_profiles,
    _passes_full,
    _passes_triage,
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


def test_auto_full_profiles_keeps_control_and_promotes_best_triage_candidates() -> None:
    settings = {
        "control_profile": "control",
        "always_full_profiles": ["control", "champion"],
        "max_auto_full_candidates": 1,
    }
    triage_rows = [
        {"profile": "control", "promotable": False, "mean_rank_ic": 0.03, "top_10_lift_global": 1.0},
        {"profile": "weak", "promotable": False, "mean_rank_ic": 0.09, "top_10_lift_global": 1.2},
        {"profile": "candidate_a", "promotable": True, "mean_rank_ic": 0.05, "top_10_lift_global": 1.4},
        {"profile": "candidate_b", "promotable": True, "mean_rank_ic": 0.07, "top_10_lift_global": 1.1},
    ]

    assert _auto_full_profiles(settings, triage_rows) == ["control", "champion", "candidate_b"]


def test_repo_experiment_profiles_keep_default_baseline_and_candidate_boundaries() -> None:
    config = load_config("config.yaml")
    assert config["features"]["active_profile"] == "baseline_plus_4h_bounded_whale_no_4h_tier1"
    assert config["experiments"]["control_profile"] == "baseline_plus_4h_bounded_whale_no_4h_tier1"
    assert config["experiments"]["full_cv_profiles"] == "auto"
    assert config["experiments"]["always_full_profiles"] == [
        "baseline_plus_4h_bounded_whale_no_4h_tier1",
        "baseline_no_4h_tier1_4h_large_trade_pressure_long",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility",
    ]
    assert config["experiments"]["max_auto_full_candidates"] == 2
    assert config["experiments"]["candidate_profiles"] == [
        "baseline_no_4h_tier1_4h_large_trade_pressure_long",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_slow_4h_bounded_flow",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_bounded_flow",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_large_trade_ratio",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_whale_zscores",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_1h_cvd_rate",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_bad_fold_guardrail_light",
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_bounded_flow",
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_bad_fold_guardrail_light",
    ]
    assert config["experiments"]["seed_audit"]["enabled"] is True
    assert config["experiments"]["seed_audit"]["profiles"] == ["baseline_plus_4h_bounded_whale_no_4h_tier1"]
    assert config["experiments"]["seed_audit"]["seeds"] == [42, 43, 44]
    assert {0, 2, 4, 8, 17, 21, 32, 39}.issubset(set(config["experiments"]["triage_fold_ids"]))
    columns = [
        "4h_large_trade_ratio",
        "4h_vpt_zscore",
        "4h_vol_per_trade_log_zscore",
        "4h_cvd_pressure_3",
        "4h_cvd_pressure_3_stable_rank",
        "4h_cvd_pressure_24_stable_zscore",
        "4h_signed_large_trade_pressure_stable_zscore",
        "4h_signed_large_trade_pressure_stable_rank",
        "4h_large_trade_pressure_3_stable_zscore",
        "4h_large_trade_pressure_3_stable_rank",
        "4h_large_trade_pressure_6_stable_zscore",
        "4h_large_trade_pressure_6_stable_rank",
        "4h_large_trade_pressure_12_stable_zscore",
        "4h_large_trade_pressure_12_stable_rank",
        "4h_large_trade_pressure_12_stable_tanh",
        "4h_large_trade_pressure_24_stable_zscore",
        "4h_large_trade_pressure_24_stable_rank",
        "4h_large_trade_pressure_24_stable_tanh",
        "4h_large_trade_pressure_24_minus_12_stable_zscore",
        "4h_large_trade_pressure_24_minus_12_stable_rank",
        "4h_large_trade_pressure_24_minus_12_stable_tanh",
        "4h_signed_large_trade_pressure_stable_tanh",
        "4h_gk_vol_14_stable_rank",
        "4h_gk_vol_14_stable_zscore",
        "4h_realized_vol_14_stable_rank",
        "4h_realized_vol_14_stable_zscore",
        "4h_atr_14_pct_stable_rank",
        "4h_atr_14_pct_stable_zscore",
        "4h_adx_14_stable_rank",
        "4h_vwap_dist_atr_stable_zscore",
        "4h_volume_log_zscore_stable_rank",
        "4h_gk_vol_14",
        "4h_realized_vol_14",
        "4h_atr_14_pct",
        "4h_adx_14",
        "4h_log_return",
        "4h_close_denoised_log_return",
        "4h_volume_log_zscore",
        "4h_volume_denoised_log_zscore",
        "4h_vwap_dist_atr",
        "4h_taker_imbalance",
        "4h_taker_buy_ratio_zscore",
        "4h_taker_buy_ratio_delta",
        "4h_taker_imbalance_slope",
        "4h_taker_imbalance_mean_12",
        "4h_taker_imbalance_mean_24",
        "4h_whale_buy_flag",
        "4h_whale_sell_flag",
        "volume_log_zscore",
        "volume_denoised_log_zscore",
        "taker_buy_ratio",
        "cvd_cumulative_rate_norm",
    ]

    pruned = profile_config(config, "baseline_no_4h_tier1_pruned_whale")
    assert "4h_large_trade_ratio" not in filter_feature_columns(columns, pruned)
    assert "4h_vpt_zscore" in filter_feature_columns(columns, pruned)

    base_no_slow = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_slow_4h_bounded_flow")
    base_no_slow_columns = filter_feature_columns(columns, base_no_slow)
    assert "4h_taker_imbalance_mean_12" not in base_no_slow_columns
    assert "4h_taker_imbalance_mean_24" not in base_no_slow_columns
    assert "4h_taker_imbalance" in base_no_slow_columns

    base_no_bounded = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_bounded_flow")
    base_no_bounded_columns = filter_feature_columns(columns, base_no_bounded)
    assert "4h_taker_imbalance" not in base_no_bounded_columns
    assert "4h_taker_imbalance_mean_24" not in base_no_bounded_columns
    assert "4h_large_trade_ratio" in base_no_bounded_columns

    base_no_ratio = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_large_trade_ratio")
    assert "4h_large_trade_ratio" not in filter_feature_columns(columns, base_no_ratio)

    base_no_whale_zscores = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_whale_zscores")
    base_no_whale_zscore_columns = filter_feature_columns(columns, base_no_whale_zscores)
    assert "4h_vpt_zscore" not in base_no_whale_zscore_columns
    assert "4h_vol_per_trade_log_zscore" not in base_no_whale_zscore_columns
    assert "4h_large_trade_ratio" in base_no_whale_zscore_columns

    base_no_volatility = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility")
    base_no_volatility_columns = filter_feature_columns(columns, base_no_volatility)
    assert "4h_gk_vol_14" not in base_no_volatility_columns
    assert "4h_atr_14_pct" not in base_no_volatility_columns

    base_no_cvd_rate = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_1h_cvd_rate")
    base_no_cvd_rate_columns = filter_feature_columns(columns, base_no_cvd_rate)
    assert "cvd_cumulative_rate_norm" not in base_no_cvd_rate_columns

    base_guardrail = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_bad_fold_guardrail_light")
    base_guardrail_columns = filter_feature_columns(columns, base_guardrail)
    assert "4h_taker_imbalance_mean_24" not in base_guardrail_columns
    assert "4h_large_trade_ratio" not in base_guardrail_columns
    assert "4h_gk_vol_14" not in base_guardrail_columns
    assert "cvd_cumulative_rate_norm" not in base_guardrail_columns

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

    pruned_large = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_stable_pruned_whale")
    pruned_large_columns = filter_feature_columns(columns, pruned_large)
    assert "4h_large_trade_ratio" not in pruned_large_columns
    assert "4h_signed_large_trade_pressure_stable_rank" in pruned_large_columns

    rank_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_rank_only")
    rank_columns = filter_feature_columns(columns, rank_only)
    assert "4h_large_trade_pressure_12_stable_rank" in rank_columns
    assert "4h_large_trade_pressure_12_stable_zscore" not in rank_columns

    zscore_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_zscore_only")
    zscore_columns = filter_feature_columns(columns, zscore_only)
    assert "4h_large_trade_pressure_12_stable_zscore" in zscore_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in zscore_columns

    short = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_short")
    short_columns = filter_feature_columns(columns, short)
    assert "4h_large_trade_pressure_3_stable_rank" in short_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in short_columns

    long = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long")
    long_columns = filter_feature_columns(columns, long)
    assert "4h_large_trade_pressure_24_stable_rank" in long_columns
    assert "4h_large_trade_pressure_6_stable_rank" not in long_columns

    no_ratio = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_large_trade_ratio")
    no_ratio_columns = filter_feature_columns(columns, no_ratio)
    assert "4h_large_trade_ratio" not in no_ratio_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_ratio_columns

    no_whale_flags = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_whale_flags")
    no_whale_columns = filter_feature_columns(columns, no_whale_flags)
    assert "4h_large_trade_ratio" not in no_whale_columns
    assert "4h_whale_buy_flag" not in no_whale_columns
    assert "4h_vpt_zscore" in no_whale_columns

    no_structure = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_structure")
    no_structure_columns = filter_feature_columns(columns, no_structure)
    assert "4h_gk_vol_14" not in no_structure_columns
    assert "4h_vwap_dist_atr" not in no_structure_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_structure_columns

    stable_structure = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_4h_structure_stable_overlay",
    )
    stable_structure_columns = filter_feature_columns(columns, stable_structure)
    assert "4h_gk_vol_14" not in stable_structure_columns
    assert "4h_gk_vol_14_stable_rank" in stable_structure_columns
    assert "4h_vwap_dist_atr_stable_zscore" in stable_structure_columns

    no_4h_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_volume_context")
    no_4h_volume_columns = filter_feature_columns(columns, no_4h_volume)
    assert "4h_volume_log_zscore" not in no_4h_volume_columns
    assert "volume_log_zscore" in no_4h_volume_columns
    assert "4h_vwap_dist_atr" in no_4h_volume_columns

    no_1h_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_1h_volume_context")
    no_1h_volume_columns = filter_feature_columns(columns, no_1h_volume)
    assert "volume_log_zscore" not in no_1h_volume_columns
    assert "4h_volume_log_zscore" in no_1h_volume_columns

    no_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_volume_context")
    no_volume_columns = filter_feature_columns(columns, no_volume)
    assert "volume_log_zscore" not in no_volume_columns
    assert "4h_volume_log_zscore" not in no_volume_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_volume_columns

    no_4h_volatility = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility")
    no_4h_volatility_columns = filter_feature_columns(columns, no_4h_volatility)
    assert "4h_gk_vol_14" not in no_4h_volatility_columns
    assert "4h_atr_14_pct" not in no_4h_volatility_columns
    assert "4h_vwap_dist_atr" in no_4h_volatility_columns

    vwap_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_4h_vwap_only_structure")
    vwap_only_columns = filter_feature_columns(columns, vwap_only)
    assert "4h_vwap_dist_atr" in vwap_only_columns
    assert "4h_gk_vol_14" not in vwap_only_columns
    assert "4h_adx_14" not in vwap_only_columns

    no_bounded_flow = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_bounded_flow")
    no_bounded_flow_columns = filter_feature_columns(columns, no_bounded_flow)
    assert "4h_taker_imbalance_mean_24" not in no_bounded_flow_columns
    assert "4h_taker_buy_ratio_zscore" not in no_bounded_flow_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_bounded_flow_columns
    assert "4h_large_trade_ratio" in no_bounded_flow_columns

    stable_vol = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_4h_stable_vol_overlay",
    )
    stable_vol_columns = filter_feature_columns(columns, stable_vol)
    assert "4h_gk_vol_14" not in stable_vol_columns
    assert "4h_gk_vol_14_stable_rank" in stable_vol_columns
    assert "4h_realized_vol_14_stable_zscore" in stable_vol_columns
    assert "4h_adx_14_stable_rank" not in stable_vol_columns
    assert "4h_volume_log_zscore_stable_rank" not in stable_vol_columns

    no_whale_zscores = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_whale_zscores")
    no_whale_zscore_columns = filter_feature_columns(columns, no_whale_zscores)
    assert "4h_vpt_zscore" not in no_whale_zscore_columns
    assert "4h_vol_per_trade_log_zscore" not in no_whale_zscore_columns
    assert "4h_large_trade_ratio" in no_whale_zscore_columns
    assert "4h_whale_buy_flag" in no_whale_zscore_columns

    pressure_24 = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_24_only")
    pressure_24_columns = filter_feature_columns(columns, pressure_24)
    assert "4h_large_trade_pressure_12_stable_rank" not in pressure_24_columns
    assert "4h_large_trade_pressure_24_stable_rank" in pressure_24_columns

    no_12_zscore = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_12_pressure_zscore")
    no_12_zscore_columns = filter_feature_columns(columns, no_12_zscore)
    assert "4h_large_trade_pressure_12_stable_zscore" not in no_12_zscore_columns
    assert "4h_large_trade_pressure_12_stable_rank" in no_12_zscore_columns
    assert "4h_large_trade_pressure_24_stable_zscore" in no_12_zscore_columns

    no_12_rank = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_12_pressure_rank")
    no_12_rank_columns = filter_feature_columns(columns, no_12_rank)
    assert "4h_large_trade_pressure_12_stable_rank" not in no_12_rank_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in no_12_rank_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_12_rank_columns

    tanh_replacement = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_12_pressure_tanh_replacement",
    )
    tanh_replacement_columns = filter_feature_columns(columns, tanh_replacement)
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_replacement_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_replacement_columns
    assert "4h_large_trade_pressure_12_stable_rank" in tanh_replacement_columns
    assert "4h_large_trade_pressure_24_stable_zscore" in tanh_replacement_columns

    tanh_pair = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_tanh_pair")
    tanh_pair_columns = filter_feature_columns(columns, tanh_pair)
    assert "4h_signed_large_trade_pressure_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_24_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_pair_columns

    spread_overlay = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_spread_overlay")
    spread_overlay_columns = filter_feature_columns(columns, spread_overlay)
    assert "4h_large_trade_pressure_24_minus_12_stable_rank" in spread_overlay_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in spread_overlay_columns

    tanh_spread = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_12_tanh_spread_overlay")
    tanh_spread_columns = filter_feature_columns(columns, tanh_spread)
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_spread_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_spread_columns
    assert "4h_large_trade_pressure_24_minus_12_stable_tanh" in tanh_spread_columns

    no_slow_flow = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_slow_4h_bounded_flow")
    no_slow_flow_columns = filter_feature_columns(columns, no_slow_flow)
    assert "4h_taker_imbalance_mean_12" not in no_slow_flow_columns
    assert "4h_taker_imbalance_mean_24" not in no_slow_flow_columns
    assert "4h_taker_imbalance" in no_slow_flow_columns
    assert "4h_taker_imbalance_slope" in no_slow_flow_columns

    no_cvd_rate = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_1h_cvd_rate")
    no_cvd_rate_columns = filter_feature_columns(columns, no_cvd_rate)
    assert "cvd_cumulative_rate_norm" not in no_cvd_rate_columns
    assert "4h_taker_imbalance_mean_24" in no_cvd_rate_columns

    combined_cvd_flow = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_slow_4h_bounded_flow_no_1h_cvd_rate",
    )
    combined_cvd_flow_columns = filter_feature_columns(columns, combined_cvd_flow)
    assert "cvd_cumulative_rate_norm" not in combined_cvd_flow_columns
    assert "4h_taker_imbalance_mean_24" not in combined_cvd_flow_columns

    guardrail = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_bad_fold_guardrail_light")
    guardrail_columns = filter_feature_columns(columns, guardrail)
    assert "cvd_cumulative_rate_norm" not in guardrail_columns
    assert "4h_taker_imbalance_mean_12" not in guardrail_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in guardrail_columns
    assert "4h_gk_vol_14" not in guardrail_columns
    assert "4h_large_trade_pressure_24_stable_rank" in guardrail_columns

    no_vpt_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_vpt_zscore_only")
    no_vpt_only_columns = filter_feature_columns(columns, no_vpt_only)
    assert "4h_vpt_zscore" not in no_vpt_only_columns
    assert "4h_vol_per_trade_log_zscore" in no_vpt_only_columns
    assert "4h_whale_buy_flag" in no_vpt_only_columns

    no_vpt_log_only = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_vol_per_trade_log_zscore_only",
    )
    no_vpt_log_only_columns = filter_feature_columns(columns, no_vpt_log_only)
    assert "4h_vol_per_trade_log_zscore" not in no_vpt_log_only_columns
    assert "4h_vpt_zscore" in no_vpt_log_only_columns
    assert "4h_whale_sell_flag" in no_vpt_log_only_columns

    no_vol_pressure_24 = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_pressure_24_only",
    )
    no_vol_pressure_24_columns = filter_feature_columns(columns, no_vol_pressure_24)
    assert "4h_gk_vol_14" not in no_vol_pressure_24_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in no_vol_pressure_24_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_vol_pressure_24_columns

    no_vol_no_whale_zscores = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_no_4h_whale_zscores",
    )
    no_vol_no_whale_zscore_columns = filter_feature_columns(columns, no_vol_no_whale_zscores)
    assert "4h_gk_vol_14" not in no_vol_no_whale_zscore_columns
    assert "4h_vpt_zscore" not in no_vol_no_whale_zscore_columns
    assert "4h_whale_sell_flag" in no_vol_no_whale_zscore_columns


def test_full_promotion_gate_uses_threshold_selected_f1() -> None:
    config = {
        "experiments": {
            "promotion_gates": {
                "full": {
                    "min_mean_rank_ic_delta": 0.005,
                    "min_positive_ic_fraction_floor": 0.75,
                    "max_std_rank_ic_delta": 0.002,
                    "min_selected_threshold_f1": 0.45,
                    "min_selected_threshold_f1_delta": 0.0,
                    "min_long_f1_delta": None,
                    "min_top_10_lift_global_delta": 0.05,
                }
            }
        }
    }
    control = {
        "mean_rank_ic": 0.048,
        "positive_ic_fraction": 0.738,
        "std_rank_ic": 0.085,
        "mean_long_f1": 0.268,
        "test_f1_at_selected_threshold": 0.463,
        "top_10_lift_global": 1.044,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    candidate = {
        "mean_rank_ic": 0.060,
        "positive_ic_fraction": 0.762,
        "std_rank_ic": 0.086,
        "mean_long_f1": 0.274,
        "test_f1_at_selected_threshold": 0.464,
        "top_10_lift_global": 1.141,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }

    assert _passes_full(candidate, control, config) == (True, "")


def test_triage_promotion_gate_uses_downside_metrics() -> None:
    config = {
        "experiments": {
            "promotion_gates": {
                "triage": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.005,
                    "min_top_10_lift_global": 1.05,
                    "min_top_10_positive_lift_fold_rate": 0.55,
                    "min_worst_5_rank_ic_delta": 0.0,
                    "max_negative_ic_fraction_delta": 0.0,
                    "min_top_10_bad_fold_lift_mean": 1.0,
                }
            }
        }
    }
    control = {
        "mean_rank_ic": 0.03,
        "std_rank_ic": 0.10,
        "positive_ic_fraction": 0.50,
        "top_10_lift_global": 1.07,
        "top_10_positive_lift_fold_rate": 0.56,
        "worst_5_rank_ic_mean": -0.11,
        "negative_ic_fraction": 0.50,
        "top_10_bad_fold_lift_mean": 0.95,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    candidate = {
        "mean_rank_ic": 0.04,
        "std_rank_ic": 0.09,
        "positive_ic_fraction": 0.62,
        "top_10_lift_global": 1.16,
        "top_10_positive_lift_fold_rate": 0.70,
        "worst_5_rank_ic_mean": -0.08,
        "negative_ic_fraction": 0.38,
        "top_10_bad_fold_lift_mean": 1.05,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    weaker_downside = dict(candidate, worst_5_rank_ic_mean=-0.13)

    assert _passes_triage(candidate, control, config) == (True, "")
    assert _passes_triage(weaker_downside, control, config) == (False, "worst_5_rank_ic_delta")


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
        "full_cv_profiles": ["control", "candidate"],
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
    assert not result["profile_delta"].empty
    assert {"rank_ic_delta", "top_10_lift_delta", "threshold_f1_delta"}.issubset(result["profile_delta"].columns)
    assert (tmp_path / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "profile_delta_vs_control.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "profile_blend.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_delta_vs_control.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_blend.csv").exists()
    assert diagnostics["zip_paths"]
    assert not diagnostics["profile_delta"].empty
    assert not diagnostics["profile_blend"].empty
    assert set(diagnostics["profile_blend"]["blend_method"]) == {"prob_mean", "rank_mean"}
    assert {"reviewable", "review_reason", "mean_rank_ic_delta_vs_control"}.issubset(diagnostics["profile_blend"].columns)
    assert "best_profile_blend" in diagnostics["decision"]
    assert (tmp_path / "reports" / "phase1_experiment_bundle_matrix.zip").exists()
    assert (tmp_path / "reports" / "phase1_latest_experiment_bundle.zip").exists()
    assert diagnostics["bundle_zip"].endswith("phase1_experiment_bundle_matrix.zip")
    assert diagnostics["latest_bundle_zip"].endswith("phase1_latest_experiment_bundle.zip")
    assert diagnostics["decision"]["recommendation"] in {
        "keep_control_profile",
        "promote_best_candidate",
        "review_profile_blend",
    }
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_bundle_matrix.zip") as archive:
        assert "matrix/profile_delta_vs_control.csv" in archive.namelist()
        assert "matrix/profile_blend.csv" in archive.namelist()


def test_seed_audit_writes_isolated_seed_summaries(synthetic_klines, tiny_config, tmp_path) -> None:
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
        "seed_audit": {
            "enabled": True,
            "profiles": ["control"],
            "seeds": [11, 12],
            "fold_ids": [0],
        },
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    result = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="seeded", device="cpu")
    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="seeded",
    )

    assert set(result["seed_audit"]["seed"]) == {11, 12}
    assert result["seed_stability"].loc[0, "seed_count"] == 2
    assert (tmp_path / "experiments" / "seeded" / "seed_audit.csv").exists()
    assert (tmp_path / "experiments" / "seeded" / "seed_stability.csv").exists()
    assert (tmp_path / "experiments" / "seeded" / "seed_ensemble.csv").exists()
    assert not result["seed_ensemble"].empty
    assert result["seed_ensemble"].loc[0, "seed_count"] == 2
    assert not diagnostics["seed_audit"].empty
    assert not diagnostics["seed_stability"].empty
    assert not diagnostics["seed_ensemble"].empty
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_bundle_seeded.zip") as archive:
        names = set(archive.namelist())
    assert "seeded/seed_audit.csv" in names
    assert "seeded/seed_stability.csv" in names
    assert "seeded/seed_ensemble.csv" in names


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


def test_write_experiment_diagnostics_raises_when_run_has_no_completed_profiles(tmp_path, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {"control": {"include_patterns": ["*"], "exclude_patterns": []}}
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
    }

    # Create an empty run directory (e.g. interrupted training that wrote no manifests/predictions).
    run_dir = tmp_path / "experiments" / "empty_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="No completed profile runs found"):
        write_experiment_diagnostics(
            checkpoint_dir=tmp_path,
            config=config,
            output_dir=tmp_path / "reports",
            run_id="empty_run",
        )
