from __future__ import annotations

import json
import zipfile

import pandas as pd

from yenibot.diagnostics import (
    bad_fold_feature_forensics,
    bad_fold_group_forensics,
    calibrate_test_probabilities_from_val,
    calibration_table,
    feature_group_diagnostics,
    feature_group_importance_summary,
    feature_profile_diagnostics,
    fold_diagnostics,
    good_bad_feature_audit,
    score_band_by_fold_diagnostics,
    score_band_diagnostics,
    score_band_summary_diagnostics,
    score_lift_diagnostics,
    score_lift_by_fold_diagnostics,
    mtf_leakage_diagnostics,
    recent_fold_diagnostics,
    regime_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC"),
            "fold": [0] * 6 + [1] * 6,
            "label": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "prob_long": [0.1, 0.7, 0.2, 0.8, 0.4, 0.6, 0.3, 0.2, 0.5, 0.9, 0.1, 0.7],
            "forward_return": [-0.01, 0.03, -0.02, 0.02, 0.0, 0.01, -0.01, -0.02, 0.01, 0.04, 0.0, 0.02],
            "regime_prob_0": [0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4],
            "regime_prob_1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "regime_prob_2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3],
            "4h_source_timestamp": pd.date_range("2021-12-31 20:00", periods=12, freq="1h", tz="UTC"),
            "4h_available_timestamp": pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC"),
            "true_cvd_zscore": [-1.0, 0.5, -0.8, 0.8, -0.5, 1.0, -0.3, -0.2, 0.4, 1.2, -0.9, 0.7],
            "4h_true_cvd_zscore": [-0.5, 0.4, -0.4, 0.7, -0.2, 0.8, -0.1, -0.2, 0.2, 0.9, -0.6, 0.5],
        }
    )


def test_calibration_bins_are_readable_numbers() -> None:
    table = calibration_table(_predictions()["label"], _predictions()["prob_long"], bins=4)

    assert table["bin"].tolist() == [0, 1, 2, 3]
    assert set(table.columns) == {"bin", "count", "mean_prob_long", "actual_long_rate"}


def test_diagnostic_bundle_contains_shareable_outputs(tmp_path) -> None:
    predictions = _predictions()
    calibration = calibration_table(predictions["label"], predictions["prob_long"], bins=4)
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20
    regime_metrics = regime_diagnostics(predictions)
    threshold_metrics = threshold_diagnostics(predictions)
    report = {
        "passed": False,
        "mean_rank_ic": 0.01,
        "std_rank_ic": 0.10,
        "positive_ic_fraction": 0.50,
        "mean_long_f1": 0.20,
        "mean_prauc": 0.35,
        "calibration_separation": 0.01,
        "checks": {"rank_ic_mean": False},
        "alerts": [],
    }

    zip_path = write_phase1_diagnostic_bundle(
        output_dir=tmp_path,
        report=report,
        predictions=predictions,
        calibration=calibration,
        fold_metrics=fold_metrics,
        regime_metrics=regime_metrics,
        threshold_metrics=threshold_metrics,
        threshold_summary=threshold_summary_diagnostics(threshold_metrics),
        stationarity_policy=stationarity_policy_diagnostics(["true_cvd_zscore"], {"features": {"stationarity": {"exclude_patterns": ["*atr_14"]}}}),
        model_feature_columns=["true_cvd_zscore"],
        score_lift_by_fold=score_lift_by_fold_diagnostics(predictions, bins=4),
        recent_fold_summary=recent_fold_diagnostics(fold_metrics, recent_folds=1),
        feature_groups=feature_group_diagnostics(["true_cvd_zscore"]),
        feature_profile=feature_profile_diagnostics(
            ["true_cvd_zscore"],
            {"features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*true_cvd*"], "exclude_patterns": []}}}},
        ),
        config={"project": {"name": "test"}, "validation": {"calibration_bins": 4}},
    )

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "phase1_report.json" in names
        assert "summary.md" in names
        assert {"test_predictions.parquet", "test_predictions.csv"} & names
        assert "calibration.csv" in names
        assert "fold_metrics.csv" in names
        assert "regime_metrics.csv" in names
        assert "threshold_summary.csv" in names
        assert "score_lift.csv" in names
        assert "score_band_lift.csv" in names
        assert "score_band_by_fold.csv" in names
        assert "score_band_summary.csv" in names
        assert "model_feature_columns.csv" in names
        assert "stationarity_policy.csv" in names
        assert "score_lift_by_fold.csv" in names
        assert "recent_fold_summary.csv" in names
        assert "feature_groups.csv" in names
        assert "feature_profile.csv" in names
        assert "bad_fold_feature_forensics.csv" in names
        assert "bad_fold_group_forensics.csv" in names
        payload = json.loads(archive.read("phase1_report.json"))
        assert payload["passed"] is False


def test_calibration_threshold_and_leakage_diagnostics() -> None:
    predictions = pd.concat(
        [
            _predictions().assign(split="val"),
            _predictions().assign(split="test"),
        ],
        ignore_index=True,
    )
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "max_rank_ic_std": 0.03,
            "min_positive_ic_fraction": 0.75,
            "min_long_f1": 0.45,
            "suspicious_rank_ic": 0.10,
            "random_like_rank_ic": 0.01,
            "calibration_bins": 4,
        }
    }

    calibrated, report, calibrated_table = calibrate_test_probabilities_from_val(predictions, config)
    thresholds = threshold_diagnostics(predictions)
    leakage = mtf_leakage_diagnostics(predictions[predictions["split"] == "test"])

    assert "prob_long_calibrated" in calibrated.columns
    assert "mean_rank_ic" in report
    assert len(calibrated_table) == 4
    assert {"selected_threshold", "test_oracle_best_f1"}.issubset(thresholds.columns)
    assert leakage["passed"].all()


def test_good_bad_feature_audit_returns_ranked_feature_differences() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20

    audit = good_bad_feature_audit(predictions, fold_metrics, top_n=5)

    assert not audit.empty
    assert {"feature", "ks_stat", "abs_standardized_diff"}.issubset(audit.columns)


def test_stationarity_policy_diagnostics_flags_raw_model_features() -> None:
    config = {
        "features": {
            "stationarity": {
                "exclude_patterns": ["*close_denoised", "*atr_14", "*true_cvd_delta"],
            }
        }
    }
    diagnostics = stationarity_policy_diagnostics(
        ["close_denoised_log_return", "4h_atr_14", "true_cvd_delta_norm"],
        config,
    )

    overall = diagnostics.loc[diagnostics["check"] == "stationarity_policy_overall"].iloc[0]
    assert not bool(overall["passed"])
    assert overall["matched_features"] == "4h_atr_14"


def test_stationarity_policy_diagnostics_flags_raw_order_flow_v2_inputs() -> None:
    config = {
        "features": {
            "order_flow_v2": {
                "enabled": True,
                "stable_only": True,
                "pressure_windows": [3],
            },
            "stationarity": {"exclude_patterns": []},
        }
    }
    diagnostics = stationarity_policy_diagnostics(
        ["cvd_pressure_3", "cvd_pressure_3_stable_rank"],
        config,
    )

    raw_check = diagnostics.loc[diagnostics["check"] == "order_flow_v2_stable_only"].iloc[0]
    assert not bool(raw_check["passed"])
    assert raw_check["matched_features"] == "cvd_pressure_3"


def test_score_lift_diagnostics_reports_top_bin_lift() -> None:
    lift = score_lift_diagnostics(_predictions(), bins=4)

    assert {"score_bin", "actual_long_rate", "base_long_rate", "lift_vs_base", "is_top_bin"}.issubset(lift.columns)
    assert lift["is_top_bin"].sum() == 1
    assert lift.loc[lift["is_top_bin"], "lift_vs_base"].iloc[0] > 1.0


def test_fold_lift_recent_and_feature_group_diagnostics() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    lift_by_fold = score_lift_by_fold_diagnostics(predictions, bins=3)
    recent = recent_fold_diagnostics(fold_metrics, recent_folds=1)
    groups = feature_group_diagnostics(
        [
            "signed_large_trade_pressure_stable_zscore",
            "4h_cvd_pressure_3_stable_rank",
            "gk_vol_14",
        ]
    )
    group_importance = feature_group_importance_summary(
        pd.DataFrame(
            {
                "feature": ["signed_large_trade_pressure_stable_zscore", "gk_vol_14"],
                "rank_ic_drop": [0.05, 0.01],
            }
        )
    )
    profile = feature_profile_diagnostics(
        ["true_cvd_zscore", "gk_vol_14"],
        {"features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*true_cvd*"], "exclude_patterns": ["*atr*"]}}}},
    )

    assert {"top_lift_vs_base", "bin_long_rate_spearman"}.issubset(lift_by_fold.columns)
    assert "recent_minus_all" in recent.columns
    assert set(groups["family"]) == {"order_flow_v2_stable", "volatility_structure"}
    assert "mean_rank_ic_drop" in group_importance.columns
    assert "profile_include_pattern" in set(profile["check"])


def test_score_band_diagnostics_reports_upper_score_ranges() -> None:
    predictions = _predictions()
    bands = [
        {"name": "top_bin", "min_bin": 3, "max_bin": 3},
        {"name": "upper_half", "min_bin": 2, "max_bin": 3},
    ]

    band_lift = score_band_diagnostics(predictions, bins=4, bands=bands)
    band_by_fold = score_band_by_fold_diagnostics(predictions, bins=4, bands=bands)
    summary = score_band_summary_diagnostics(band_by_fold)

    assert band_lift["band"].tolist() == ["top_bin", "upper_half"]
    assert {"selection_rate", "lift_vs_base", "mean_forward_return"}.issubset(band_lift.columns)
    assert set(summary["band"]) == {"top_bin", "upper_half"}
    assert "positive_lift_fold_rate" in summary.columns


def test_bad_fold_forensics_reports_group_signal_changes() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20

    feature_forensics = bad_fold_feature_forensics(
        predictions,
        fold_metrics,
        feature_columns=["true_cvd_zscore", "4h_true_cvd_zscore"],
    )
    group_forensics = bad_fold_group_forensics(
        predictions,
        fold_metrics,
        feature_columns=["true_cvd_zscore", "4h_true_cvd_zscore"],
    )

    assert {"bad_fold", "feature", "delta_feature_ic_bad_minus_good", "signal_reversal"}.issubset(
        feature_forensics.columns
    )
    assert {"timeframe", "family", "mean_abs_delta_feature_ic", "top_delta_features"}.issubset(
        group_forensics.columns
    )
