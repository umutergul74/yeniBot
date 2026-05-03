from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from yenibot.diagnostics.metrics import calibration_table, phase1_report


def calibrate_test_probabilities_from_val(
    predictions: pd.DataFrame,
    config: object,
    *,
    method: str = "isotonic",
    prob_column: str = "prob_long",
    label_column: str = "label",
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit calibration on each fold's validation split and apply it to test rows."""

    if method not in {"isotonic", "platt"}:
        raise ValueError("method must be one of: isotonic, platt")
    if "split" not in predictions.columns:
        raise ValueError("predictions must contain val/test split labels")

    calibrated_parts = []
    for fold, fold_part in predictions.groupby("fold"):
        val = fold_part[fold_part["split"] == "val"].copy()
        test = fold_part[fold_part["split"] == "test"].copy()
        if val.empty or test.empty:
            continue
        calibrated = test.copy()
        calibrated["prob_long_raw"] = calibrated[prob_column]
        calibrated["prob_long_calibrated"] = _fit_transform_calibrator(
            val[prob_column].to_numpy(dtype=float),
            val[label_column].to_numpy(dtype=int),
            test[prob_column].to_numpy(dtype=float),
            method=method,
        )
        calibrated["calibration_method"] = method
        calibrated_parts.append(calibrated)

    if not calibrated_parts:
        raise ValueError("No folds with both val and test predictions were found")

    calibrated_test = pd.concat(calibrated_parts, ignore_index=True)
    report_frame = calibrated_test.copy()
    report_frame["prob_long"] = report_frame["prob_long_calibrated"]
    calibrated_report = phase1_report(report_frame, config)
    bins = _config_get(config, ["validation", "calibration_bins"], 10)
    calibrated_table = calibration_table(
        report_frame["label"],
        report_frame["prob_long"],
        bins=int(bins),
    )
    return calibrated_test, calibrated_report, calibrated_table


def _fit_transform_calibrator(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    test_probs: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    if len(np.unique(train_labels)) < 2 or len(np.unique(train_probs)) < 2:
        return np.full_like(test_probs, float(train_labels.mean()), dtype=float)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(train_probs, train_labels)
        return calibrator.transform(test_probs).clip(0.0, 1.0)

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(train_probs.reshape(-1, 1), train_labels)
    return calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1].clip(0.0, 1.0)


def _config_get(config: object, path: list[str], default: object) -> object:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current
