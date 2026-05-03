"""Diagnostics and validation metrics."""

from yenibot.diagnostics.calibration import calibrate_test_probabilities_from_val
from yenibot.diagnostics.metrics import (
    calibration_table,
    classification_metrics,
    phase1_report,
    rank_ic,
)
from yenibot.diagnostics.model_analysis import (
    extract_embeddings,
    load_fold_model,
    permutation_importance_rank_ic,
    predict_probabilities,
    tsne_embeddings,
)
from yenibot.diagnostics.reporting import (
    best_f1_threshold,
    fold_diagnostics,
    good_bad_feature_audit,
    good_bad_fold_summary,
    model_feature_columns_frame,
    mtf_leakage_diagnostics,
    regime_diagnostics,
    score_lift_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)

__all__ = [
    "best_f1_threshold",
    "calibrate_test_probabilities_from_val",
    "calibration_table",
    "classification_metrics",
    "extract_embeddings",
    "fold_diagnostics",
    "good_bad_feature_audit",
    "good_bad_fold_summary",
    "load_fold_model",
    "model_feature_columns_frame",
    "mtf_leakage_diagnostics",
    "permutation_importance_rank_ic",
    "phase1_report",
    "predict_probabilities",
    "rank_ic",
    "regime_diagnostics",
    "score_lift_diagnostics",
    "stationarity_policy_diagnostics",
    "threshold_diagnostics",
    "threshold_summary_diagnostics",
    "tsne_embeddings",
    "write_phase1_diagnostic_bundle",
]
