"""Diagnostics and validation metrics."""

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
    fold_diagnostics,
    good_bad_fold_summary,
    regime_diagnostics,
    write_phase1_diagnostic_bundle,
)

__all__ = [
    "calibration_table",
    "classification_metrics",
    "extract_embeddings",
    "fold_diagnostics",
    "good_bad_fold_summary",
    "load_fold_model",
    "permutation_importance_rank_ic",
    "phase1_report",
    "predict_probabilities",
    "rank_ic",
    "regime_diagnostics",
    "tsne_embeddings",
    "write_phase1_diagnostic_bundle",
]
