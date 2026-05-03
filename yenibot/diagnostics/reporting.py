from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from yenibot.diagnostics.metrics import classification_metrics, rank_ic


def fold_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in predictions.groupby("fold"):
        metrics = classification_metrics(part["label"], part["prob_long"])
        rows.append(
            {
                "fold": int(fold),
                "count": int(len(part)),
                "start": str(part["timestamp"].min()) if "timestamp" in part.columns else "",
                "end": str(part["timestamp"].max()) if "timestamp" in part.columns else "",
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "long_f1": metrics["long_f1"],
                "prauc": metrics["prauc"],
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float((part["prob_long"] >= 0.5).mean()),
                "prob_long_mean": float(part["prob_long"].mean()),
                "prob_long_std": float(part["prob_long"].std(ddof=0)),
                "prob_long_p10": float(part["prob_long"].quantile(0.10)),
                "prob_long_p50": float(part["prob_long"].quantile(0.50)),
                "prob_long_p90": float(part["prob_long"].quantile(0.90)),
                "forward_return_mean": float(part["forward_return"].mean()),
                "forward_return_std": float(part["forward_return"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def regime_diagnostics(predictions: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    regime_columns = [column for column in predictions.columns if column.startswith("regime_prob_")]
    if not regime_columns:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["regime"] = frame[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    rows = []
    for regime, part in frame.groupby("regime"):
        y_true = part["label"].astype(int)
        y_pred = (part["prob_long"] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "regime": int(regime),
                "count": int(len(part)),
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float(y_pred.mean()),
                "precision_050": float(precision),
                "recall_050": float(recall),
                "long_f1_050": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def good_bad_fold_summary(fold_metrics: pd.DataFrame, *, good_ic: float = 0.10, bad_ic: float = -0.08) -> dict[str, Any]:
    good = fold_metrics.loc[fold_metrics["rank_ic"] >= good_ic, "fold"].astype(int).tolist()
    bad = fold_metrics.loc[fold_metrics["rank_ic"] <= bad_ic, "fold"].astype(int).tolist()
    return {
        "good_ic_threshold": good_ic,
        "bad_ic_threshold": bad_ic,
        "good_folds": good,
        "bad_folds": bad,
        "good_fold_count": len(good),
        "bad_fold_count": len(bad),
    }


def write_phase1_diagnostic_bundle(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
    tsne: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    prefix: str = "phase1_diagnostics",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_dir = output_path / f"{prefix}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    serializable_report = _json_safe(report)
    (bundle_dir / "phase1_report.json").write_text(
        json.dumps(serializable_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if config is not None:
        (bundle_dir / "config.json").write_text(
            json.dumps(_json_safe(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (bundle_dir / "summary.md").write_text(
        _summary_markdown(serializable_report, fold_metrics),
        encoding="utf-8",
    )

    predictions.to_parquet(bundle_dir / "test_predictions.parquet", index=False)
    calibration.to_csv(bundle_dir / "calibration.csv", index=False)
    fold_metrics.to_csv(bundle_dir / "fold_metrics.csv", index=False)
    if regime_metrics is not None and not regime_metrics.empty:
        regime_metrics.to_csv(bundle_dir / "regime_metrics.csv", index=False)
    if importance is not None and not importance.empty:
        importance.to_csv(bundle_dir / "permutation_importance.csv", index=False)
    if tsne is not None and not tsne.empty:
        tsne.to_parquet(bundle_dir / "tsne_embeddings.parquet", index=False)

    fold_summary = good_bad_fold_summary(fold_metrics)
    (bundle_dir / "good_bad_folds.json").write_text(
        json.dumps(_json_safe(fold_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    zip_path = output_path / f"{bundle_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            archive.write(path, path.relative_to(bundle_dir))
    return zip_path


def _summary_markdown(report: dict[str, Any], fold_metrics: pd.DataFrame) -> str:
    top_good = fold_metrics.sort_values("rank_ic", ascending=False).head(5)[["fold", "rank_ic"]]
    top_bad = fold_metrics.sort_values("rank_ic", ascending=True).head(5)[["fold", "rank_ic"]]
    lines = [
        "# Phase 1 Diagnostics",
        "",
        f"Decision: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Mean Rank IC: {report.get('mean_rank_ic'):.6f}",
        f"Rank IC Std: {report.get('std_rank_ic'):.6f}",
        f"Positive IC Fraction: {report.get('positive_ic_fraction'):.6f}",
        f"Mean Long F1: {report.get('mean_long_f1'):.6f}",
        f"Mean PRAUC: {report.get('mean_prauc'):.6f}",
        f"Calibration Separation: {report.get('calibration_separation'):.6f}",
        "",
        "## Checks",
    ]
    checks = report.get("checks", {})
    lines.extend(f"- {name}: {value}" for name, value in checks.items())
    alerts = report.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("## Alerts")
        lines.extend(f"- {alert}" for alert in alerts)
    lines.append("")
    lines.append("## Best Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_good.itertuples())
    lines.append("")
    lines.append("## Worst Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_bad.itertuples())
    lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
