"""Pinned historical OOF input builder; never a frozen future-OOS candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_cdf_test_scores(
    frame: pd.DataFrame, *, spec: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Map scores using past validation only; test labels never enter the map."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame[["timestamp", "fold", "split", "prob_long"]].isna().any().any():
        raise ValueError("Missing OOF identity/time/score")
    if set(frame.split.unique()) != {"val", "test"}:
        raise ValueError("Full OOF requires val and test splits only")
    fold = pd.to_numeric(frame.fold, errors="raise")
    if not np.isfinite(fold).all() or not (fold == np.floor(fold)).all():
        raise ValueError("Fold ids must be finite integers")
    frame["fold"] = fold.astype(int)
    if sorted(frame.fold.unique()) != list(range(int(spec["fold_count"]))):
        raise ValueError("OOF fold ids/count differ from pinned contract")
    score = pd.to_numeric(frame.prob_long, errors="raise")
    if not np.isfinite(score).all() or not score.between(0, 1).all():
        raise ValueError("OOF scores must be finite and in [0, 1]")
    frame["prob_long"] = score
    cutoff = pd.Timestamp(spec["historical_cutoff"])
    if cutoff.tzinfo is None or frame.timestamp.max() > cutoff:
        raise ValueError("OOF exceeds explicit historical-only cutoff")
    test_parts = []
    audits = []
    prior_test_end = None
    maturity = pd.Timedelta(hours=float(spec["label_maturity_hours"]) + 1)
    for fold_id, group in frame.groupby("fold", sort=True):
        if group.timestamp.duplicated().any():
            raise ValueError("Duplicate/overlapping timestamps within an OOF fold")
        val = group.loc[group.split.eq("val")].sort_values("timestamp")
        test = group.loc[group.split.eq("test")].sort_values("timestamp").copy()
        if (
            len(val) != spec["validation_rows_per_fold"]
            or len(test) != spec["test_rows_per_fold"]
        ):
            raise ValueError(
                f"Fold {fold_id} val/test counts differ from pinned contract"
            )
        if val.timestamp.max() + maturity > test.timestamp.min():
            raise ValueError("Validation labels are not mature before test starts")
        if prior_test_end is not None and test.timestamp.min() <= prior_test_end:
            raise ValueError("Test fold windows overlap or are not chronological")
        prior_test_end = test.timestamp.max()
        for part in (val, test):
            if not part.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():
                raise ValueError("OOF fold has missing/non-hourly bars")
        reference = np.sort(val.prob_long.to_numpy(dtype=float))
        test["raw_prob_long"] = test.prob_long
        test["prob_long"] = np.searchsorted(
            reference, test.raw_prob_long.to_numpy(), side="right"
        ) / len(reference)
        test_parts.append(test)
        audits.append(
            {
                "fold": int(fold_id),
                "validation_rows": len(val),
                "test_rows": len(test),
                "validation_end": val.timestamp.max().isoformat(),
                "test_start": test.timestamp.min().isoformat(),
                "test_end": test.timestamp.max().isoformat(),
                "validation_labels_mature_before_test": True,
            }
        )
    result = pd.concat(test_parts, ignore_index=True)
    if result.timestamp.duplicated().any():
        raise ValueError("Duplicate OOF test timestamp")
    return result, audits


def build_full_oof_inputs(
    scope_dir: str | Path, *, spec: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scope = Path(scope_dir)
    prediction_path = scope / "predictions_all.parquet"
    manifest_path = scope / "training_manifest.json"
    for path, expected in (
        (prediction_path, spec["predictions_sha256"]),
        (manifest_path, spec["training_manifest_sha256"]),
    ):
        if not expected or file_sha256(path) != str(expected).lower():
            raise ValueError(f"Pinned source SHA-256 mismatch: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completed") is not True or manifest.get("fold_scope") != "full":
        raise ValueError("Source must be completed full-scope OOF")
    if manifest.get("profile") != spec["profile"]:
        raise ValueError("Source profile differs from pinned contract")
    if pd.Timestamp(manifest["data_end"]) > pd.Timestamp(spec["historical_cutoff"]):
        raise ValueError("Source training data exceeds historical cutoff")
    if spec["source_timestamp_semantics"] != "binance_bar_open_1h":
        raise ValueError("Unsupported or ambiguous source timestamp semantics")
    if spec["score_transform"] != "same_fold_validation_empirical_cdf_right":
        raise ValueError("Unsupported score transform")
    columns = [
        "timestamp",
        "fold",
        "split",
        "prob_long",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "label",
        "forward_return",
        "tb_return",
    ]
    frame = pd.read_parquet(prediction_path, columns=columns)
    if len(frame) != manifest["prediction_rows"]:
        raise ValueError("Source prediction rows differ from training manifest")
    if frame.timestamp.max() > pd.Timestamp(manifest["data_end"]):
        raise ValueError("Predictions exceed source manifest data end")
    test, fold_audit = validation_cdf_test_scores(frame, spec=spec)
    candidate_id = f"{spec['source_run_id']}_retained_full_oof_validation_cdf_v1"
    bars = test[["timestamp", "open", "high", "low", "close", "atr_14"]].rename(
        columns={"timestamp": "bar_open_time"}
    )
    bars["bar_close_time"] = bars.bar_open_time + pd.Timedelta(hours=1)
    signals = test[
        [
            "timestamp",
            "fold",
            "split",
            "prob_long",
            "raw_prob_long",
            "label",
            "forward_return",
            "tb_return",
        ]
    ].rename(columns={"timestamp": "source_bar_open_time"})
    signals["decision_time"] = signals.source_bar_open_time + pd.Timedelta(hours=1)
    signals["candidate_id"] = candidate_id
    metadata = {
        "input_kind": "historical_full_oof_not_a_frozen_candidate",
        "candidate_id": candidate_id,
        "source_scope_dir": str(scope.resolve()),
        "source_training_manifest": manifest,
        "spec": spec,
        "source_hashes_verified": True,
        "source_prediction_rows": len(frame),
        "test_rows": len(test),
        "fold_audit": fold_audit,
        "test_labels_used_for_score_transform": False,
        "score_semantics": "past_validation_percentile_not_calibrated_probability",
        "historical_profile_selection_bias_remains": True,
        "promotion_allowed": False,
        "live_trading_allowed": False,
    }
    return bars, signals, metadata
