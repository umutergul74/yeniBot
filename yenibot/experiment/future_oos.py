"""No-refit future out-of-sample evaluation for frozen candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from yenibot.experiment.common import _cfg, _rank_ic_for_frame, _table_markdown, _write_json
from yenibot.experiment.frozen import verify_frozen_manifest_artifacts
from yenibot.experiment.holdout import _aggregate_holdout_predictions, _predict_holdout_for_profile

__all__ = ["evaluate_future_oos"]


def _moving_block_sample_indices(
    n_rows: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block = max(1, min(int(block_length), n_rows))
    starts = rng.integers(0, max(1, n_rows - block + 1), size=int(np.ceil(n_rows / block)))
    return np.concatenate([np.arange(start, start + block) for start in starts])[:n_rows]


def _top_decile_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    score = pd.to_numeric(frame["prob_long"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    returns = pd.to_numeric(frame["forward_return"], errors="coerce")
    valid = score.notna() & labels.notna() & returns.notna()
    score, labels, returns = score[valid], labels[valid], returns[valid]
    if len(score) < 10:
        return np.nan, np.nan
    threshold = float(score.quantile(0.90))
    selected = score >= threshold
    base_rate = float(labels.mean())
    selected_rate = float(labels[selected].mean()) if selected.any() else np.nan
    lift = selected_rate / base_rate if base_rate > 0 and np.isfinite(selected_rate) else np.nan
    mean_return = float(returns[selected].mean()) if selected.any() else np.nan
    return lift, mean_return


def _bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    block_length: int,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    if len(frame) < 3 or repeats <= 0:
        return {
            "rank_ic_ci_low": np.nan,
            "rank_ic_ci_high": np.nan,
            "top_10_forward_return_ci_low": np.nan,
            "top_10_forward_return_ci_high": np.nan,
        }
    rng = np.random.default_rng(random_seed)
    rank_ics: list[float] = []
    top_returns: list[float] = []
    for _ in range(repeats):
        sampled = frame.iloc[
            _moving_block_sample_indices(len(frame), block_length=block_length, rng=rng)
        ]
        rank_ics.append(_rank_ic_for_frame(sampled))
        _, top_return = _top_decile_metrics(sampled)
        top_returns.append(top_return)
    alpha = (1.0 - confidence_level) / 2.0

    def interval(values: list[float]) -> tuple[float, float]:
        clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if clean.size == 0:
            return np.nan, np.nan
        return float(np.quantile(clean, alpha)), float(np.quantile(clean, 1.0 - alpha))

    rank_low, rank_high = interval(rank_ics)
    return_low, return_high = interval(top_returns)
    return {
        "rank_ic_ci_low": rank_low,
        "rank_ic_ci_high": rank_high,
        "top_10_forward_return_ci_low": return_low,
        "top_10_forward_return_ci_high": return_high,
    }


def _profile_predictions(
    *,
    manifest: dict[str, Any],
    run_dir: Path,
    future_context: pd.DataFrame,
    future_start: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    for component in manifest.get("components", []) or []:
        profile = str(component["profile"])
        raw = _predict_holdout_for_profile(
            scope_dir=run_dir / str(component["scope_relative_path"]),
            manifest={
                "profile": profile,
                "feature_columns": list(component["feature_columns"]),
            },
            holdout_context=future_context,
            holdout_start=future_start,
            config=config,
        )
        predictions[profile] = _aggregate_holdout_predictions(raw, profile=profile)
    return predictions


def _candidate_predictions(
    manifest: dict[str, Any],
    component_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    profiles = [str(item) for item in manifest.get("profiles", []) or []]
    if manifest.get("candidate_type") == "profile":
        return component_predictions.get(profiles[0], pd.DataFrame()).copy()
    if any(profile not in component_predictions or component_predictions[profile].empty for profile in profiles):
        return pd.DataFrame()
    weights = np.asarray(manifest.get("weights", []) or [], dtype=float)
    if len(weights) != len(profiles) or weights.sum() <= 0:
        return pd.DataFrame()
    weights = weights / weights.sum()
    merged = component_predictions[profiles[0]].copy()
    merged = merged.rename(columns={"prob_long": f"prob_long_{0}"})
    for index, profile in enumerate(profiles[1:], start=1):
        other = component_predictions[profile][["timestamp", "prob_long"]].rename(
            columns={"prob_long": f"prob_long_{index}"}
        )
        merged = merged.merge(other, on="timestamp", how="inner")
    merged["prob_long"] = sum(
        merged[f"prob_long_{index}"].astype(float) * weights[index]
        for index in range(len(profiles))
    )
    return merged.drop(columns=[f"prob_long_{index}" for index in range(len(profiles))])


def _evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold = float((manifest.get("threshold") or {}).get("value", 0.5))
    frame = predictions.copy().replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["prob_long", "label", "forward_return"])
    labels = frame["label"].astype(int)
    scores = frame["prob_long"].astype(float)
    selected = scores >= threshold
    prevalence = float(labels.mean())
    pred_rate = float(selected.mean())
    precision = float(precision_score(labels, selected, zero_division=0))
    recall = float(recall_score(labels, selected, zero_division=0))
    f1 = float(f1_score(labels, selected, zero_division=0))
    prauc = (
        float(average_precision_score(labels, scores))
        if labels.nunique(dropna=True) > 1
        else np.nan
    )
    random_f1 = (
        float(2.0 * prevalence * pred_rate / (prevalence + pred_rate))
        if prevalence + pred_rate > 0
        else 0.0
    )
    top_lift, top_return = _top_decile_metrics(frame)
    selected_return = (
        float(pd.to_numeric(frame.loc[selected, "forward_return"], errors="coerce").mean())
        if selected.any()
        else np.nan
    )
    future_cfg = _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
    intervals = _bootstrap_intervals(
        frame,
        block_length=int(future_cfg.get("block_length", 24)),
        repeats=int(future_cfg.get("bootstrap_repeats", 500)),
        confidence_level=float(future_cfg.get("confidence_level", 0.95)),
        random_seed=int(future_cfg.get("random_seed", 42)),
    )
    gates = future_cfg.get("gates", {}) or {}
    checks = {
        "rank_ic": _rank_ic_for_frame(frame) >= float(gates.get("min_rank_ic", 0.03)),
        "rank_ic_lower_ci": intervals["rank_ic_ci_low"]
        >= float(gates.get("min_rank_ic_lower_ci", 0.0)),
        "top_10_lift": top_lift >= float(gates.get("min_top_10_lift", 1.05)),
        "top_10_forward_return": top_return
        > float(gates.get("min_top_10_forward_return", 0.0)),
        "prauc_lift": prauc / prevalence
        >= float(gates.get("min_prauc_lift_vs_prevalence", 1.05))
        if prevalence > 0 and np.isfinite(prauc)
        else False,
        "precision_lift": precision / prevalence
        >= float(gates.get("min_precision_lift_vs_prevalence", 1.05))
        if prevalence > 0
        else False,
        "f1_skill": f1 - random_f1
        > float(gates.get("min_f1_skill_vs_rate_random", 0.0)),
        "pred_long_rate": pred_rate <= float(gates.get("max_pred_long_rate", 0.70)),
        "selected_forward_return": selected_return > 0.0,
    }
    return {
        "candidate_id": manifest["candidate_id"],
        "candidate_type": manifest["candidate_type"],
        "rows": int(len(frame)),
        "data_start": str(pd.to_datetime(frame["timestamp"], utc=True).min()),
        "data_end": str(pd.to_datetime(frame["timestamp"], utc=True).max()),
        "threshold": threshold,
        "threshold_source": str((manifest.get("threshold") or {}).get("source", "")),
        "rank_ic": _rank_ic_for_frame(frame),
        **intervals,
        "label_prevalence": prevalence,
        "pred_long_rate": pred_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rate_matched_random_f1": random_f1,
        "f1_skill_vs_rate_random": f1 - random_f1,
        "prauc": prauc,
        "prauc_lift_vs_prevalence": prauc / prevalence if prevalence > 0 else np.nan,
        "precision_lift_vs_prevalence": precision / prevalence if prevalence > 0 else np.nan,
        "top_10_lift": top_lift,
        "top_10_forward_return": top_return,
        "selected_forward_return": selected_return,
        "evidence_passed": all(checks.values()),
        "failed_gates": ";".join(name for name, passed in checks.items() if not passed),
        "no_refit_verified": True,
        "manifest_hash": manifest["manifest_hash"],
    }


def evaluate_future_oos(
    *,
    run_dir: str | Path,
    report_dir: str | Path,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score only rows after the frozen anchor using transform/predict operations."""

    run_path = Path(run_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    future_cfg = _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
    frozen_cfg = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    anchor_value = frozen_cfg.get("anchor_data_end")
    min_rows = int(future_cfg.get("min_rows", 720))
    preferred_rows = int(future_cfg.get("preferred_rows", 2160))
    if not bool(future_cfg.get("enabled", False)) or not anchor_value:
        evaluation = pd.DataFrame(
            columns=["candidate_id", "rank_ic", "evidence_passed"]
        )
        status = {
            "enabled": bool(future_cfg.get("enabled", False)),
            "ready_for_evaluation": False,
            "evaluation_completed": False,
            "primary_candidate_passed": False,
            "promotion_allowed": False,
            "promotion_block_reason": "future_oos_protocol_disabled_or_missing_anchor",
            "fit_operations_performed": 0,
            "artifact_integrity_errors": [],
        }
        evaluation.to_csv(report_path / "future_oos_evaluation.csv", index=False)
        (report_path / "future_oos_evaluation.md").write_text(
            _table_markdown("Future OOS Evaluation", evaluation),
            encoding="utf-8",
        )
        _write_json(
            report_path / "future_oos_evaluation.json",
            {"status": status, "rows": []},
        )
        _write_json(report_path / "future_oos_readiness.json", status)
        return evaluation, status
    anchor = pd.to_datetime(anchor_value, utc=True, errors="raise")
    data_dir = Path(str(_cfg(config, ["paths", "data_dir"], "data")))
    labeled_path = data_dir / "processed" / "labeled_1h.parquet"
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    if not labeled_path.exists():
        future_count = 0
        latest = None
        errors.append(f"missing_labeled_data:{labeled_path}")
        labeled = pd.DataFrame()
    else:
        labeled = pd.read_parquet(labeled_path).copy()
        labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], utc=True)
        future = labeled.loc[labeled["timestamp"] > anchor].copy()
        future_count = int(len(future))
        latest = future["timestamp"].max() if not future.empty else None

    ready = bool(future_count >= min_rows)
    if ready:
        seq_len = int(_cfg(config, ["model", "seq_len"], 64))
        context = labeled.loc[labeled["timestamp"] <= anchor].tail(max(seq_len - 1, 0))
        future_context = pd.concat(
            [context, labeled.loc[labeled["timestamp"] > anchor]],
            ignore_index=True,
        ).drop_duplicates(subset=["timestamp"], keep="last")
        future_context = future_context.sort_values("timestamp").reset_index(drop=True)
        future_start = anchor + pd.Timedelta(hours=1)
        for manifest in manifests:
            if not bool(manifest.get("available", False)):
                errors.extend(
                    f"{manifest.get('candidate_id')}:{reason}"
                    for reason in manifest.get("unavailable_reasons", []) or []
                )
                continue
            integrity_errors = verify_frozen_manifest_artifacts(manifest, run_dir=run_path)
            if integrity_errors:
                errors.extend(
                    f"{manifest.get('candidate_id')}:{reason}" for reason in integrity_errors
                )
                continue
            components = _profile_predictions(
                manifest=manifest,
                run_dir=run_path,
                future_context=future_context,
                future_start=future_start,
                config=config,
            )
            predictions = _candidate_predictions(manifest, components)
            if predictions.empty:
                errors.append(f"{manifest.get('candidate_id')}:no_predictions")
                continue
            predictions["candidate_id"] = manifest["candidate_id"]
            prediction_frames.append(predictions)
            rows.append(_evaluate_predictions(predictions, manifest=manifest, config=config))

    evaluation = pd.DataFrame(rows)
    if evaluation.empty:
        evaluation = pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_type",
                "rows",
                "rank_ic",
                "rank_ic_ci_low",
                "rank_ic_ci_high",
                "f1",
                "f1_skill_vs_rate_random",
                "prauc_lift_vs_prevalence",
                "precision_lift_vs_prevalence",
                "top_10_lift",
                "top_10_forward_return",
                "selected_forward_return",
                "evidence_passed",
                "failed_gates",
                "no_refit_verified",
                "manifest_hash",
            ]
        )
    primary_id = str(frozen_cfg.get("primary_candidate_id", ""))
    primary = evaluation.loc[evaluation["candidate_id"].astype(str) == primary_id]
    primary_passed = bool(
        not primary.empty and primary["evidence_passed"].astype(bool).iloc[0]
    )
    charter_active = str(
        _cfg(config, ["validation", "charter", "active_version"], "v3_legacy")
    )
    status = {
        "enabled": bool(future_cfg.get("enabled", False)),
        "anchor_data_end": anchor.isoformat(),
        "latest_available_data_end": latest.isoformat() if latest is not None else None,
        "new_labeled_rows": future_count,
        "min_rows": min_rows,
        "preferred_rows": preferred_rows,
        "min_rows_remaining": max(0, min_rows - future_count),
        "preferred_rows_remaining": max(0, preferred_rows - future_count),
        "ready_for_evaluation": ready,
        "evaluation_completed": bool(ready and not evaluation.empty),
        "primary_candidate_id": primary_id,
        "primary_candidate_passed": primary_passed,
        "active_charter_version": charter_active,
        "promotion_allowed": bool(primary_passed and charter_active != "v3_legacy"),
        "promotion_block_reason": (
            "future_oos_not_ready"
            if not ready
            else "future_oos_evaluation_missing_or_failed"
            if not primary_passed
            else "draft_charter_not_active"
            if charter_active == "v3_legacy"
            else ""
        ),
        "fit_operations_performed": 0,
        "artifact_integrity_errors": errors,
    }
    evaluation.to_csv(report_path / "future_oos_evaluation.csv", index=False)
    (report_path / "future_oos_evaluation.md").write_text(
        _table_markdown("Future OOS Evaluation", evaluation),
        encoding="utf-8",
    )
    _write_json(
        report_path / "future_oos_evaluation.json",
        {"status": status, "rows": evaluation.to_dict(orient="records")},
    )
    _write_json(report_path / "future_oos_readiness.json", status)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_parquet(report_path / "future_oos_predictions.parquet", index=False)
        predictions.head(200).to_csv(
            report_path / "future_oos_prediction_sample.csv",
            index=False,
        )
    return evaluation, status
