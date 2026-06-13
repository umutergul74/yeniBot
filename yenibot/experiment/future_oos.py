"""No-refit future out-of-sample evaluation for frozen candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from yenibot.experiment.common import _cfg, _rank_ic_for_frame, _table_markdown, _write_json
from yenibot.experiment.frozen import (
    frozen_manifest_source_run_dir,
    verify_frozen_manifest_artifacts,
)
from yenibot.experiment.future_oos_diagnostics import (
    future_oos_diagnostic_frames,
    future_oos_failure_markdown,
    future_oos_failure_summary,
    future_oos_model_metrics,
)
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
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    aggregated_predictions: dict[str, pd.DataFrame] = {}
    raw_predictions: dict[str, pd.DataFrame] = {}
    source_run_dir = frozen_manifest_source_run_dir(manifest, run_dir=run_dir)
    for component in manifest.get("components", []) or []:
        profile = str(component["profile"])
        raw = _predict_holdout_for_profile(
            scope_dir=source_run_dir / str(component["scope_relative_path"]),
            manifest={
                "profile": profile,
                "feature_columns": list(component["feature_columns"]),
            },
            holdout_context=future_context,
            holdout_start=future_start,
            config=config,
        )
        aggregated = _aggregate_holdout_predictions(raw, profile=profile)
        aggregated_predictions[profile] = aggregated
        raw_predictions[profile] = raw
    return aggregated_predictions, raw_predictions


def _plain_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy prediction values without propagating pandas metadata.

    Pandas compares ``DataFrame.attrs`` during merge operations. Storing a
    DataFrame inside attrs makes that comparison return another DataFrame,
    whose truth value is ambiguous. Frozen-candidate prediction frames must
    therefore cross merge/concat boundaries without attrs.
    """

    out = frame.copy()
    out.attrs = {}
    return out


def _candidate_predictions(
    manifest: dict[str, Any],
    component_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    profiles = [str(item) for item in manifest.get("profiles", []) or []]
    if manifest.get("candidate_type") == "profile":
        return _plain_frame(
            component_predictions.get(profiles[0], pd.DataFrame())
        )
    if any(profile not in component_predictions or component_predictions[profile].empty for profile in profiles):
        return pd.DataFrame()
    weights = np.asarray(manifest.get("weights", []) or [], dtype=float)
    if len(weights) != len(profiles) or weights.sum() <= 0:
        return pd.DataFrame()
    weights = weights / weights.sum()
    merged = _plain_frame(component_predictions[profiles[0]])
    merged = merged.rename(columns={"prob_long": f"prob_long_{0}"})
    for index, profile in enumerate(profiles[1:], start=1):
        other = _plain_frame(
            component_predictions[profile][["timestamp", "prob_long"]]
        ).rename(
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
    preflight: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score only rows after the frozen anchor using transform/predict operations."""

    run_path = Path(run_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    future_cfg = _cfg(config, ["experiments", "future_oos_validation"], {}) or {}
    frozen_cfg = _cfg(config, ["experiments", "frozen_candidates"], {}) or {}
    anchor_value = frozen_cfg.get("anchor_data_end")
    primary_id = str(frozen_cfg.get("primary_candidate_id", ""))
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
            "evaluation_state": "protocol_disabled",
            "primary_candidate_passed": None,
            "promotion_allowed": False,
            "promotion_block_reason": "future_oos_protocol_disabled_or_missing_anchor",
            "fit_operations_performed": 0,
            "artifact_integrity_errors": [],
            "required_candidate_errors": [],
            "optional_candidate_warnings": [],
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
    temporal_diagnostics: list[pd.DataFrame] = []
    score_band_diagnostics: list[pd.DataFrame] = []
    regime_diagnostics: list[pd.DataFrame] = []
    disagreement_diagnostics: list[pd.DataFrame] = []
    model_diagnostics: list[pd.DataFrame] = []
    artifact_integrity_errors: list[str] = []
    required_candidate_errors: list[str] = []
    optional_candidate_warnings: list[str] = []
    preflight_errors = (
        [
            f"preflight:{item}"
            for item in preflight.get("failed_checks", []) or []
        ]
        if preflight is not None and not bool(preflight.get("invariants_passed", False))
        else []
    )
    required_candidate_errors.extend(preflight_errors)

    primary_manifest = next(
        (
            manifest
            for manifest in manifests
            if str(manifest.get("candidate_id", "")) == primary_id
        ),
        None,
    )
    if primary_manifest is None:
        required_candidate_errors.append(f"missing_primary_candidate_manifest:{primary_id}")
    for manifest in manifests:
        if bool(manifest.get("available", False)):
            continue
        candidate_id = str(manifest.get("candidate_id", ""))
        messages = [
            f"{candidate_id}:{reason}"
            for reason in manifest.get("unavailable_reasons", []) or ["candidate_unavailable"]
        ]
        required = bool(
            manifest.get("required_for_evaluation", candidate_id == primary_id)
        )
        if required:
            required_candidate_errors.extend(messages)
        else:
            optional_candidate_warnings.extend(messages)

    if not labeled_path.exists():
        future_count = 0
        latest = None
        required_candidate_errors.append(f"missing_labeled_data:{labeled_path}")
        labeled = pd.DataFrame()
    else:
        labeled = pd.read_parquet(labeled_path).copy()
        labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], utc=True)
        future = labeled.loc[labeled["timestamp"] > anchor].copy()
        future_count = int(len(future))
        latest = future["timestamp"].max() if not future.empty else None

    ready = bool(future_count >= min_rows)
    if ready and not preflight_errors:
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
                continue
            candidate_id = str(manifest.get("candidate_id", ""))
            required = bool(
                manifest.get("required_for_evaluation", candidate_id == primary_id)
            )
            integrity_errors = verify_frozen_manifest_artifacts(manifest, run_dir=run_path)
            if integrity_errors:
                messages = [f"{candidate_id}:{reason}" for reason in integrity_errors]
                if required:
                    artifact_integrity_errors.extend(messages)
                    required_candidate_errors.extend(messages)
                else:
                    optional_candidate_warnings.extend(messages)
                continue
            profile_output = _profile_predictions(
                manifest=manifest,
                run_dir=run_path,
                future_context=future_context,
                future_start=future_start,
                config=config,
            )
            if isinstance(profile_output, tuple):
                components, raw_components = profile_output
            else:
                # Compatibility for tests and third-party wrappers written
                # before raw predictions became an explicit return value.
                components = profile_output
                raw_components = {}
            predictions = _candidate_predictions(manifest, components)
            if predictions.empty:
                message = f"{candidate_id}:no_predictions"
                if required:
                    required_candidate_errors.append(message)
                else:
                    optional_candidate_warnings.append(message)
                continue
            predictions["candidate_id"] = manifest["candidate_id"]
            prediction_frames.append(predictions)
            evaluation_row = _evaluate_predictions(
                predictions,
                manifest=manifest,
                config=config,
            )
            rows.append(evaluation_row)
            threshold = float((manifest.get("threshold") or {}).get("value", 0.5))
            diagnostics = future_oos_diagnostic_frames(
                predictions,
                threshold=threshold,
                block_hours=int(future_cfg.get("diagnostic_block_hours", 168)),
            )
            temporal_diagnostics.append(diagnostics["temporal_blocks"])
            score_band_diagnostics.append(diagnostics["score_bands"])
            regime_diagnostics.append(diagnostics["regime_metrics"])
            disagreement_diagnostics.append(diagnostics["ensemble_disagreement"])
            for profile in components:
                raw = raw_components.get(profile, pd.DataFrame())
                model_diagnostics.append(
                    future_oos_model_metrics(
                        raw,
                        candidate_id=candidate_id,
                        profile=profile,
                        threshold=threshold,
                    )
                )

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
    primary = evaluation.loc[evaluation["candidate_id"].astype(str) == primary_id]
    evaluation_completed = bool(ready and not primary.empty)
    primary_passed = (
        bool(primary["evidence_passed"].astype(bool).iloc[0])
        if evaluation_completed
        else None
    )
    charter_active = str(
        _cfg(config, ["validation", "charter", "active_version"], "v3_legacy")
    )
    if preflight_errors:
        evaluation_state = "blocked_required_candidate"
        promotion_block_reason = "future_oos_preflight_failed"
    elif not ready:
        evaluation_state = "waiting_for_min_rows"
        promotion_block_reason = "future_oos_not_ready"
    elif required_candidate_errors:
        evaluation_state = "blocked_required_candidate"
        promotion_block_reason = "required_candidate_unavailable_or_modified"
    elif not evaluation_completed:
        evaluation_state = "evaluation_incomplete"
        promotion_block_reason = "future_oos_evaluation_missing"
    elif primary_passed:
        evaluation_state = "evaluated_passed"
        promotion_block_reason = ""
    else:
        evaluation_state = "evaluated_failed"
        promotion_block_reason = "future_oos_candidate_failed"
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
        "evaluation_completed": evaluation_completed,
        "evaluation_state": evaluation_state,
        "primary_candidate_id": primary_id,
        "primary_candidate_passed": primary_passed,
        "active_charter_version": charter_active,
        "promotion_allowed": bool(primary_passed),
        "promotion_block_reason": promotion_block_reason,
        "fit_operations_performed": 0,
        "preflight_state": (
            str(preflight.get("state", "")) if preflight is not None else None
        ),
        "preflight_invariants_passed": (
            bool(preflight.get("invariants_passed", False))
            if preflight is not None
            else None
        ),
        "artifact_integrity_errors": artifact_integrity_errors,
        "required_candidate_errors": required_candidate_errors,
        "optional_candidate_warnings": optional_candidate_warnings,
    }
    evaluation.to_csv(report_path / "future_oos_evaluation.csv", index=False)
    markdown = _table_markdown("Future OOS Evaluation", evaluation)
    if evaluation.empty:
        markdown = (
            "# Future OOS Evaluation\n\n"
            f"- State: `{evaluation_state}`\n"
            f"- Fresh labeled rows: `{future_count}` / `{min_rows}` minimum\n"
            f"- Rows remaining: `{max(0, min_rows - future_count)}`\n"
            f"- Required candidate errors: `{len(required_candidate_errors)}`\n"
            f"- Optional candidate warnings: `{len(optional_candidate_warnings)}`\n"
            "- Model scoring performed: `False`\n"
        )
    (report_path / "future_oos_evaluation.md").write_text(markdown, encoding="utf-8")
    _write_json(
        report_path / "future_oos_evaluation.json",
        {"status": status, "rows": evaluation.to_dict(orient="records")},
    )
    _write_json(report_path / "future_oos_readiness.json", status)
    diagnostic_outputs = {
        "future_oos_temporal_blocks.csv": pd.concat(
            [item for item in temporal_diagnostics if not item.empty],
            ignore_index=True,
        )
        if any(not item.empty for item in temporal_diagnostics)
        else pd.DataFrame(),
        "future_oos_score_bands.csv": pd.concat(
            [item for item in score_band_diagnostics if not item.empty],
            ignore_index=True,
        )
        if any(not item.empty for item in score_band_diagnostics)
        else pd.DataFrame(),
        "future_oos_regime_metrics.csv": pd.concat(
            [item for item in regime_diagnostics if not item.empty],
            ignore_index=True,
        )
        if any(not item.empty for item in regime_diagnostics)
        else pd.DataFrame(),
        "future_oos_ensemble_disagreement.csv": pd.concat(
            [item for item in disagreement_diagnostics if not item.empty],
            ignore_index=True,
        )
        if any(not item.empty for item in disagreement_diagnostics)
        else pd.DataFrame(),
        "future_oos_model_metrics.csv": pd.concat(
            [item for item in model_diagnostics if not item.empty],
            ignore_index=True,
        )
        if any(not item.empty for item in model_diagnostics)
        else pd.DataFrame(),
    }
    for filename, frame in diagnostic_outputs.items():
        frame.to_csv(report_path / filename, index=False)
    if evaluation_completed:
        primary_row = primary.iloc[0].to_dict()
        summary = future_oos_failure_summary(
            primary_row,
            temporal_blocks=diagnostic_outputs["future_oos_temporal_blocks.csv"].loc[
                lambda frame: frame.get(
                    "candidate_id",
                    pd.Series(index=frame.index, dtype=str),
                ).astype(str).eq(primary_id)
            ],
            ensemble_disagreement=diagnostic_outputs[
                "future_oos_ensemble_disagreement.csv"
            ].loc[
                lambda frame: frame.get(
                    "candidate_id",
                    pd.Series(index=frame.index, dtype=str),
                ).astype(str).eq(primary_id)
            ],
            model_metrics=diagnostic_outputs["future_oos_model_metrics.csv"].loc[
                lambda frame: frame.get(
                    "candidate_id",
                    pd.Series(index=frame.index, dtype=str),
                ).astype(str).eq(primary_id)
            ],
        )
        _write_json(report_path / "future_oos_failure_summary.json", summary)
        (report_path / "future_oos_failure_summary.md").write_text(
            future_oos_failure_markdown(summary),
            encoding="utf-8",
        )
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_parquet(report_path / "future_oos_predictions.parquet", index=False)
        predictions.head(200).to_csv(
            report_path / "future_oos_prediction_sample.csv",
            index=False,
        )
    return evaluation, status
