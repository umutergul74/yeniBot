"""Leakage-safe primitives for the post-failure rolling research cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from yenibot.experiment.common import _hash_payload, _rank_ic_for_frame, _write_json
from yenibot.experiment.configuration import profile_config
from yenibot.experiment.holdout import _predict_holdout_for_profile
from yenibot.training import PurgedWalkForwardCV

__all__ = [
    "RollingResearchWindow",
    "rolling_origin_schedule",
    "recency_weights",
    "aggregate_recency_predictions",
    "research_protocol_payload",
    "run_recency_ensemble_research",
]


@dataclass(frozen=True)
class RollingResearchWindow:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    latest_eligible_model_fold: int
    eligible_model_folds: tuple[int, ...]


def rolling_origin_schedule(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return the exact causal fold schedule used to compare deployment policies."""

    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()
    cv_cfg = config.get("walk_forward", {}) or {}
    cv = PurgedWalkForwardCV(
        train_bars=int(cv_cfg["train_bars"]),
        val_bars=int(cv_cfg["val_bars"]),
        test_bars=int(cv_cfg["test_bars"]),
        step_bars=int(cv_cfg["step_bars"]),
        purge_bars=int(cv_cfg["purge_bars"]),
        embargo_bars=int(cv_cfg["embargo_bars"]),
    )
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    rows: list[dict[str, Any]] = []
    for fold in cv.split(len(frame)):
        eligible = tuple(range(int(fold.fold) + 1))
        window = RollingResearchWindow(
            fold=int(fold.fold),
            train_start=timestamps.iloc[int(fold.train[0])].isoformat(),
            train_end=timestamps.iloc[int(fold.train[-1])].isoformat(),
            validation_start=timestamps.iloc[int(fold.val[0])].isoformat(),
            validation_end=timestamps.iloc[int(fold.val[-1])].isoformat(),
            test_start=timestamps.iloc[int(fold.test[0])].isoformat(),
            test_end=timestamps.iloc[int(fold.test[-1])].isoformat(),
            latest_eligible_model_fold=int(fold.fold),
            eligible_model_folds=eligible,
        )
        row = asdict(window)
        row["eligible_model_folds"] = ",".join(str(item) for item in eligible)
        row["eligible_model_count"] = len(eligible)
        row["future_model_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def recency_weights(
    model_folds: Iterable[int],
    *,
    target_fold: int,
    policy: str,
    recent_k: int | None = None,
    half_life_folds: float | None = None,
) -> dict[int, float]:
    """Create normalized weights using only models available by the target fold."""

    eligible = sorted({int(fold) for fold in model_folds if int(fold) <= int(target_fold)})
    if not eligible:
        raise ValueError("No causally eligible models are available for the target fold")
    future = sorted({int(fold) for fold in model_folds if int(fold) > int(target_fold)})
    if future:
        raise ValueError(f"Future model folds are not eligible for target fold {target_fold}: {future}")
    if policy == "latest_only":
        selected = [eligible[-1]]
        raw = np.ones(1, dtype=float)
    elif policy == "equal_recent_k":
        k = max(1, int(recent_k or 1))
        selected = eligible[-k:]
        raw = np.ones(len(selected), dtype=float)
    elif policy == "exponential_decay":
        half_life = float(half_life_folds or 1.0)
        if half_life <= 0:
            raise ValueError("half_life_folds must be positive")
        selected = eligible
        ages = np.asarray([target_fold - fold for fold in selected], dtype=float)
        raw = np.power(0.5, ages / half_life)
    else:
        raise ValueError(f"Unknown recency policy: {policy}")
    normalized = raw / raw.sum()
    return {
        int(fold): float(weight)
        for fold, weight in zip(selected, normalized, strict=True)
    }


def aggregate_recency_predictions(
    raw_predictions: pd.DataFrame,
    *,
    target_fold: int,
    policy: str,
    recent_k: int | None = None,
    half_life_folds: float | None = None,
) -> pd.DataFrame:
    """Aggregate cross-model predictions after enforcing as-of eligibility."""

    required = {"timestamp", "model_fold", "prob_long"}
    missing = sorted(required.difference(raw_predictions.columns))
    if missing:
        raise ValueError(f"Missing recency prediction columns: {missing}")
    frame = raw_predictions.copy()
    frame["model_fold"] = pd.to_numeric(frame["model_fold"], errors="raise").astype(int)
    observed_folds = sorted(frame["model_fold"].unique().tolist())
    weights = recency_weights(
        observed_folds,
        target_fold=target_fold,
        policy=policy,
        recent_k=recent_k,
        half_life_folds=half_life_folds,
    )
    frame = frame.loc[frame["model_fold"].isin(weights)].copy()
    frame["model_weight"] = frame["model_fold"].map(weights).astype(float)
    frame["weighted_prob_long"] = (
        pd.to_numeric(frame["prob_long"], errors="coerce") * frame["model_weight"]
    )
    first_columns = [
        column
        for column in (
            "label",
            "forward_return",
            "tb_return",
            "hit_type",
        )
        if column in frame.columns
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        column: (column, "first") for column in first_columns
    }
    aggregations["prob_long"] = ("weighted_prob_long", "sum")
    aggregations["weight_sum"] = ("model_weight", "sum")
    aggregations["model_count"] = ("model_fold", "nunique")
    out = frame.groupby("timestamp", as_index=False).agg(**aggregations)
    if not np.allclose(out["weight_sum"].to_numpy(dtype=float), 1.0):
        raise ValueError("Recency weights do not sum to one for every prediction timestamp")
    out["target_fold"] = int(target_fold)
    out["policy"] = policy
    out["policy_parameters"] = (
        f"recent_k={recent_k}"
        if policy == "equal_recent_k"
        else f"half_life_folds={half_life_folds}"
        if policy == "exponential_decay"
        else ""
    )
    return out.sort_values("timestamp").reset_index(drop=True)


def research_protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Expose the immutable rules for the next research cycle in reports."""

    cycle = (
        config.get("experiments", {}).get("next_research_cycle", {}) or {}
    )
    return {
        "status": cycle.get("status", "not_configured"),
        "source_failed_candidate_id": cycle.get("source_failed_candidate_id"),
        "failed_oos_role": cycle.get("failed_oos_role"),
        "same_window_selection_allowed": bool(
            cycle.get("same_window_selection_allowed", False)
        ),
        "new_future_oos_anchor_required": bool(
            cycle.get("new_future_oos_anchor_required", True)
        ),
        "rolling_origin": cycle.get("rolling_origin", {}),
        "recency_ensemble": cycle.get("recency_ensemble", {}),
        "phase2_code_allowed": False,
    }


def _select_validation_threshold(
    predictions: pd.DataFrame,
    *,
    max_pred_long_rate: float,
    min_precision: float,
) -> dict[str, float]:
    labels = pd.to_numeric(predictions["label"], errors="coerce").astype(int)
    scores = pd.to_numeric(predictions["prob_long"], errors="coerce")
    quantiles = np.linspace(0.05, 0.95, 37)
    thresholds = sorted(
        {
            0.5,
            *[
                float(value)
                for value in scores.quantile(quantiles).dropna().tolist()
            ],
        }
    )
    candidates = []
    for threshold in thresholds:
        selected = scores >= threshold
        pred_rate = float(selected.mean())
        precision = float(precision_score(labels, selected, zero_division=0))
        recall = float(recall_score(labels, selected, zero_division=0))
        f1 = float(f1_score(labels, selected, zero_division=0))
        candidates.append(
            {
                "threshold": threshold,
                "pred_long_rate": pred_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "guarded": pred_rate <= max_pred_long_rate and precision >= min_precision,
            }
        )
    guarded = [item for item in candidates if item["guarded"]]
    pool = guarded or candidates
    return max(
        pool,
        key=lambda item: (
            item["f1"],
            item["precision"],
            -item["pred_long_rate"],
        ),
    )


def _policy_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float]:
    frame = predictions.dropna(subset=["prob_long", "label", "forward_return"]).copy()
    labels = frame["label"].astype(int)
    scores = pd.to_numeric(frame["prob_long"], errors="coerce")
    returns = pd.to_numeric(frame["forward_return"], errors="coerce")
    selected = scores >= threshold
    prevalence = float(labels.mean())
    precision = float(precision_score(labels, selected, zero_division=0))
    top_count = max(1, int(np.ceil(len(frame) * 0.10)))
    top = frame.assign(_score=scores).nlargest(top_count, "_score")
    top_label_rate = float(pd.to_numeric(top["label"], errors="coerce").mean())
    prauc = (
        float(average_precision_score(labels, scores))
        if labels.nunique(dropna=True) > 1
        else np.nan
    )
    return {
        "rows": int(len(frame)),
        "rank_ic": _rank_ic_for_frame(frame),
        "label_prevalence": prevalence,
        "pred_long_rate": float(selected.mean()),
        "precision": precision,
        "recall": float(recall_score(labels, selected, zero_division=0)),
        "f1": float(f1_score(labels, selected, zero_division=0)),
        "prauc": prauc,
        "prauc_lift_vs_prevalence": (
            prauc / prevalence if prevalence > 0 and np.isfinite(prauc) else np.nan
        ),
        "precision_lift_vs_prevalence": (
            precision / prevalence if prevalence > 0 else np.nan
        ),
        "top_10_lift": (
            top_label_rate / prevalence if prevalence > 0 else np.nan
        ),
        "top_10_forward_return": float(
            pd.to_numeric(top["forward_return"], errors="coerce").mean()
        ),
        "selected_forward_return": (
            float(returns[selected].mean()) if bool(selected.any()) else np.nan
        ),
    }


def _policy_kwargs(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": str(policy["policy"]),
        "recent_k": policy.get("recent_k"),
        "half_life_folds": policy.get("half_life_folds"),
    }


def _research_summary(by_fold: pd.DataFrame) -> pd.DataFrame:
    if by_fold.empty:
        return pd.DataFrame()
    rows = []
    for policy_name, part in by_fold.groupby("policy_name", sort=False):
        rank_ic = pd.to_numeric(part["rank_ic"], errors="coerce")
        rows.append(
            {
                "policy_name": str(policy_name),
                "fold_count": int(len(part)),
                "mean_rank_ic": float(rank_ic.mean()),
                "std_rank_ic": float(rank_ic.std(ddof=0)),
                "positive_ic_fraction": float((rank_ic > 0).mean()),
                "worst_5_rank_ic_mean": float(rank_ic.nsmallest(min(5, len(rank_ic))).mean()),
                "mean_f1": float(pd.to_numeric(part["f1"], errors="coerce").mean()),
                "mean_prauc_lift": float(
                    pd.to_numeric(
                        part["prauc_lift_vs_prevalence"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_precision_lift": float(
                    pd.to_numeric(
                        part["precision_lift_vs_prevalence"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_top_10_lift": float(
                    pd.to_numeric(part["top_10_lift"], errors="coerce").mean()
                ),
                "positive_top_10_return_fraction": float(
                    (
                        pd.to_numeric(
                            part["top_10_forward_return"],
                            errors="coerce",
                        )
                        > 0
                    ).mean()
                ),
                "positive_selected_return_fraction": float(
                    (
                        pd.to_numeric(
                            part["selected_forward_return"],
                            errors="coerce",
                        )
                        > 0
                    ).mean()
                ),
                "mean_pred_long_rate": float(
                    pd.to_numeric(part["pred_long_rate"], errors="coerce").mean()
                ),
                "selection_data_role": "historical_walk_forward_only",
                "failed_future_oos_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_rank_ic", "std_rank_ic"],
        ascending=[False, True],
    )


def run_recency_ensemble_research(
    *,
    frame: pd.DataFrame,
    scope_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Cross-score historical folds and compare causal deployment ensembles.

    This performs inference only. It does not fit models, scalers, HMMs, or
    thresholds on test rows.
    """

    scope_path = Path(scope_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = scope_path / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = str(manifest["profile"])
    feature_columns = list(manifest["feature_columns"])
    research_cfg = (
        config.get("experiments", {})
        .get("next_research_cycle", {})
        .get("recency_ensemble", {})
        or {}
    )
    policies = list(research_cfg.get("policies", []) or [])
    if not bool(research_cfg.get("enabled", False)) or not policies:
        return {
            "enabled": False,
            "status": "disabled",
            "summary": pd.DataFrame(),
            "by_fold": pd.DataFrame(),
        }
    profile_cfg = profile_config(config, profile)
    schedule = rolling_origin_schedule(frame, profile_cfg)
    if schedule.empty:
        raise ValueError("Rolling-origin schedule is empty")
    available_model_folds = sorted(
        int(path.stem.rsplit("_", 1)[-1])
        for path in scope_path.glob("model_fold_*.pt")
    )
    if not available_model_folds:
        raise FileNotFoundError(f"No model_fold_*.pt files found under {scope_path}")
    signature = _hash_payload(
        {
            "profile": profile,
            "feature_columns": feature_columns,
            "training_signature": manifest.get("signature_hash"),
            "data_start": str(pd.to_datetime(frame["timestamp"], utc=True).min()),
            "data_end": str(pd.to_datetime(frame["timestamp"], utc=True).max()),
            "rows": int(len(frame)),
            "policies": policies,
            "walk_forward": profile_cfg.get("walk_forward", {}),
        }
    )
    summary_path = output_path / "recency_ensemble_summary.csv"
    by_fold_path = output_path / "recency_ensemble_by_fold.csv"
    protocol_path = output_path / "recency_ensemble_manifest.json"
    if summary_path.exists() and by_fold_path.exists() and protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous.get("signature_hash") == signature:
            return {
                "enabled": True,
                "status": "reused",
                "summary": pd.read_csv(summary_path),
                "by_fold": pd.read_csv(by_fold_path),
                "schedule": pd.read_csv(output_path / "recency_ensemble_schedule.csv"),
                "eligibility_audit": pd.read_csv(
                    output_path / "recency_ensemble_eligibility_audit.csv"
                ),
                "signature_hash": signature,
            }
    cv_cfg = profile_cfg.get("walk_forward", {}) or {}
    cv = PurgedWalkForwardCV(
        train_bars=int(cv_cfg["train_bars"]),
        val_bars=int(cv_cfg["val_bars"]),
        test_bars=int(cv_cfg["test_bars"]),
        step_bars=int(cv_cfg["step_bars"]),
        purge_bars=int(cv_cfg["purge_bars"]),
        embargo_bars=int(cv_cfg["embargo_bars"]),
    )
    seq_len = int(profile_cfg.get("model", {}).get("seq_len", 64))
    threshold_cfg = profile_cfg.get("validation", {}).get("threshold_checks", {}) or {}
    max_pred_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    schedule_by_fold = schedule.set_index("fold")
    for fold in cv.split(len(frame)):
        target_fold = int(fold.fold)
        eligible = [item for item in available_model_folds if item <= target_fold]
        if target_fold not in eligible:
            continue
        target_test_start = pd.to_datetime(
            schedule_by_fold.loc[target_fold, "test_start"],
            utc=True,
        )
        for model_fold in eligible:
            validation_end = pd.to_datetime(
                schedule_by_fold.loc[model_fold, "validation_end"],
                utc=True,
            )
            passed = bool(validation_end < target_test_start)
            eligibility_rows.append(
                {
                    "target_fold": target_fold,
                    "model_fold": model_fold,
                    "model_validation_end": validation_end,
                    "target_test_start": target_test_start,
                    "eligible": passed,
                    "rule": "model_validation_end_before_target_test_start",
                }
            )
            if not passed:
                raise ValueError(
                    "Recency ensemble eligibility violation: "
                    f"model fold {model_fold} validation ends at {validation_end}, "
                    f"target fold {target_fold} starts at {target_test_start}"
                )
        context_start = max(0, int(fold.val[0]) - seq_len + 1)
        context_end = int(fold.test[-1]) + 1
        context = frame.iloc[context_start:context_end].copy().reset_index(drop=True)
        cache_path = output_path / (
            f"cross_predictions_{signature[:12]}_fold_{target_fold:03d}.parquet"
        )
        if cache_path.exists():
            raw = pd.read_parquet(cache_path)
        else:
            raw = _predict_holdout_for_profile(
                scope_dir=scope_path,
                manifest={
                    "profile": profile,
                    "feature_columns": feature_columns,
                },
                holdout_context=context,
                holdout_start=timestamps.iloc[int(fold.val[0])],
                holdout_end=timestamps.iloc[int(fold.test[-1])],
                model_folds=set(eligible),
                config=profile_cfg,
            )
            compact_columns = [
                column
                for column in (
                    "timestamp",
                    "model_fold",
                    "prob_long",
                    "label",
                    "forward_return",
                    "tb_return",
                    "hit_type",
                )
                if column in raw.columns
            ]
            raw = raw[compact_columns].copy()
            raw.to_parquet(cache_path, index=False)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        val_raw = raw.loc[
            raw["timestamp"].between(
                timestamps.iloc[int(fold.val[0])],
                timestamps.iloc[int(fold.val[-1])],
            )
        ].copy()
        test_raw = raw.loc[
            raw["timestamp"].between(
                timestamps.iloc[int(fold.test[0])],
                timestamps.iloc[int(fold.test[-1])],
            )
        ].copy()
        for policy in policies:
            kwargs = _policy_kwargs(policy)
            validation = aggregate_recency_predictions(
                val_raw,
                target_fold=target_fold,
                **kwargs,
            )
            threshold = _select_validation_threshold(
                validation,
                max_pred_long_rate=max_pred_rate,
                min_precision=min_precision,
            )
            test = aggregate_recency_predictions(
                test_raw,
                target_fold=target_fold,
                **kwargs,
            )
            rows.append(
                {
                    "profile": profile,
                    "policy_name": str(policy["name"]),
                    "policy": str(policy["policy"]),
                    "target_fold": target_fold,
                    "test_start": timestamps.iloc[int(fold.test[0])],
                    "test_end": timestamps.iloc[int(fold.test[-1])],
                    "eligible_model_count": len(eligible),
                    "latest_eligible_model_fold": max(eligible),
                    "validation_threshold": threshold["threshold"],
                    "validation_f1": threshold["f1"],
                    "validation_pred_long_rate": threshold["pred_long_rate"],
                    "threshold_guarded": bool(threshold["guarded"]),
                    **_policy_metrics(test, threshold=threshold["threshold"]),
                }
            )
    by_fold = pd.DataFrame(rows)
    summary = _research_summary(by_fold)
    eligibility_audit = pd.DataFrame(eligibility_rows)
    schedule.to_csv(output_path / "recency_ensemble_schedule.csv", index=False)
    eligibility_audit.to_csv(
        output_path / "recency_ensemble_eligibility_audit.csv",
        index=False,
    )
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    protocol = {
        **research_protocol_payload(config),
        "signature_hash": signature,
        "profile": profile,
        "source_scope_dir": str(scope_path),
        "fit_operations_performed": 0,
        "threshold_selection": "validation_only_per_target_fold",
        "test_labels_used_for_policy_selection": False,
        "failed_future_oos_used_for_policy_selection": False,
        "eligibility_audit_passed": bool(
            not eligibility_audit.empty and eligibility_audit["eligible"].all()
        ),
        "status": "completed",
    }
    _write_json(protocol_path, protocol)
    return {
        "enabled": True,
        "status": "completed",
        "summary": summary,
        "by_fold": by_fold,
        "schedule": schedule,
        "eligibility_audit": eligibility_audit,
        "signature_hash": signature,
    }
