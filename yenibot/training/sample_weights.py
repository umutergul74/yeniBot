from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import numpy as np
import pandas as pd


AUDIT_COLUMNS = [
    "fold",
    "component",
    "enabled",
    "row_count",
    "mean_weight",
    "std_weight",
    "min_weight",
    "p10_weight",
    "p50_weight",
    "p90_weight",
    "p99_weight",
    "max_weight",
    "p90_p10_spread",
    "dominant_weight_fraction",
    "active_fraction",
    "effective_sample_size",
    "effective_sample_fraction",
    "selected_column_count",
    "selected_columns",
    "horizon_bars",
    "event_quantile",
    "event_strength",
    "event_aggregation",
    "notes",
]


def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
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


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def _effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(weights, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    if clean.size == 0:
        return 0.0, 0.0
    ess = float(clean.sum() ** 2 / (np.square(clean).sum() + 1e-12))
    return ess, float(ess / clean.size)


def _audit_row(
    *,
    component: str,
    enabled: bool,
    weights: np.ndarray,
    selected_columns: list[str] | None = None,
    horizon_bars: int | None = None,
    event_quantile: float | None = None,
    event_strength: float | None = None,
    event_aggregation: str = "",
    active_fraction: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    clean = np.asarray(weights, dtype=float)
    clean = np.where(np.isfinite(clean), clean, np.nan)
    valid = clean[np.isfinite(clean)]
    ess, ess_fraction = _effective_sample_size(valid)
    if valid.size == 0:
        stats = {
            "mean_weight": np.nan,
            "std_weight": np.nan,
            "min_weight": np.nan,
            "p10_weight": np.nan,
            "p50_weight": np.nan,
            "p90_weight": np.nan,
            "p99_weight": np.nan,
            "max_weight": np.nan,
            "p90_p10_spread": np.nan,
            "dominant_weight_fraction": np.nan,
        }
    else:
        p10 = float(np.quantile(valid, 0.10))
        p90 = float(np.quantile(valid, 0.90))
        _, counts = np.unique(np.round(valid, 6), return_counts=True)
        stats = {
            "mean_weight": float(valid.mean()),
            "std_weight": float(valid.std(ddof=0)),
            "min_weight": float(valid.min()),
            "p10_weight": p10,
            "p50_weight": float(np.quantile(valid, 0.50)),
            "p90_weight": p90,
            "p99_weight": float(np.quantile(valid, 0.99)),
            "max_weight": float(valid.max()),
            "p90_p10_spread": float(p90 - p10),
            "dominant_weight_fraction": float(counts.max() / valid.size),
        }
    selected_columns = selected_columns or []
    return {
        "fold": np.nan,
        "component": component,
        "enabled": bool(enabled),
        "row_count": int(len(clean)),
        **stats,
        "effective_sample_size": ess,
        "effective_sample_fraction": ess_fraction,
        "selected_column_count": int(len(selected_columns)),
        "selected_columns": ",".join(selected_columns),
        "horizon_bars": np.nan if horizon_bars is None else int(horizon_bars),
        "event_quantile": np.nan if event_quantile is None else float(event_quantile),
        "event_strength": np.nan if event_strength is None else float(event_strength),
        "event_aggregation": str(event_aggregation),
        "active_fraction": np.nan if active_fraction is None else float(active_fraction),
        "notes": notes,
    }


def _normalize(weights: np.ndarray, *, min_weight: float, max_weight: float, normalize_mean: bool) -> np.ndarray:
    out = np.asarray(weights, dtype=float)
    out = np.where(np.isfinite(out) & (out > 0), out, 1.0)
    lower = float(min_weight)
    upper = float(max_weight)
    out = np.clip(out, lower, upper)
    if normalize_mean:
        for _ in range(3):
            mean = float(out.mean()) if len(out) else 1.0
            if not np.isfinite(mean) or mean <= 0:
                break
            out = np.clip(out / mean, lower, upper)
        mean = float(out.mean()) if len(out) else 1.0
        if np.isfinite(mean) and mean > 0:
            out = out / mean
    out = np.clip(out, lower, upper)
    return out.astype("float32")


def label_uniqueness_weights(
    frame: pd.DataFrame,
    *,
    horizon_bars: int,
    power: float = 1.0,
) -> np.ndarray:
    """Estimate train-fold-only label uniqueness for overlapping fixed-horizon labels.

    Each row's target is an event spanning the row through at most
    ``horizon_bars`` future rows inside the same train fold. The weight is the
    average inverse concurrency over that event span, normalized later by the
    caller. This deliberately does not look into validation/test rows.
    """

    n = int(len(frame))
    if n == 0:
        return np.asarray([], dtype="float32")
    horizon = max(1, int(horizon_bars))
    starts = np.arange(n, dtype=int)
    ends = np.minimum(starts + horizon, n - 1)
    if "exit_timestamp" in frame.columns and "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        exit_timestamps = pd.to_datetime(frame["exit_timestamp"], utc=True, errors="coerce")
        valid_exit = exit_timestamps.notna() & timestamps.notna()
        if valid_exit.any():
            ts_values = timestamps.astype("int64").to_numpy()
            exit_values = exit_timestamps.astype("int64").to_numpy()
            mapped = np.searchsorted(ts_values, exit_values, side="left")
            local_exit = (mapped >= starts) & (mapped < n)
            usable = valid_exit.to_numpy() & local_exit
            ends[usable] = mapped[usable]

    diff = np.zeros(n + 1, dtype=float)
    for start, end in zip(starts, ends):
        diff[start] += 1.0
        diff[end + 1] -= 1.0
    concurrency = np.cumsum(diff[:-1])
    inverse = 1.0 / np.maximum(concurrency, 1.0)
    weights = np.empty(n, dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(inverse)])
    for idx, (start, end) in enumerate(zip(starts, ends)):
        weights[idx] = (prefix[end + 1] - prefix[start]) / max(1, end - start + 1)
    if float(power) != 1.0:
        weights = np.power(np.maximum(weights, 1e-12), float(power))
    return weights.astype("float32")


def _matching_columns(columns: list[str], patterns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        if any(fnmatch(column, pattern) for pattern in patterns):
            selected.append(column)
    return sorted(dict.fromkeys(selected))


def event_strength_weights(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    include_patterns: list[str],
    quantile: float,
    strength: float,
    use_abs: bool = True,
    power: float = 1.0,
    aggregation: str = "mean_feature_rank",
) -> tuple[np.ndarray, list[str], str, dict[str, float | str]]:
    """Return smooth train-fold event emphasis from configured feature families."""

    n = int(len(frame))
    if n == 0:
        return (
            np.asarray([], dtype="float32"),
            [],
            "empty_frame",
            {"aggregation": str(aggregation), "active_fraction": 0.0},
        )
    selected = _matching_columns(feature_columns, include_patterns)
    selected = [column for column in selected if column in frame.columns]
    if not selected:
        return (
            np.ones(n, dtype="float32"),
            [],
            "no_matching_event_columns",
            {"aggregation": str(aggregation), "active_fraction": 0.0},
        )

    values = frame[selected].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if use_abs:
        values = values.abs()
    rankable = [column for column in selected if values[column].notna().sum() >= 3]
    if not rankable:
        return (
            np.ones(n, dtype="float32"),
            selected,
            "event_columns_not_rankable",
            {"aggregation": str(aggregation), "active_fraction": 0.0},
        )
    values = values[rankable]

    method = str(aggregation).strip().lower()
    if method == "mean_abs_then_rank":
        raw_score = values.mean(axis=1)
        event_score = raw_score.rank(pct=True, method="average").fillna(0.5).to_numpy(dtype=float)
    elif method == "max_feature_rank":
        ranked = values.rank(pct=True, method="average").fillna(0.5)
        event_score = ranked.max(axis=1).to_numpy(dtype=float)
    elif method == "mean_feature_rank":
        ranked = values.rank(pct=True, method="average").fillna(0.5)
        event_score = ranked.mean(axis=1).to_numpy(dtype=float)
    else:
        raise ValueError(
            "Unknown sample-weight event aggregation "
            f"{aggregation!r}; expected mean_abs_then_rank, max_feature_rank, "
            "or mean_feature_rank"
        )

    q = float(np.clip(quantile, 0.0, 0.99))
    denom = max(1e-8, 1.0 - q)
    tail = np.clip((event_score - q) / denom, 0.0, 1.0)
    if float(power) != 1.0:
        tail = np.power(tail, float(power))
    weights = 1.0 + float(strength) * tail
    return (
        weights.astype("float32"),
        rankable,
        "ok",
        {
            "aggregation": method,
            "active_fraction": float(np.mean(tail > 0)),
            "event_score_p50": float(np.quantile(event_score, 0.50)),
            "event_score_p90": float(np.quantile(event_score, 0.90)),
        },
    )


def _validate_event_weight_effectiveness(
    weights: np.ndarray,
    diagnostics: dict[str, float | str],
    guard: dict[str, Any],
) -> None:
    """Fail before training when an enabled event component is effectively inert."""

    if not bool(guard.get("enabled", False)):
        return
    clean = np.asarray(weights, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    if clean.size == 0:
        raise ValueError("Event sample weighting produced no finite positive weights")

    p10 = float(np.quantile(clean, 0.10))
    p90 = float(np.quantile(clean, 0.90))
    spread = p90 - p10
    _, counts = np.unique(np.round(clean, 6), return_counts=True)
    dominant_fraction = float(counts.max() / clean.size)
    active_fraction = float(diagnostics.get("active_fraction", 0.0))

    failures: list[str] = []
    min_active = float(guard.get("min_active_fraction", 0.10))
    max_active = float(guard.get("max_active_fraction", 0.35))
    min_spread = float(guard.get("min_p90_p10_spread", 0.03))
    max_dominant = float(guard.get("max_dominant_weight_fraction", 0.90))
    if active_fraction < min_active or active_fraction > max_active:
        failures.append(
            f"active_fraction={active_fraction:.6f} outside [{min_active:.6f}, {max_active:.6f}]"
        )
    if spread < min_spread:
        failures.append(f"p90_p10_spread={spread:.6f} < {min_spread:.6f}")
    if dominant_fraction > max_dominant:
        failures.append(
            f"dominant_weight_fraction={dominant_fraction:.6f} > {max_dominant:.6f}"
        )
    if failures:
        raise ValueError(
            "Enabled event sample weighting is effectively a no-op: "
            + "; ".join(failures)
        )


def build_sample_weights(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    config: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build train-fold sample weights plus an audit table.

    The default is disabled and returns all-ones. When enabled, every fitted
    quantity is computed only from ``train_frame``. Validation/test rows never
    enter the calculation.
    """

    n = int(len(train_frame))
    cfg = _cfg(config, ["training", "sample_weighting"], {}) or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        weights = np.ones(n, dtype="float32")
        audit = pd.DataFrame(
            [
                _audit_row(
                    component="combined",
                    enabled=False,
                    weights=weights,
                    notes="sample_weighting_disabled",
                )
            ],
            columns=AUDIT_COLUMNS,
        )
        return weights, audit

    min_weight = float(cfg.get("min_weight", 0.25))
    max_weight = float(cfg.get("max_weight", 3.0))
    normalize_mean = bool(cfg.get("normalize_mean", True))
    components = cfg.get("components", {}) or {}
    weights = np.ones(n, dtype="float32")
    audit_rows: list[dict[str, Any]] = []

    uniqueness_cfg = components.get("uniqueness", {}) or {}
    uniqueness_enabled = bool(uniqueness_cfg.get("enabled", True))
    horizon = int(
        uniqueness_cfg.get(
            "horizon_bars",
            _cfg(config, ["labeling", "max_holding_bars"], 10),
        )
        or _cfg(config, ["labeling", "max_holding_bars"], 10)
    )
    if uniqueness_enabled:
        uniqueness = label_uniqueness_weights(
            train_frame,
            horizon_bars=horizon,
            power=float(uniqueness_cfg.get("power", 1.0)),
        )
        weights *= uniqueness
        audit_rows.append(
            _audit_row(
                component="uniqueness",
                enabled=True,
                weights=_normalize(
                    uniqueness,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    normalize_mean=True,
                ),
                horizon_bars=horizon,
                notes="fixed_horizon_inverse_concurrency_train_fold_only",
            )
        )
    else:
        audit_rows.append(
            _audit_row(
                component="uniqueness",
                enabled=False,
                weights=np.ones(n, dtype="float32"),
                horizon_bars=horizon,
                notes="uniqueness_component_disabled",
            )
        )

    event_cfg = components.get("event", {}) or {}
    event_enabled = bool(event_cfg.get("enabled", False))
    event_columns: list[str] = []
    event_notes = "event_component_disabled"
    event_diagnostics: dict[str, float | str] = {
        "aggregation": str(event_cfg.get("aggregation", "mean_feature_rank")),
        "active_fraction": 0.0,
    }
    if event_enabled:
        event_weights, event_columns, event_notes, event_diagnostics = event_strength_weights(
            train_frame,
            feature_columns,
            include_patterns=[str(item) for item in event_cfg.get("include_patterns", []) or []],
            quantile=float(event_cfg.get("quantile", 0.80)),
            strength=float(event_cfg.get("strength", 0.35)),
            use_abs=bool(event_cfg.get("use_abs", True)),
            power=float(event_cfg.get("power", 1.0)),
            aggregation=str(event_cfg.get("aggregation", "mean_feature_rank")),
        )
        normalized_event_weights = _normalize(
            event_weights,
            min_weight=min_weight,
            max_weight=max_weight,
            normalize_mean=True,
        )
        _validate_event_weight_effectiveness(
            normalized_event_weights,
            event_diagnostics,
            event_cfg.get("effectiveness_guard", {}) or {},
        )
        weights *= event_weights
        audit_rows.append(
            _audit_row(
                component="event",
                enabled=True,
                weights=normalized_event_weights,
                selected_columns=event_columns,
                event_quantile=float(event_cfg.get("quantile", 0.80)),
                event_strength=float(event_cfg.get("strength", 0.35)),
                event_aggregation=str(event_diagnostics.get("aggregation", "")),
                active_fraction=float(event_diagnostics.get("active_fraction", 0.0)),
                notes=event_notes,
            )
        )
    else:
        audit_rows.append(
            _audit_row(
                component="event",
                enabled=False,
                weights=np.ones(n, dtype="float32"),
                selected_columns=[],
                notes=event_notes,
            )
        )

    weights = _normalize(
        weights,
        min_weight=min_weight,
        max_weight=max_weight,
        normalize_mean=normalize_mean,
    )
    audit_rows.append(
        _audit_row(
            component="combined",
            enabled=True,
            weights=weights,
            selected_columns=event_columns,
            horizon_bars=horizon,
            event_quantile=event_cfg.get("quantile", np.nan) if event_enabled else None,
            event_strength=event_cfg.get("strength", np.nan) if event_enabled else None,
            event_aggregation=str(event_diagnostics.get("aggregation", "")) if event_enabled else "",
            active_fraction=float(event_diagnostics.get("active_fraction", 0.0))
            if event_enabled
            else None,
            notes="multiplicative_components_normalized_train_fold_only",
        )
    )
    return weights, pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
