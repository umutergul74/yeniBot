from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PHASE2_INPUT_ADAPTER_VERSION = "phase2_input_adapter_v1"
SUPPORTED_CANDIDATE_TYPES = {"profile", "recency_profile"}
PREDICTION_FILENAMES = ("predictions_all.parquet", "predictions_all.csv")


@dataclass(frozen=True)
class Phase2InputBuildResult:
    """Paths and audit metadata for generated Phase 2 sandbox inputs."""

    bars_path: Path
    signals_path: Path
    input_manifest_path: Path
    candidate_manifest_path: Path | None
    prediction_path: Path
    candidate_id: str
    threshold: float
    rows_read: int
    rows_after_split_filter: int
    rows_used: int
    duplicate_prediction_rows_collapsed: int
    bar_count: int
    signal_count: int
    split_filter: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bars_path": str(self.bars_path),
            "signals_path": str(self.signals_path),
            "input_manifest_path": str(self.input_manifest_path),
            "candidate_manifest_path": (
                str(self.candidate_manifest_path)
                if self.candidate_manifest_path is not None
                else None
            ),
            "prediction_path": str(self.prediction_path),
            "candidate_id": self.candidate_id,
            "threshold": self.threshold,
            "rows_read": self.rows_read,
            "rows_after_split_filter": self.rows_after_split_filter,
            "rows_used": self.rows_used,
            "duplicate_prediction_rows_collapsed": (
                self.duplicate_prediction_rows_collapsed
            ),
            "bar_count": self.bar_count,
            "signal_count": self.signal_count,
            "split_filter": self.split_filter,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported prediction table format: {path}")


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe component scope_relative_path: {value!r}")
    return relative


def _resolve_existing_manifest_path(
    report_dir: Path,
    manifest_path: str | Path | None,
) -> Path | None:
    if manifest_path is not None:
        explicit = Path(manifest_path)
        if not explicit.exists():
            raise FileNotFoundError(f"Frozen candidate manifest not found: {explicit}")
        return explicit

    report_manifest = report_dir / "frozen_candidate_manifest.json"
    candidate_paths: list[Path] = []
    if report_manifest.exists():
        candidate_paths.append(report_manifest)

    preflight = _read_json(report_dir / "future_oos_preflight.json")
    preflight_path = str(
        (preflight.get("primary_candidate", {}) or {}).get("manifest_path", "")
        or ""
    ).strip()
    if preflight_path:
        candidate = Path(preflight_path)
        if candidate.exists():
            candidate_paths.append(candidate)

    for candidate in candidate_paths:
        manifest = _read_json(candidate)
        if bool(manifest.get("available", False)):
            return candidate
    for candidate in candidate_paths:
        if _read_json(candidate):
            return candidate
    return None


def load_frozen_candidate_manifest(
    report_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Load the primary frozen candidate manifest used for sandbox inputs.

    The report-local ``frozen_candidate_manifest.json`` is preferred when it is
    available because Notebook 05 regenerates it from the active config. The
    immutable path recorded in ``future_oos_preflight.json`` remains a fallback
    for environments where only the checkpoint artifact is present.
    """

    report_path = Path(report_dir)
    resolved = _resolve_existing_manifest_path(report_path, manifest_path)
    if resolved is None:
        raise FileNotFoundError(
            "Could not find a frozen candidate manifest. Expected either "
            "future_oos_preflight.primary_candidate.manifest_path or "
            f"{report_path / 'frozen_candidate_manifest.json'}."
        )
    manifest = _read_json(resolved)
    if not manifest:
        raise ValueError(f"Frozen candidate manifest is empty: {resolved}")
    return manifest, resolved


def _resolve_source_run_dir(checkpoint_dir: str | Path, source_run_id: str) -> Path:
    base = Path(checkpoint_dir)
    candidates = [
        base,
        base / "experiments" / source_run_id,
        base / source_run_id,
    ]
    if base.name == "experiments":
        candidates.insert(1, base / source_run_id)
    if base.name != source_run_id:
        candidates = [item for item in candidates if item.name == source_run_id]
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    for candidate in unique:
        if candidate.exists():
            return candidate
    return unique[0] if unique else base / "experiments" / source_run_id


def _component_scope_dir(
    manifest: dict[str, Any],
    *,
    checkpoint_dir: str | Path | None,
    scope_dir: str | Path | None,
) -> Path:
    if scope_dir is not None:
        return Path(scope_dir)

    candidate_type = str(manifest.get("candidate_type", ""))
    if candidate_type not in SUPPORTED_CANDIDATE_TYPES:
        raise ValueError(
            "Phase 2 input adapter currently supports profile and "
            f"recency_profile candidates, not {candidate_type!r}."
        )
    if checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required when --scope-dir is not supplied.")

    components = list(manifest.get("components", []) or [])
    if len(components) != 1:
        raise ValueError(
            "Profile/recency frozen candidates must have exactly one component "
            f"for this adapter; found {len(components)}."
        )
    component = components[0]
    source_run_id = str(manifest.get("source_run_id", "") or "").strip()
    if not source_run_id:
        raise ValueError("Frozen candidate manifest is missing source_run_id.")
    source_run_dir = _resolve_source_run_dir(checkpoint_dir, source_run_id)
    return source_run_dir / _safe_relative_path(str(component.get("scope_relative_path", "")))


def _prediction_path(scope_dir: str | Path) -> Path:
    root = Path(scope_dir)
    for filename in PREDICTION_FILENAMES:
        path = root / filename
        if path.exists():
            return path
    expected = ", ".join(PREDICTION_FILENAMES)
    raise FileNotFoundError(f"Missing predictions in {root}; expected one of {expected}.")


def _time_column(frame: pd.DataFrame) -> str:
    for column in ("timestamp", "bar_open_time", "bar_close_time", "decision_time"):
        if column in frame.columns:
            return column
    raise ValueError("Predictions are missing a timestamp/bar time column.")


def _collapse_unique(values: pd.Series) -> str:
    unique = [
        str(value)
        for value in values.dropna().astype(str).unique().tolist()
        if str(value).strip()
    ]
    return unique[0] if len(unique) <= 1 else "|".join(unique)


def _prepare_prediction_rows(
    predictions: pd.DataFrame,
    *,
    split: str,
    score_column: str,
    atr_column: str,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    frame = predictions.copy()
    rows_read = int(len(frame))
    if split.lower() not in {"", "all", "*"} and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower() == split.lower()].copy()
    rows_after_split = int(len(frame))

    time_column = _time_column(frame)
    required = [time_column, "open", "high", "low", "close", atr_column, score_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Predictions are missing required Phase 2 columns: {missing}")

    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    numeric_columns = ["open", "high", "low", "close", atr_column, score_column]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).sort_values(time_column).reset_index(drop=True)
    duplicate_rows = int(len(frame) - frame[time_column].nunique())
    stats = {
        "rows_read": rows_read,
        "rows_after_split_filter": rows_after_split,
        "rows_used": int(len(frame)),
        "duplicate_prediction_rows_collapsed": duplicate_rows,
    }
    return frame, time_column, stats


def phase2_inputs_from_predictions(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    threshold: float,
    split: str = "test",
    score_column: str = "prob_long",
    atr_column: str = "atr_14",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Convert frozen Phase 1 predictions into causal Phase 2 input tables."""

    frame, time_column, stats = _prepare_prediction_rows(
        predictions,
        split=split,
        score_column=score_column,
        atr_column=atr_column,
    )
    bar_columns = [time_column, "open", "high", "low", "close", atr_column]
    for optional in ("volume", "quote_volume", "trades"):
        if optional in frame.columns and optional not in bar_columns:
            bar_columns.append(optional)
    bars = (
        frame[bar_columns]
        .drop_duplicates(subset=[time_column], keep="last")
        .rename(columns={time_column: "bar_close_time"})
        .reset_index(drop=True)
    )
    # Native Phase 1 timestamp is Binance bar OPEN time. Keep it explicitly;
    # features/scores become available only when that hourly bar completes.
    if time_column in {"timestamp", "bar_open_time"}:
        bars["bar_open_time"] = bars["bar_close_time"]
        bars["bar_close_time"] = bars["bar_open_time"] + pd.Timedelta(hours=1)
    if atr_column != "atr_14":
        bars = bars.rename(columns={atr_column: "atr_14"})

    frame = frame.copy()
    frame["candidate_id"] = candidate_id
    frame["threshold"] = float(threshold)
    signal_columns = [time_column, score_column, "candidate_id", "threshold"]
    passthrough_columns = [
        "profile",
        "fold",
        "split",
        "label",
        "forward_return",
        "tb_return",
        "source_row_position",
        "auxiliary_return_prediction",
    ]
    for column in passthrough_columns:
        if column in frame.columns and column not in signal_columns:
            signal_columns.append(column)

    signal_source = frame[signal_columns].copy()
    aggregations: dict[str, Any] = {
        score_column: "mean",
        "candidate_id": "first",
        "threshold": "first",
    }
    for column in signal_columns:
        if column in {time_column, score_column, "candidate_id", "threshold"}:
            continue
        if column in {"label", "forward_return", "tb_return", "auxiliary_return_prediction"}:
            signal_source[column] = pd.to_numeric(signal_source[column], errors="coerce")
            aggregations[column] = "mean"
        else:
            aggregations[column] = _collapse_unique
    signals = (
        signal_source.groupby(time_column, as_index=False)
        .agg(aggregations)
        .rename(columns={time_column: "decision_time", score_column: "prob_long"})
        .reset_index(drop=True)
    )
    if time_column in {"timestamp", "bar_open_time"}:
        signals["source_bar_open_time"] = signals["decision_time"]
        signals["decision_time"] = signals["decision_time"] + pd.Timedelta(hours=1)
    return bars, signals, stats


def attach_phase2_market_columns(
    predictions: pd.DataFrame,
    market_context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach causal OHLC/ATR columns to prediction-only artifacts."""

    if predictions.empty or "timestamp" not in predictions.columns:
        return predictions.copy()
    market_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "atr_14",
            "volume",
            "quote_volume",
        )
        if column in market_context.columns
    ]
    if not market_columns:
        return predictions.copy()
    market = (
        market_context[["timestamp", *market_columns]]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .copy()
    )
    output = predictions.drop(
        columns=[
            column
            for column in market_columns
            if column in predictions.columns
        ]
    )
    return output.merge(market, on="timestamp", how="left", validate="many_to_one")


def build_phase2_sandbox_inputs(
    *,
    report_dir: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    scope_dir: str | Path | None = None,
    split: str = "test",
    threshold: float | None = None,
) -> Phase2InputBuildResult:
    """Generate ``phase2_bars.csv`` and ``phase2_signals.csv`` from frozen artifacts."""

    report_path = Path(report_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest, resolved_manifest_path = load_frozen_candidate_manifest(
        report_path,
        manifest_path=manifest_path,
    )
    if not bool(manifest.get("available", False)):
        reasons = manifest.get("unavailable_reasons", []) or ["candidate_unavailable"]
        raise ValueError(f"Frozen candidate is not available: {reasons}")

    candidate_id = str(manifest.get("candidate_id", "") or "").strip()
    if not candidate_id:
        raise ValueError("Frozen candidate manifest is missing candidate_id.")
    resolved_threshold = float(
        threshold
        if threshold is not None
        else (manifest.get("threshold", {}) or {}).get("value", 0.5)
    )
    scope_path = _component_scope_dir(
        manifest,
        checkpoint_dir=checkpoint_dir,
        scope_dir=scope_dir,
    )
    predictions_path = _prediction_path(scope_path)
    predictions = _read_table(predictions_path)
    bars, signals, stats = phase2_inputs_from_predictions(
        predictions,
        candidate_id=candidate_id,
        threshold=resolved_threshold,
        split=split,
    )
    if bars.empty or signals.empty:
        raise ValueError(
            "Phase 2 adapter produced empty inputs. Check the split filter and "
            f"prediction path: {predictions_path}"
        )

    bars_path = output_path / "phase2_bars.csv"
    signals_path = output_path / "phase2_signals.csv"
    input_manifest_path = output_path / "phase2_input_manifest.json"
    bars.to_csv(bars_path, index=False)
    signals.to_csv(signals_path, index=False)

    result = Phase2InputBuildResult(
        bars_path=bars_path,
        signals_path=signals_path,
        input_manifest_path=input_manifest_path,
        candidate_manifest_path=resolved_manifest_path,
        prediction_path=predictions_path,
        candidate_id=candidate_id,
        threshold=resolved_threshold,
        rows_read=stats["rows_read"],
        rows_after_split_filter=stats["rows_after_split_filter"],
        rows_used=stats["rows_used"],
        duplicate_prediction_rows_collapsed=stats[
            "duplicate_prediction_rows_collapsed"
        ],
        bar_count=int(len(bars)),
        signal_count=int(len(signals)),
        split_filter=split,
    )
    payload = {
        "adapter_version": PHASE2_INPUT_ADAPTER_VERSION,
        "result": result.as_dict(),
        "candidate_manifest": {
            "candidate_id": candidate_id,
            "candidate_type": manifest.get("candidate_type"),
            "source_run_id": manifest.get("source_run_id"),
            "manifest_hash": manifest.get("manifest_hash"),
            "expected_manifest_hash": manifest.get("expected_manifest_hash"),
            "threshold": manifest.get("threshold"),
            "components": manifest.get("components"),
        },
        "sandbox_warning": (
            "These inputs are for Phase 2 sandbox engineering only. They do not "
            "promote the model or bypass Future-OOS gates."
        ),
    }
    _write_json(input_manifest_path, payload)
    return result
