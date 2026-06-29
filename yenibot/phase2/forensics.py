from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _compounded(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float((1.0 + values.astype(float)).prod() - 1.0)


def _profit_factor(values: pd.Series) -> float | None:
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    return wins / losses if losses > 0 else None


def _group_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    columns = [
        group_column,
        "trade_count",
        "hit_rate",
        "mean_gross_return",
        "mean_net_return",
        "compounded_gross_return",
        "compounded_net_return",
        "profit_factor",
        "mean_score",
        "mean_holding_hours",
        "mean_mfe",
        "mean_mae",
    ]
    if frame.empty or group_column not in frame.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_column, observed=True, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
        rows.append(
            {
                group_column: str(key),
                "trade_count": int(len(group)),
                "hit_rate": float((net > 0).mean()) if not net.empty else 0.0,
                "mean_gross_return": float(gross.mean()) if not gross.empty else 0.0,
                "mean_net_return": float(net.mean()) if not net.empty else 0.0,
                "compounded_gross_return": _compounded(gross),
                "compounded_net_return": _compounded(net),
                "profit_factor": _profit_factor(net),
                "mean_score": float(group["score"].mean()),
                "mean_holding_hours": float(group["holding_hours"].mean()),
                "mean_mfe": float(group["max_favorable_excursion"].mean()),
                "mean_mae": float(group["max_adverse_excursion"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _moving_block_bootstrap(
    values: pd.Series,
    *,
    samples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    returns = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = int(len(returns))
    if count == 0:
        return {
            "method": "circular_moving_block_bootstrap",
            "trade_count": 0,
            "samples": 0,
            "block_length": 0,
            "mean_return_ci_95": [None, None],
            "compounded_return_ci_95": [None, None],
            "probability_mean_return_positive": None,
            "probability_compounded_return_positive": None,
        }
    block_length = max(2, min(count, int(round(math.sqrt(count)))))
    block_count = int(math.ceil(count / block_length))
    rng = np.random.default_rng(seed)
    mean_samples = np.empty(samples, dtype=float)
    compounded_samples = np.empty(samples, dtype=float)
    offsets = np.arange(block_length)
    for sample_index in range(samples):
        starts = rng.integers(0, count, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % count).reshape(-1)[:count]
        sample = returns[indices]
        mean_samples[sample_index] = float(sample.mean())
        compounded_samples[sample_index] = float(np.prod(1.0 + sample) - 1.0)
    return {
        "method": "circular_moving_block_bootstrap",
        "trade_count": count,
        "samples": int(samples),
        "block_length": block_length,
        "seed": seed,
        "mean_return_ci_95": [
            float(np.quantile(mean_samples, 0.025)),
            float(np.quantile(mean_samples, 0.975)),
        ],
        "compounded_return_ci_95": [
            float(np.quantile(compounded_samples, 0.025)),
            float(np.quantile(compounded_samples, 0.975)),
        ],
        "probability_mean_return_positive": float((mean_samples > 0).mean()),
        "probability_compounded_return_positive": float(
            (compounded_samples > 0).mean()
        ),
    }


def phase2_trade_forensics(
    trades: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    execution_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build read-only Phase 2 diagnostics without changing strategy policy."""

    frame = trades.copy()
    signal_frame = signals.copy()
    execution = execution_summary or {}
    numeric_columns = (
        "net_return",
        "gross_return",
        "total_cost_return",
        "score",
        "holding_hours",
        "holding_bars",
        "max_favorable_excursion",
        "max_adverse_excursion",
    )
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=float)
    for column in ("exit_reason", "strategy_id"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=str)
    if "dynamic_stop_activated" not in frame.columns:
        frame["dynamic_stop_activated"] = pd.Series(dtype=bool)
    for column in ("decision_time", "entry_time", "exit_time"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    if "decision_time" not in signal_frame.columns:
        signal_frame["decision_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    for column in ("decision_time", "entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if "decision_time" in signal_frame.columns:
        signal_frame["decision_time"] = pd.to_datetime(
            signal_frame["decision_time"],
            utc=True,
            errors="coerce",
        )
    if "fold" in signal_frame.columns and "decision_time" in frame.columns:
        fold_map = signal_frame[["decision_time", "fold"]].drop_duplicates(
            "decision_time",
            keep="last",
        )
        frame = frame.merge(fold_map, on="decision_time", how="left")

    frame["month"] = frame["entry_time"].dt.strftime("%Y-%m")
    if len(frame) >= 4:
        ranked_scores = frame["score"].rank(method="first")
        frame["score_band"] = pd.qcut(
            ranked_scores,
            q=4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        )
    else:
        frame["score_band"] = "insufficient_rows"
    frame["holding_bucket"] = pd.cut(
        frame["holding_bars"],
        bins=[0, 3, 6, 9, float("inf")],
        labels=["1_3", "4_6", "7_9", "10_plus"],
        include_lowest=True,
    )

    group_frames = {
        "exit_reason": _group_summary(frame, "exit_reason"),
        "fold": _group_summary(frame, "fold"),
        "month": _group_summary(frame, "month"),
        "score_band": _group_summary(frame, "score_band"),
        "holding": _group_summary(frame, "holding_bucket"),
    }

    selected = int(execution.get("selected_signal_count", 0) or 0)
    entry_filter_skipped = int(execution.get("entry_filter_skipped_count", 0) or 0)
    entry_filter_passed = int(execution.get("entry_filter_passed_count", 0) or 0)
    executed = int(len(frame))
    funnel = pd.DataFrame(
        [
            {"stage": "input_signals", "count": int(len(signal_frame))},
            {"stage": "selected_above_threshold", "count": selected},
            {"stage": "skipped_entry_filter", "count": entry_filter_skipped},
            {"stage": "entry_filter_passed", "count": entry_filter_passed},
            {
                "stage": "skipped_open_position",
                "count": int(
                    execution.get("skipped_during_open_position_count", 0) or 0
                ),
            },
            {
                "stage": "skipped_stale_entry",
                "count": int(execution.get("skipped_stale_entry_count", 0) or 0),
            },
            {
                "stage": "skipped_invalid_atr",
                "count": int(execution.get("skipped_invalid_atr_count", 0) or 0),
            },
            {"stage": "executed_trades", "count": executed},
        ]
    )

    net = pd.to_numeric(frame["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(frame["gross_return"], errors="coerce").dropna()
    costs = pd.to_numeric(frame["total_cost_return"], errors="coerce").dropna()
    without_best_trade = frame.drop(index=frame["net_return"].idxmax()) if len(frame) else frame
    without_best_five = (
        frame.drop(index=frame["net_return"].nlargest(min(5, len(frame))).index)
        if len(frame)
        else frame
    )
    month_returns = group_frames["month"]
    best_month = None
    without_best_month = frame
    if not month_returns.empty:
        best_month = str(
            month_returns.sort_values(
                "compounded_net_return",
                ascending=False,
            ).iloc[0]["month"]
        )
        without_best_month = frame.loc[frame["month"] != best_month]

    summary = {
        "forensics_version": "phase2_forensics_v1",
        "strategy_id": (
            str(frame["strategy_id"].iloc[0])
            if not frame.empty and "strategy_id" in frame.columns
            else ""
        ),
        "trade_count": executed,
        "selected_signal_count": selected,
        "selected_signal_execution_rate": (
            float(executed / selected) if selected > 0 else None
        ),
        "entry_filter_skipped_count": entry_filter_skipped,
        "entry_filter_passed_count": entry_filter_passed,
        "entry_filter_pass_rate": (
            float(entry_filter_passed / (entry_filter_passed + entry_filter_skipped))
            if entry_filter_passed + entry_filter_skipped > 0
            else None
        ),
        "gross_edge_bps_per_trade": (
            float(gross.mean() * 10_000.0) if not gross.empty else None
        ),
        "cost_bps_per_trade": (
            float(costs.mean() * 10_000.0) if not costs.empty else None
        ),
        "net_edge_bps_per_trade": (
            float(net.mean() * 10_000.0) if not net.empty else None
        ),
        "break_even_round_trip_cost_bps": (
            float(gross.mean() * 10_000.0) if not gross.empty else None
        ),
        "compounded_gross_return": _compounded(gross),
        "compounded_net_return": _compounded(net),
        "max_holding_exit_share": (
            float((frame["exit_reason"] == "max_holding_bars").mean())
            if not frame.empty
            else None
        ),
        "dynamic_stop_activation_share": (
            float(frame["dynamic_stop_activated"].astype(bool).mean())
            if not frame.empty and "dynamic_stop_activated" in frame.columns
            else 0.0
        ),
        "best_trade_removed_compounded_return": _compounded(
            without_best_trade["net_return"]
        ),
        "best_five_trades_removed_compounded_return": _compounded(
            without_best_five["net_return"]
        ),
        "best_month": best_month,
        "best_month_removed_compounded_return": _compounded(
            without_best_month["net_return"]
        ),
        "automatic_policy_selection_allowed": False,
        "selection_reason": "already_seen_test_window_exploratory_only",
    }
    bootstrap = _moving_block_bootstrap(net)
    return {
        "trades": frame,
        "summary": summary,
        "bootstrap": bootstrap,
        "signal_funnel": funnel,
        **group_frames,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_phase2_forensics(
    output_dir: str | Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    frame_files = {
        "exit_reason": "phase2_exit_reason_forensics.csv",
        "fold": "phase2_fold_forensics.csv",
        "month": "phase2_month_forensics.csv",
        "score_band": "phase2_score_band_forensics.csv",
        "holding": "phase2_holding_forensics.csv",
        "signal_funnel": "phase2_signal_funnel.csv",
    }
    for key, filename in frame_files.items():
        diagnostics[key].to_csv(path / filename, index=False)

    summary = _json_ready(diagnostics["summary"])
    bootstrap = _json_ready(diagnostics["bootstrap"])
    (path / "phase2_forensics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "phase2_bootstrap_summary.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown = "\n".join(
        [
            "# Phase 2 Forensics",
            "",
            "> Diagnostic only. Automatic policy selection is disabled on this seen test window.",
            "",
            f"- Strategy: `{summary.get('strategy_id')}`",
            f"- Trades: `{summary.get('trade_count')}`",
            f"- Entry-filter passed: `{summary.get('entry_filter_passed_count')}`",
            f"- Entry-filter skipped: `{summary.get('entry_filter_skipped_count')}`",
            f"- Gross edge per trade (bps): `{summary.get('gross_edge_bps_per_trade')}`",
            f"- Cost per trade (bps): `{summary.get('cost_bps_per_trade')}`",
            f"- Net edge per trade (bps): `{summary.get('net_edge_bps_per_trade')}`",
            f"- Compounded gross return: `{summary.get('compounded_gross_return')}`",
            f"- Compounded net return: `{summary.get('compounded_net_return')}`",
            f"- Max-holding exit share: `{summary.get('max_holding_exit_share')}`",
            f"- Best month removed return: `{summary.get('best_month_removed_compounded_return')}`",
            f"- Bootstrap net-return CI: `{bootstrap.get('compounded_return_ci_95')}`",
            "",
        ]
    )
    (path / "phase2_forensics_report.md").write_text(markdown, encoding="utf-8")
    return {"summary": summary, "bootstrap": bootstrap}
