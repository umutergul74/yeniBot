from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import CostScenario
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.engine import Phase2BacktestResult
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.execution_cache import assert_cache_matches_reference
from yenibot.phase2.execution_cache import build_fold_execution_cache
from yenibot.phase2.readiness import Phase2Gate


ECONOMIC_ATTRIBUTION_VERSION = "phase2_economic_attribution_v1"


@dataclass(frozen=True)
class EconomicAttributionSpec:
    """Fail-closed contract for measuring whether model ranking adds value.

    The null destroys the relationship between score and timestamp while
    preserving each fold-month score distribution and therefore the number of
    scores above every fixed threshold inside that group.
    """

    permutations: int = 500
    seed: int = 20260830
    required_split: str = "test"
    minimum_input_rows: int = 100
    minimum_completed_trades: int = 100
    minimum_positive_fold_share: float = 2.0 / 3.0
    permutation_group_columns: tuple[str, ...] = ("fold", "calendar_month")
    primary_metric: str = "base_cost_compounded_return"
    evidence_scope: str = "already_seen_phase2_test_window_retrospective"

    def validate(self) -> None:
        if self.permutations < 20:
            raise ValueError("At least 20 permutations are required for attribution")
        if self.minimum_input_rows < 1 or self.minimum_completed_trades < 1:
            raise ValueError("Minimum row/trade counts must be positive")
        if not 0 < self.minimum_positive_fold_share <= 1:
            raise ValueError("minimum_positive_fold_share must be in (0, 1]")
        if not self.required_split.strip():
            raise ValueError("required_split must be explicit")


@dataclass(frozen=True)
class EconomicAttributionResult:
    report: dict[str, Any]
    null_trials: pd.DataFrame
    score_bands: pd.DataFrame
    fold_outcomes: pd.DataFrame


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = selected[column].astype(str)
    values = pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _validate_inputs(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    contract: Phase2StrategyContract,
    spec: EconomicAttributionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec.validate()
    contract.validate()
    bar_required = {
        contract.bar_time_column,
        "open",
        "high",
        "low",
        "close",
        contract.atr_column,
    }
    signal_required = {
        contract.decision_time_column,
        contract.score_column,
        "fold",
        "split",
    }
    missing_bars = sorted(bar_required.difference(bars.columns))
    missing_signals = sorted(signal_required.difference(signals.columns))
    if missing_bars or missing_signals:
        raise ValueError(
            "Economic attribution input contract failed: "
            f"missing_bars={missing_bars}, missing_signals={missing_signals}"
        )

    bar_frame = bars.copy()
    signal_frame = signals.copy()
    if signal_frame[["fold", "split"]].isna().any().any():
        raise ValueError("Attribution fold and split identity must not be missing")
    bar_frame[contract.bar_time_column] = pd.to_datetime(
        bar_frame[contract.bar_time_column], utc=True, errors="coerce"
    )
    signal_frame[contract.decision_time_column] = pd.to_datetime(
        signal_frame[contract.decision_time_column], utc=True, errors="coerce"
    )
    score = pd.to_numeric(signal_frame[contract.score_column], errors="coerce")
    invalid_time_count = int(
        bar_frame[contract.bar_time_column].isna().sum()
        + signal_frame[contract.decision_time_column].isna().sum()
    )
    invalid_score_count = int((~np.isfinite(score)).sum())
    out_of_range_score_count = int(((score < 0) | (score > 1)).sum())
    if invalid_time_count or invalid_score_count or out_of_range_score_count:
        raise ValueError(
            "Economic attribution requires finite UTC times and scores in [0, 1]: "
            f"invalid_times={invalid_time_count}, invalid_scores={invalid_score_count}, "
            f"out_of_range_scores={out_of_range_score_count}"
        )
    signal_frame[contract.score_column] = score.astype(float)

    split_values = sorted(
        signal_frame["split"].dropna().astype(str).str.lower().unique().tolist()
    )
    if split_values != [spec.required_split.lower()]:
        raise ValueError(
            "Attribution accepts one explicit untouched split only: "
            f"expected={spec.required_split!r}, found={split_values}"
        )
    if len(signal_frame) < spec.minimum_input_rows:
        raise ValueError(
            f"Attribution needs at least {spec.minimum_input_rows} signals; "
            f"found {len(signal_frame)}"
        )
    duplicate_bars = int(
        bar_frame[contract.bar_time_column].duplicated(keep=False).sum()
    )
    duplicate_signals = int(
        signal_frame[contract.decision_time_column].duplicated(keep=False).sum()
    )
    if duplicate_bars or duplicate_signals:
        raise ValueError(
            "Attribution refuses ambiguous timestamps: "
            f"duplicate_bars={duplicate_bars}, duplicate_signals={duplicate_signals}"
        )
    if "candidate_id" in signal_frame.columns:
        candidate_ids = signal_frame["candidate_id"].dropna().astype(str).unique()
        if (
            signal_frame["candidate_id"].isna().any()
            or len(candidate_ids) != 1
            or candidate_ids[0] != contract.candidate_id
        ):
            raise ValueError(
                "Candidate identity mismatch between signals and strategy contract"
            )
    if "threshold" in signal_frame.columns:
        thresholds = pd.to_numeric(signal_frame["threshold"], errors="coerce")
        if thresholds.isna().any() or not np.allclose(
            thresholds.to_numpy(), contract.threshold, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "Frozen threshold mismatch between signals and strategy contract"
            )

    bar_frame = bar_frame.sort_values(contract.bar_time_column).reset_index(drop=True)
    signal_frame = signal_frame.sort_values(contract.decision_time_column).reset_index(
        drop=True
    )
    signal_frame["calendar_month"] = signal_frame[
        contract.decision_time_column
    ].dt.strftime("%Y-%m")
    group_columns = list(spec.permutation_group_columns)
    missing_groups = sorted(set(group_columns).difference(signal_frame.columns))
    if missing_groups:
        raise ValueError(f"Missing permutation group columns: {missing_groups}")
    group_sizes = signal_frame.groupby(group_columns, dropna=False).size()
    immutable_rows = int(group_sizes.loc[group_sizes < 2].sum())
    if immutable_rows:
        raise ValueError(
            "Every permutation group needs at least two rows; "
            f"immutable_rows={immutable_rows}"
        )
    numeric_bars = ["open", "high", "low", "close", contract.atr_column]
    for column in numeric_bars:
        bar_frame[column] = pd.to_numeric(bar_frame[column], errors="coerce")
    if not np.isfinite(bar_frame[numeric_bars].to_numpy(dtype=float)).all():
        raise ValueError("Bars contain non-finite OHLC/ATR values")
    if (bar_frame[contract.atr_column] <= 0).any():
        raise ValueError("Bars contain non-positive ATR values")
    invalid_ohlc = (
        bar_frame["high"] < bar_frame[["open", "close", "low"]].max(axis=1)
    ) | (bar_frame["low"] > bar_frame[["open", "close", "high"]].min(axis=1))
    if invalid_ohlc.any():
        raise ValueError(f"Bars violate OHLC bounds in {int(invalid_ohlc.sum())} rows")

    gap_hours = bar_frame[contract.bar_time_column].diff().dt.total_seconds() / 3600.0
    within_fold_gap_count = 0
    previous_fold_end = None
    for _, fold_signals in signal_frame.groupby("fold", sort=True, dropna=False):
        fold_start = fold_signals[contract.decision_time_column].min()
        fold_end = fold_signals[contract.decision_time_column].max()
        if previous_fold_end is not None and fold_start <= previous_fold_end:
            raise ValueError(
                "Attribution fold windows must be disjoint and chronological"
            )
        previous_fold_end = fold_end
        fold_bar_times = bar_frame.loc[
            (bar_frame[contract.bar_time_column] >= fold_start)
            & (bar_frame[contract.bar_time_column] <= fold_end),
            contract.bar_time_column,
        ]
        fold_gaps = fold_bar_times.diff().dt.total_seconds() / 3600.0
        within_fold_gap_count += int((fold_gaps > contract.max_bar_gap_hours).sum())
    total_gap_count = int((gap_hours > contract.max_bar_gap_hours).sum())
    diagnostics = {
        "bar_count": int(len(bar_frame)),
        "signal_count": int(len(signal_frame)),
        "time_start": signal_frame[contract.decision_time_column].min().isoformat(),
        "time_end": signal_frame[contract.decision_time_column].max().isoformat(),
        "fold_count": int(signal_frame["fold"].nunique()),
        "permutation_group_count": int(len(group_sizes)),
        "selected_signal_count": int(
            (signal_frame[contract.score_column] >= contract.threshold).sum()
        ),
        "total_bar_gap_count": total_gap_count,
        "within_fold_bar_gap_count": within_fold_gap_count,
        "inter_fold_gap_count": max(total_gap_count - within_fold_gap_count, 0),
        "fold_segmented_evaluation": True,
        "max_bar_gap_hours": (
            float(gap_hours.max()) if not gap_hours.dropna().empty else None
        ),
        "bars_hash": _frame_hash(
            bar_frame,
            [contract.bar_time_column, *numeric_bars],
        ),
        "signals_hash": _frame_hash(
            signal_frame,
            [
                contract.decision_time_column,
                contract.score_column,
                "fold",
                "split",
            ],
        ),
    }
    return bar_frame, signal_frame, diagnostics


def _summary_view(result: Phase2BacktestResult) -> dict[str, Any]:
    keys = (
        "cost_scenario",
        "trade_count",
        "selected_signal_count",
        "entry_filter_passed_count",
        "compounded_return",
        "completed_trade_compounded_return",
        "mean_net_return",
        "hit_rate",
        "profit_factor",
        "max_drawdown",
        "data_contract_complete",
        "data_gap_forced_close_count",
        "censored_position_count",
        "fold_count",
        "return_basis",
        "drawdown_basis",
    )
    return {key: _json_ready(result.summary.get(key)) for key in keys}


def _run_fold_segmented_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    gate: Phase2Gate,
    contract: Phase2StrategyContract,
    cost_scenario: CostScenario,
) -> Phase2BacktestResult:
    """Evaluate disjoint walk-forward folds without trading across fold gaps.

    Each fold starts with fresh unit equity. Only completed trades contribute to
    the aggregate return; an end-of-fold open position remains censored. This
    avoids both carrying exposure through the embargo gap and silently treating
    the gap as a flat market.
    """

    fold_results: list[tuple[Any, Phase2BacktestResult]] = []
    bar_time = contract.bar_time_column
    decision_time = contract.decision_time_column
    for fold, fold_signals in signals.groupby("fold", sort=True, dropna=False):
        start = fold_signals[decision_time].min()
        end = fold_signals[decision_time].max()
        fold_bars = bars.loc[(bars[bar_time] >= start) & (bars[bar_time] <= end)].copy()
        if fold_bars.empty:
            raise ValueError(f"No bars cover attribution fold {fold!r}")
        result = run_long_only_backtest(
            fold_bars,
            fold_signals,
            gate=gate,
            contract=contract,
            cost_scenario=cost_scenario,
            mode="sandbox",
        )
        fold_results.append((fold, result))

    trade_frames: list[pd.DataFrame] = []
    for fold, result in fold_results:
        frame = result.trades.copy()
        if not frame.empty:
            frame["evaluation_fold"] = fold
            trade_frames.append(frame)
    trades = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )
    completed = (
        trades.loc[trades["trade_status"].eq("completed")].copy()
        if not trades.empty
        else trades
    )
    returns = (
        pd.to_numeric(completed["net_return"], errors="coerce").dropna()
        if not completed.empty
        else pd.Series(dtype=float)
    )
    if returns.empty:
        equity = pd.DataFrame(columns=["timestamp", "equity", "net_return", "drawdown"])
        compounded_return = 0.0
        max_drawdown = 0.0
        profit_factor = 0.0
        hit_rate = 0.0
        mean_return = 0.0
    else:
        compounded = (1.0 + returns).cumprod()
        drawdown = compounded / compounded.cummax().clip(lower=1.0) - 1.0
        equity = pd.DataFrame(
            {
                "timestamp": completed.loc[returns.index, "exit_time"].to_numpy(),
                "equity": compounded.to_numpy(),
                "net_return": returns.to_numpy(),
                "drawdown": drawdown.to_numpy(),
            }
        )
        compounded_return = float(compounded.iloc[-1] - 1.0)
        max_drawdown = float(drawdown.min())
        wins = float(returns.loc[returns > 0].sum())
        losses = float(-returns.loc[returns < 0].sum())
        profit_factor = float(wins / losses) if losses > 0 else None
        hit_rate = float((returns > 0).mean())
        mean_return = float(returns.mean())

    fold_summaries = {str(fold): _summary_view(result) for fold, result in fold_results}
    summary = {
        "mode": "sandbox",
        "evidence_status": gate.evidence_status,
        "cost_scenario": cost_scenario.name,
        "trade_count": int(len(returns)),
        "selected_signal_count": int(
            sum(
                result.summary.get("selected_signal_count", 0)
                for _, result in fold_results
            )
        ),
        "entry_filter_passed_count": int(
            sum(
                result.summary.get("entry_filter_passed_count", 0)
                for _, result in fold_results
            )
        ),
        "compounded_return": compounded_return,
        "completed_trade_compounded_return": compounded_return,
        "mean_net_return": mean_return,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "data_contract_complete": all(
            bool(result.summary.get("data_contract_complete"))
            for _, result in fold_results
        ),
        "data_gap_forced_close_count": int(
            sum(
                result.summary.get("data_gap_forced_close_count", 0)
                for _, result in fold_results
            )
        ),
        "censored_position_count": int(
            sum(
                result.summary.get("censored_position_count", 0)
                for _, result in fold_results
            )
        ),
        "fold_count": len(fold_results),
        "fold_summaries": fold_summaries,
        "return_basis": "completed_trades_compounded_across_independent_walk_forward_folds",
        "drawdown_basis": "completed_trade_close_equity_across_independent_walk_forward_folds",
    }
    return Phase2BacktestResult(
        trades=trades,
        equity=equity,
        summary=summary,
        metadata={
            "fold_segmentation": "independent_walk_forward_folds",
            "positions_carried_across_fold_gaps": False,
            "end_of_fold_positions": "censored_excluded_from_aggregate_return",
            "fold_summaries": fold_summaries,
        },
    )


def _run_controls(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    gate: Phase2Gate,
    contract: Phase2StrategyContract,
    scenarios: tuple[CostScenario, ...],
    spec: EconomicAttributionSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(spec.seed)
    groups = [
        np.asarray(index, dtype=int)
        for index in signals.groupby(
            list(spec.permutation_group_columns),
            sort=True,
            dropna=False,
        ).indices.values()
    ]
    scores = signals[contract.score_column].to_numpy(dtype=float)
    null_rows: list[dict[str, Any]] = []
    base_scenario = next(
        (item for item in scenarios if item.name == "base"), scenarios[0]
    )
    cache = build_fold_execution_cache(
        bars, signals, contract=contract, scenario=base_scenario
    )
    reference = _run_fold_segmented_backtest(
        bars, signals, gate=gate, contract=contract, cost_scenario=base_scenario
    )
    assert_cache_matches_reference(cache.evaluate(scores), reference.summary)
    for trial in range(spec.permutations):
        shuffled = scores.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(shuffled[indices])
        summary = cache.evaluate(shuffled)
        if trial == 0:
            null_signals = signals.copy()
            null_signals[contract.score_column] = shuffled
            reference = _run_fold_segmented_backtest(
                bars,
                null_signals,
                gate=gate,
                contract=contract,
                cost_scenario=base_scenario,
            )
            assert_cache_matches_reference(summary, reference.summary)
        null_rows.append(
            {
                "trial": trial,
                "seed": spec.seed,
                "cost_scenario": base_scenario.name,
                "compounded_return": summary.get("compounded_return"),
                "completed_trade_compounded_return": summary.get(
                    "completed_trade_compounded_return"
                ),
                "max_drawdown": summary.get("max_drawdown"),
                "trade_count": summary.get("trade_count"),
                "selected_signal_count": summary.get("selected_signal_count"),
                "profit_factor": summary.get("profit_factor"),
                "hit_rate": summary.get("hit_rate"),
                "data_contract_complete": summary.get("data_contract_complete"),
            }
        )
    null_frame = pd.DataFrame(null_rows)

    deterministic_controls: dict[str, Any] = {}
    always_on = signals.copy()
    always_on[contract.score_column] = 1.0
    inverted = signals.copy()
    inverted_scores = scores.copy()
    for indices in groups:
        order = np.argsort(scores[indices], kind="stable")
        reversed_values = np.sort(scores[indices])[::-1]
        replacement = np.empty(len(indices), dtype=float)
        replacement[order] = reversed_values
        inverted_scores[indices] = replacement
    inverted[contract.score_column] = inverted_scores
    for control_name, control_signals in (
        ("always_on_long_context", always_on),
        ("inverted_model_ranking", inverted),
    ):
        deterministic_controls[control_name] = {}
        for scenario in scenarios:
            result = _run_fold_segmented_backtest(
                bars,
                control_signals,
                gate=gate,
                contract=contract,
                cost_scenario=scenario,
            )
            deterministic_controls[control_name][scenario.name] = _summary_view(result)
            if scenario.name == base_scenario.name:
                assert_cache_matches_reference(
                    cache.evaluate(control_signals[contract.score_column].to_numpy()),
                    result.summary,
                )
    return deterministic_controls, null_frame


def _score_band_summary(
    signals: pd.DataFrame,
    *,
    contract: Phase2StrategyContract,
) -> pd.DataFrame:
    frame = signals.copy()
    score = frame[contract.score_column]
    frame["score_decile"] = (
        frame.groupby("fold", dropna=False)[contract.score_column]
        .rank(method="first", pct=True)
        .mul(10)
        .apply(np.ceil)
        .clip(1, 10)
        .astype(int)
    )
    aggregations: dict[str, tuple[str, str]] = {
        "rows": (contract.score_column, "size"),
        "mean_score": (contract.score_column, "mean"),
    }
    for column in ("forward_return", "tb_return", "label"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            aggregations[f"mean_{column}"] = (column, "mean")
    summary = frame.groupby("score_decile", as_index=False).agg(**aggregations)
    summary["selected_share"] = (
        frame.assign(selected=score >= contract.threshold)
        .groupby("score_decile")["selected"]
        .mean()
        .reindex(summary["score_decile"])
        .to_numpy()
    )
    return summary


def _fold_outcomes(
    result: Phase2BacktestResult,
    signals: pd.DataFrame,
    *,
    contract: Phase2StrategyContract,
) -> pd.DataFrame:
    trades = result.trades.copy()
    all_folds = sorted(signals["fold"].unique())
    if trades.empty:
        return pd.DataFrame(
            {
                "fold": all_folds,
                "trade_count": 0,
                "compounded_net_return": 0.0,
                "mean_net_return": 0.0,
            }
        )
    trades = trades.loc[trades["trade_status"].eq("completed")].copy()
    if "evaluation_fold" in trades.columns:
        trades["fold"] = trades["evaluation_fold"]
    else:
        fold_map = signals[[contract.decision_time_column, "fold"]].drop_duplicates(
            contract.decision_time_column, keep="last"
        )
        trades = trades.merge(
            fold_map,
            left_on="decision_time",
            right_on=contract.decision_time_column,
            how="left",
            validate="many_to_one",
        )
    rows = []
    for fold in all_folds:
        group = trades.loc[trades["fold"].eq(fold)]
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        rows.append(
            {
                "fold": fold,
                "trade_count": int(len(returns)),
                "compounded_net_return": float(np.prod(1.0 + returns) - 1.0),
                "mean_net_return": float(returns.mean()) if len(returns) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_economic_attribution(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    gate: Phase2Gate,
    contract: Phase2StrategyContract,
    spec: EconomicAttributionSpec = EconomicAttributionSpec(),
    scenarios: tuple[CostScenario, ...] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> EconomicAttributionResult:
    """Measure model-ranking contribution without tuning the model or policy."""

    resolved_scenarios = scenarios or (
        next(item for item in contract.cost_scenarios if item.name == "base"),
        next(item for item in contract.cost_scenarios if item.name == "adverse"),
    )
    if not any(item.name == "base" for item in resolved_scenarios):
        raise ValueError("Attribution requires an explicit base cost scenario")
    if not any(item.name == "adverse" for item in resolved_scenarios):
        raise ValueError("Attribution requires an explicit adverse cost scenario")
    bar_frame, signal_frame, input_diagnostics = _validate_inputs(
        bars,
        signals,
        contract=contract,
        spec=spec,
    )

    actual_results: dict[str, Phase2BacktestResult] = {}
    actual_summaries: dict[str, Any] = {}
    for scenario in resolved_scenarios:
        actual = _run_fold_segmented_backtest(
            bar_frame,
            signal_frame,
            gate=gate,
            contract=contract,
            cost_scenario=scenario,
        )
        actual_results[scenario.name] = actual
        actual_summaries[scenario.name] = _summary_view(actual)

    controls, null_trials = _run_controls(
        bar_frame,
        signal_frame,
        gate=gate,
        contract=contract,
        scenarios=resolved_scenarios,
        spec=spec,
    )
    base_actual = actual_summaries["base"]
    base_return = float(base_actual["compounded_return"] or 0.0)
    null_returns = pd.to_numeric(
        null_trials["compounded_return"], errors="coerce"
    ).dropna()
    permutation_p = float(
        (1 + int((null_returns >= base_return).sum())) / (len(null_returns) + 1)
    )
    null_summary = {
        "method": "within_fold_calendar_month_score_permutation",
        "preserves": [
            "score_distribution_per_fold_month",
            "above_threshold_count_per_fold_month",
            "market_bars",
            "execution_contract",
            "base_cost_contract",
        ],
        "trials": int(len(null_trials)),
        "seed": spec.seed,
        "return_mean": float(null_returns.mean()),
        "return_median": float(null_returns.median()),
        "return_ci_95": [
            float(null_returns.quantile(0.025)),
            float(null_returns.quantile(0.975)),
        ],
        "one_sided_p_value_actual_not_better_than_null": permutation_p,
        "actual_minus_null_median": base_return - float(null_returns.median()),
    }

    score_bands = _score_band_summary(signal_frame, contract=contract)
    fold_outcomes = _fold_outcomes(
        actual_results["base"],
        signal_frame,
        contract=contract,
    )
    positive_fold_share = (
        float((fold_outcomes["compounded_net_return"] > 0).mean())
        if not fold_outcomes.empty
        else 0.0
    )
    score_return_rankic = None
    if "forward_return" in signal_frame.columns:
        forward = pd.to_numeric(signal_frame["forward_return"], errors="coerce")
        valid = forward.notna()
        if (
            int(valid.sum()) >= 3
            and signal_frame.loc[valid, contract.score_column].nunique() > 1
            and forward.loc[valid].nunique() > 1
        ):
            score_return_rankic = float(
                signal_frame.loc[valid, contract.score_column].corr(
                    forward.loc[valid], method="spearman"
                )
            )
    top_bottom_return_spread = None
    if "mean_forward_return" in score_bands.columns:
        bottom = score_bands.loc[score_bands.score_decile == 1, "mean_forward_return"]
        top = score_bands.loc[score_bands.score_decile == 10, "mean_forward_return"]
        if not bottom.empty and not top.empty:
            top_bottom_return_spread = float(top.iloc[0] - bottom.iloc[0])

    criteria = {
        "base_cost_return_positive": base_return > 0,
        "adverse_cost_return_positive": float(
            actual_summaries["adverse"]["compounded_return"] or 0.0
        )
        > 0,
        "beats_rank_destroyed_null_at_5pct": permutation_p <= 0.05,
        "completed_trade_count_sufficient": int(base_actual["trade_count"] or 0)
        >= spec.minimum_completed_trades,
        "positive_fold_share_sufficient": positive_fold_share
        >= spec.minimum_positive_fold_share,
        "score_forward_return_rankic_positive": score_return_rankic is not None
        and score_return_rankic > 0,
        "top_minus_bottom_forward_return_positive": top_bottom_return_spread is not None
        and top_bottom_return_spread > 0,
        "execution_data_contract_complete": bool(base_actual["data_contract_complete"]),
    }
    diagnostic_gate_passed = all(criteria.values())
    contract_payload = {
        "attribution_version": ECONOMIC_ATTRIBUTION_VERSION,
        "fold_segmentation": "independent_walk_forward_folds",
        "strategy": asdict(contract),
        "spec": asdict(spec),
        "scenarios": [asdict(item) for item in resolved_scenarios],
    }
    report = {
        "version": ECONOMIC_ATTRIBUTION_VERSION,
        "status": (
            "retrospective_diagnostic_gate_passed_requires_clean_confirmation"
            if diagnostic_gate_passed
            else "retrospective_diagnostic_gate_failed"
        ),
        "evidence_scope": spec.evidence_scope,
        "model_or_strategy_refit_performed": False,
        "automatic_strategy_selection_allowed": False,
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "independent_confirmation_required": True,
        "strategy_id": contract.strategy_id,
        "candidate_id": contract.candidate_id,
        "contract_hash": _stable_hash(contract_payload),
        "source_metadata": source_metadata or {},
        "input_diagnostics": input_diagnostics,
        "actual": actual_summaries,
        "deterministic_controls": controls,
        "rank_destroyed_null": null_summary,
        "score_diagnostics": {
            "score_forward_return_rankic": score_return_rankic,
            "top_minus_bottom_forward_return_spread": top_bottom_return_spread,
        },
        "fold_diagnostics": {
            "positive_fold_share": positive_fold_share,
            "fold_count": int(len(fold_outcomes)),
        },
        "assessment": {
            "diagnostic_gate_passed": diagnostic_gate_passed,
            "criteria": criteria,
            "failed_criteria": [
                name for name, passed in criteria.items() if not passed
            ],
            "interpretation": (
                "Passing would only justify one pre-registered clean confirmation; "
                "this seen window can never authorize promotion."
            ),
        },
    }
    return EconomicAttributionResult(
        report=_json_ready(report),
        null_trials=null_trials,
        score_bands=score_bands,
        fold_outcomes=fold_outcomes,
    )


def write_economic_attribution(
    output_dir: str | Path,
    result: EconomicAttributionResult,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "phase2_economic_attribution.json").write_text(
        json.dumps(result.report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result.null_trials.to_csv(
        path / "phase2_rank_destroyed_null_trials.csv", index=False
    )
    result.score_bands.to_csv(path / "phase2_score_decile_attribution.csv", index=False)
    result.fold_outcomes.to_csv(path / "phase2_fold_economic_outcomes.csv", index=False)
    report = result.report
    actual = report["actual"]
    null = report["rank_destroyed_null"]
    assessment = report["assessment"]
    lines = [
        "# Phase 2 Economic Attribution",
        "",
        "> Retrospective diagnostic only. This already-seen window cannot promote a model or strategy.",
        "",
        f"- Strategy: `{report['strategy_id']}`",
        f"- Candidate: `{report['candidate_id']}`",
        f"- Base-cost return: `{actual['base']['compounded_return']}`",
        f"- Adverse-cost return: `{actual['adverse']['compounded_return']}`",
        f"- Completed trades: `{actual['base']['trade_count']}`",
        f"- Rank-destroyed null median: `{null['return_median']}`",
        f"- Actual minus null median: `{null['actual_minus_null_median']}`",
        f"- One-sided permutation p-value: `{null['one_sided_p_value_actual_not_better_than_null']}`",
        f"- Score/forward-return RankIC: `{report['score_diagnostics']['score_forward_return_rankic']}`",
        f"- Positive fold share: `{report['fold_diagnostics']['positive_fold_share']}`",
        f"- Diagnostic gate passed: `{assessment['diagnostic_gate_passed']}`",
        f"- Failed criteria: `{assessment['failed_criteria']}`",
        "- Promotion allowed: `False`",
        "- Live trading allowed: `False`",
        "",
    ]
    (path / "phase2_economic_attribution.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
