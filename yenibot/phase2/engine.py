from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.contracts import CostScenario
from yenibot.phase2.contracts import Phase2Mode
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.costs import net_long_return
from yenibot.phase2.readiness import Phase2Gate


@dataclass(frozen=True)
class Phase2BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _prepare_bars(frame: pd.DataFrame, contract: Phase2StrategyContract) -> pd.DataFrame:
    required = {contract.bar_time_column, "open", "high", "low", "close", contract.atr_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Bars are missing required columns: {missing}")
    bars = frame.copy()
    bars[contract.bar_time_column] = bars[contract.bar_time_column].map(_to_timestamp)
    bars = bars.sort_values(contract.bar_time_column).drop_duplicates(contract.bar_time_column)
    return bars.reset_index(drop=True)


def _prepare_signals(frame: pd.DataFrame, contract: Phase2StrategyContract) -> pd.DataFrame:
    required = {contract.decision_time_column, contract.score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Signals are missing required columns: {missing}")
    signals = frame.copy()
    signals[contract.decision_time_column] = signals[contract.decision_time_column].map(
        _to_timestamp
    )
    signals = signals.sort_values(contract.decision_time_column)
    return signals.reset_index(drop=True)


def _entry_index_after_decision(
    bar_times: pd.Series,
    decision_time: pd.Timestamp,
) -> int | None:
    candidates = bar_times[bar_times > decision_time]
    if candidates.empty:
        return None
    return int(candidates.index[0])


def _holding_hours(
    bars: pd.DataFrame,
    contract: Phase2StrategyContract,
    entry_idx: int,
    exit_idx: int,
) -> float:
    start = bars.loc[entry_idx, contract.bar_time_column]
    end = bars.loc[exit_idx, contract.bar_time_column]
    return max((end - start).total_seconds() / 3600.0, 0.0)


def _resolve_exit(
    bars: pd.DataFrame,
    contract: Phase2StrategyContract,
    *,
    entry_idx: int,
    entry_price: float,
    atr: float,
) -> tuple[int, float, str]:
    take_price = entry_price + contract.take_profit_atr * atr
    stop_price = entry_price - contract.stop_loss_atr * atr
    last_idx = min(entry_idx + contract.max_holding_bars - 1, len(bars) - 1)

    for idx in range(entry_idx, last_idx + 1):
        if idx > entry_idx:
            previous_time = bars.loc[idx - 1, contract.bar_time_column]
            current_time = bars.loc[idx, contract.bar_time_column]
            gap_hours = (current_time - previous_time).total_seconds() / 3600.0
            if gap_hours > contract.max_bar_gap_hours:
                prior_idx = idx - 1
                return (
                    prior_idx,
                    float(bars.loc[prior_idx, "close"]),
                    "data_gap_forced_close",
                )
        high = float(bars.loc[idx, "high"])
        low = float(bars.loc[idx, "low"])
        touched_take = high >= take_price
        touched_stop = low <= stop_price
        if touched_take and touched_stop:
            if contract.same_bar_policy == "skip_ambiguous":
                return idx, float(bars.loc[idx, "close"]), "ambiguous_same_bar_skipped_to_close"
            if contract.same_bar_policy == "take_profit_first":
                return idx, take_price, "take_profit_same_bar"
            return idx, stop_price, "stop_loss_same_bar_conservative"
        if touched_stop:
            return idx, stop_price, "stop_loss"
        if touched_take:
            return idx, take_price, "take_profit"

    return last_idx, float(bars.loc[last_idx, "close"]), "max_holding_bars"


def _summarize(trades: pd.DataFrame, *, mode: Phase2Mode, gate: Phase2Gate) -> dict[str, Any]:
    if trades.empty:
        return {
            "mode": mode,
            "evidence_status": gate.evidence_status,
            "trade_count": 0,
            "mean_net_return": 0.0,
            "sum_net_return": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": None,
            "average_loss": None,
            "payoff_ratio": None,
            "final_equity": 1.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "max_single_trade_net_return": None,
            "min_single_trade_net_return": None,
        }
    wins = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()
    compounded = (1.0 + trades["net_return"]).cumprod()
    running_peak = compounded.cummax().clip(lower=1.0)
    drawdown = compounded / running_peak - 1.0
    winning_trades = trades.loc[trades["net_return"] > 0, "net_return"]
    losing_trades = trades.loc[trades["net_return"] < 0, "net_return"]
    average_win = float(winning_trades.mean()) if not winning_trades.empty else None
    average_loss = float(losing_trades.mean()) if not losing_trades.empty else None
    return {
        "mode": mode,
        "evidence_status": gate.evidence_status,
        "trade_count": int(len(trades)),
        "mean_net_return": float(trades["net_return"].mean()),
        "sum_net_return": float(trades["net_return"].sum()),
        "hit_rate": float((trades["net_return"] > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": (
            float(average_win / abs(average_loss))
            if average_win is not None and average_loss not in {None, 0.0}
            else None
        ),
        "final_equity": float(compounded.iloc[-1]),
        "compounded_return": float(compounded.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "max_single_trade_net_return": float(trades["net_return"].max()),
        "min_single_trade_net_return": float(trades["net_return"].min()),
        "cost_scenario": str(trades["cost_scenario"].iloc[0]),
    }


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "net_return", "drawdown"])
    equity = 1.0
    peak = 1.0
    rows: list[dict[str, Any]] = []
    for row in trades.itertuples(index=False):
        equity *= 1.0 + float(row.net_return)
        peak = max(peak, equity)
        rows.append(
            {
                "timestamp": row.exit_time,
                "equity": equity,
                "net_return": float(row.net_return),
                "drawdown": equity / peak - 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_long_only_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    gate: Phase2Gate,
    contract: Phase2StrategyContract = DEFAULT_PHASE2_CONTRACT,
    cost_scenario: CostScenario | None = None,
    mode: Phase2Mode = "sandbox",
) -> Phase2BacktestResult:
    """Run the first pre-registered long-only Phase 2 sandbox backtest.

    Official mode is fail-closed and requires the Phase 2 gate to be fully open.
    Sandbox mode may run while Future-OOS is pending, but the metadata marks the
    result as non-promotable.
    """

    contract.validate()
    gate.assert_mode_allowed(mode)
    scenario = cost_scenario or contract.cost_scenarios[1]
    bars = _prepare_bars(bars, contract)
    signals = _prepare_signals(signals, contract)
    bar_times = bars[contract.bar_time_column]

    trades: list[dict[str, Any]] = []
    min_entry_idx = 0
    selected_signal_count = 0
    skipped_no_next_bar_count = 0
    skipped_stale_entry_count = 0
    skipped_during_open_position_count = 0
    skipped_invalid_atr_count = 0
    data_gap_forced_close_count = 0
    entry_delay_hours: list[float] = []
    for signal in signals.itertuples(index=False):
        score = float(getattr(signal, contract.score_column))
        if score < contract.threshold:
            continue
        selected_signal_count += 1
        decision_time = getattr(signal, contract.decision_time_column)
        entry_idx = _entry_index_after_decision(
            bar_times,
            decision_time,
        )
        if entry_idx is None:
            skipped_no_next_bar_count += 1
            continue
        entry_time = bars.loc[entry_idx, contract.bar_time_column]
        entry_delay = max((entry_time - decision_time).total_seconds() / 3600.0, 0.0)
        if entry_delay > contract.max_bar_gap_hours:
            skipped_stale_entry_count += 1
            continue
        if not contract.allow_overlapping_positions and entry_idx < min_entry_idx:
            skipped_during_open_position_count += 1
            continue
        atr = float(bars.loc[max(entry_idx - 1, 0), contract.atr_column])
        if atr <= 0:
            skipped_invalid_atr_count += 1
            continue
        entry_price = float(bars.loc[entry_idx, "open"])
        exit_idx, exit_price, exit_reason = _resolve_exit(
            bars,
            contract,
            entry_idx=entry_idx,
            entry_price=entry_price,
            atr=atr,
        )
        if exit_reason == "data_gap_forced_close":
            data_gap_forced_close_count += 1
        holding_hours = _holding_hours(bars, contract, entry_idx, exit_idx)
        entry_delay_hours.append(entry_delay)
        trade_bars = bars.loc[entry_idx:exit_idx]
        returns = net_long_return(
            scenario,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_hours=holding_hours,
        )
        trades.append(
            {
                "candidate_id": contract.candidate_id,
                "decision_time": decision_time,
                "entry_time": entry_time,
                "exit_time": bars.loc[exit_idx, contract.bar_time_column],
                "score": score,
                "threshold": contract.threshold,
                "entry_delay_hours": entry_delay,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr": atr,
                "take_profit_atr": contract.take_profit_atr,
                "stop_loss_atr": contract.stop_loss_atr,
                "exit_reason": exit_reason,
                "holding_bars": int(exit_idx - entry_idx + 1),
                "holding_hours": holding_hours,
                "max_favorable_excursion": (
                    float(trade_bars["high"].max()) / entry_price - 1.0
                ),
                "max_adverse_excursion": (
                    float(trade_bars["low"].min()) / entry_price - 1.0
                ),
                "gross_return": returns["gross_return"],
                "net_return": returns["net_return"],
                "total_cost_return": returns["total_cost_return"],
                "cost_scenario": scenario.name,
                "phase2_mode": mode,
                "evidence_status": gate.evidence_status,
            }
        )
        min_entry_idx = exit_idx + 1 if not contract.allow_overlapping_positions else entry_idx + 1

    trade_frame = pd.DataFrame(trades)
    equity = _equity_curve(trade_frame)
    summary = _summarize(trade_frame, mode=mode, gate=gate)
    summary["cost_scenario"] = scenario.name
    execution_diagnostics = {
        "selected_signal_count": selected_signal_count,
        "skipped_no_next_bar_count": skipped_no_next_bar_count,
        "skipped_stale_entry_count": skipped_stale_entry_count,
        "skipped_during_open_position_count": skipped_during_open_position_count,
        "skipped_invalid_atr_count": skipped_invalid_atr_count,
        "data_gap_forced_close_count": data_gap_forced_close_count,
        "max_entry_delay_hours": max(entry_delay_hours) if entry_delay_hours else None,
        "mean_entry_delay_hours": (
            sum(entry_delay_hours) / len(entry_delay_hours)
            if entry_delay_hours
            else None
        ),
    }
    summary.update(execution_diagnostics)
    metadata = {
        "mode": mode,
        "contract": {
            "candidate_id": contract.candidate_id,
            "score_column": contract.score_column,
            "threshold": contract.threshold,
            "entry_rule": contract.entry_rule,
            "same_bar_policy": contract.same_bar_policy,
            "take_profit_atr": contract.take_profit_atr,
            "stop_loss_atr": contract.stop_loss_atr,
            "max_holding_bars": contract.max_holding_bars,
            "expected_bar_interval_hours": contract.expected_bar_interval_hours,
            "max_bar_gap_hours": contract.max_bar_gap_hours,
        },
        "gate": gate.as_dict(),
        "cost_scenario": scenario.name,
        "execution_diagnostics": execution_diagnostics,
        "not_promotable_reason": None
        if gate.official_allowed and mode == "official"
        else "future_oos_or_phase2_gate_not_passed",
    }
    return Phase2BacktestResult(
        trades=trade_frame,
        equity=equity,
        summary=summary,
        metadata=metadata,
    )
