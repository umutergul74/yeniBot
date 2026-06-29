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
    *,
    min_index: int,
) -> int | None:
    candidates = bar_times[(bar_times > decision_time) & (bar_times.index >= min_index)]
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
        }
    wins = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()
    return {
        "mode": mode,
        "evidence_status": gate.evidence_status,
        "trade_count": int(len(trades)),
        "mean_net_return": float(trades["net_return"].mean()),
        "sum_net_return": float(trades["net_return"].sum()),
        "hit_rate": float((trades["net_return"] > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_single_trade_net_return": float(trades["net_return"].max()),
        "min_single_trade_net_return": float(trades["net_return"].min()),
        "cost_scenario": str(trades["cost_scenario"].iloc[0]),
    }


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "net_return"])
    equity = 1.0
    rows: list[dict[str, Any]] = []
    for row in trades.itertuples(index=False):
        equity *= 1.0 + float(row.net_return)
        rows.append(
            {
                "timestamp": row.exit_time,
                "equity": equity,
                "net_return": float(row.net_return),
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
    for signal in signals.itertuples(index=False):
        score = float(getattr(signal, contract.score_column))
        if score < contract.threshold:
            continue
        decision_time = getattr(signal, contract.decision_time_column)
        entry_idx = _entry_index_after_decision(
            bar_times,
            decision_time,
            min_index=min_entry_idx,
        )
        if entry_idx is None:
            continue
        atr = float(bars.loc[max(entry_idx - 1, 0), contract.atr_column])
        if atr <= 0:
            continue
        entry_price = float(bars.loc[entry_idx, "open"])
        exit_idx, exit_price, exit_reason = _resolve_exit(
            bars,
            contract,
            entry_idx=entry_idx,
            entry_price=entry_price,
            atr=atr,
        )
        holding_hours = _holding_hours(bars, contract, entry_idx, exit_idx)
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
                "entry_time": bars.loc[entry_idx, contract.bar_time_column],
                "exit_time": bars.loc[exit_idx, contract.bar_time_column],
                "score": score,
                "threshold": contract.threshold,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr": atr,
                "take_profit_atr": contract.take_profit_atr,
                "stop_loss_atr": contract.stop_loss_atr,
                "exit_reason": exit_reason,
                "holding_bars": int(exit_idx - entry_idx + 1),
                "holding_hours": holding_hours,
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
        },
        "gate": gate.as_dict(),
        "cost_scenario": scenario.name,
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
