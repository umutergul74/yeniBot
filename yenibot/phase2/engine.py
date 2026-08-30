from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.contracts import CostScenario
from yenibot.phase2.contracts import Phase2Mode
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.costs import net_long_return
from yenibot.phase2.costs import historical_funding_return, validate_funding_events
from yenibot.phase2.market_contract import normalize_execution_inputs
from yenibot.phase2.accounting import marked_equity_curve
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.risk import Phase2RiskPolicy


@dataclass(frozen=True)
class Phase2BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExitResolution:
    exit_idx: int
    exit_price: float
    exit_reason: str
    initial_stop_price: float
    take_profit_price: float
    final_active_stop_price: float
    dynamic_stop_activated: bool
    exit_at_open: bool = False


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _prepare_bars(
    frame: pd.DataFrame, contract: Phase2StrategyContract
) -> pd.DataFrame:
    required = {contract.bar_time_column, "open", "high", "low", "close", contract.atr_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Bars are missing required columns: {missing}")
    bars = frame.copy()
    bars[contract.bar_time_column] = bars[contract.bar_time_column].map(_to_timestamp)
    bars = bars.sort_values(contract.bar_time_column)
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
    candidates = bar_times[bar_times >= decision_time]
    if candidates.empty:
        return None
    return int(candidates.index[0])


def _holding_hours(
    bars: pd.DataFrame,
    contract: Phase2StrategyContract,
    entry_idx: int,
    exit_idx: int,
) -> float:
    start = bars.loc[entry_idx, "bar_open_time"]
    end = bars.loc[exit_idx, contract.bar_time_column]
    return max((end - start).total_seconds() / 3600.0, 0.0)


def _entry_filter_rejection_reason(
    contract: Phase2StrategyContract,
    *,
    score: float,
    entry_price: float,
    atr: float,
) -> str | None:
    score_margin = score - contract.threshold
    if score_margin < contract.min_score_margin:
        return "score_margin_below_min"
    atr_fraction = atr / entry_price if entry_price > 0 else None
    if atr_fraction is None:
        return "invalid_entry_price_for_atr_fraction"
    if (
        contract.min_entry_atr_fraction is not None
        and atr_fraction < contract.min_entry_atr_fraction
    ):
        return "entry_atr_fraction_below_min"
    if (
        contract.max_entry_atr_fraction is not None
        and atr_fraction > contract.max_entry_atr_fraction
    ):
        return "entry_atr_fraction_above_max"
    return None


def _resolve_exit(
    bars: pd.DataFrame,
    contract: Phase2StrategyContract,
    *,
    entry_idx: int,
    entry_price: float,
    atr: float,
) -> ExitResolution:
    take_price = entry_price + contract.take_profit_atr * atr
    initial_stop_price = entry_price - contract.stop_loss_atr * atr
    active_stop_price = initial_stop_price
    active_stop_reason = "stop_loss"
    dynamic_stop_activated = False
    running_high = entry_price
    last_idx = min(entry_idx + contract.max_holding_bars - 1, len(bars) - 1)

    for idx in range(entry_idx, last_idx + 1):
        if idx > entry_idx:
            previous_time = bars.loc[idx - 1, contract.bar_time_column]
            current_time = bars.loc[idx, contract.bar_time_column]
            gap_hours = (current_time - previous_time).total_seconds() / 3600.0
            if gap_hours > contract.max_bar_gap_hours:
                prior_idx = idx - 1
                return ExitResolution(
                    exit_idx=prior_idx,
                    exit_price=float(bars.loc[prior_idx, "close"]),
                    exit_reason="data_gap_censored",
                    initial_stop_price=initial_stop_price,
                    take_profit_price=take_price,
                    final_active_stop_price=active_stop_price,
                    dynamic_stop_activated=dynamic_stop_activated,
                )
        opening = float(bars.loc[idx, "open"])
        if idx > entry_idx and opening <= active_stop_price:
            return ExitResolution(
                idx,
                opening,
                f"{active_stop_reason}_gap_open",
                initial_stop_price,
                take_price,
                active_stop_price,
                dynamic_stop_activated,
                True,
            )
        high = float(bars.loc[idx, "high"])
        low = float(bars.loc[idx, "low"])
        touched_take = high >= take_price
        touched_stop = low <= active_stop_price
        if touched_take and touched_stop:
            if contract.same_bar_policy == "skip_ambiguous":
                return ExitResolution(
                    exit_idx=idx,
                    exit_price=float(bars.loc[idx, "close"]),
                    exit_reason="ambiguous_same_bar_skipped_to_close",
                    initial_stop_price=initial_stop_price,
                    take_profit_price=take_price,
                    final_active_stop_price=active_stop_price,
                    dynamic_stop_activated=dynamic_stop_activated,
                )
            if contract.same_bar_policy == "take_profit_first":
                return ExitResolution(
                    exit_idx=idx,
                    exit_price=take_price,
                    exit_reason="take_profit_same_bar",
                    initial_stop_price=initial_stop_price,
                    take_profit_price=take_price,
                    final_active_stop_price=active_stop_price,
                    dynamic_stop_activated=dynamic_stop_activated,
                )
            return ExitResolution(
                exit_idx=idx,
                exit_price=active_stop_price,
                exit_reason=f"{active_stop_reason}_same_bar_conservative",
                initial_stop_price=initial_stop_price,
                take_profit_price=take_price,
                final_active_stop_price=active_stop_price,
                dynamic_stop_activated=dynamic_stop_activated,
            )
        if touched_stop:
            return ExitResolution(
                exit_idx=idx,
                exit_price=active_stop_price,
                exit_reason=active_stop_reason,
                initial_stop_price=initial_stop_price,
                take_profit_price=take_price,
                final_active_stop_price=active_stop_price,
                dynamic_stop_activated=dynamic_stop_activated,
            )
        if touched_take:
            return ExitResolution(
                exit_idx=idx,
                exit_price=take_price,
                exit_reason="take_profit",
                initial_stop_price=initial_stop_price,
                take_profit_price=take_price,
                final_active_stop_price=active_stop_price,
                dynamic_stop_activated=dynamic_stop_activated,
            )

        running_high = max(running_high, high)
        if (
            contract.exit_policy in {"breakeven", "atr_trailing"}
            and contract.breakeven_trigger_atr is not None
            and running_high >= entry_price + contract.breakeven_trigger_atr * atr
        ):
            candidate_stop = entry_price
            candidate_reason = "breakeven_stop"
            if (
                contract.exit_policy == "atr_trailing"
                and contract.trailing_stop_atr is not None
            ):
                trailing_stop = running_high - contract.trailing_stop_atr * atr
                if trailing_stop > candidate_stop:
                    candidate_stop = trailing_stop
                    candidate_reason = "atr_trailing_stop"
            if candidate_stop > active_stop_price:
                active_stop_price = candidate_stop
                active_stop_reason = candidate_reason
                dynamic_stop_activated = True

    return ExitResolution(
        exit_idx=last_idx,
        exit_price=float(bars.loc[last_idx, "close"]),
        exit_reason=(
            "max_holding_bars"
            if last_idx == entry_idx + contract.max_holding_bars - 1
            else "end_of_data_censored"
        ),
        initial_stop_price=initial_stop_price,
        take_profit_price=take_price,
        final_active_stop_price=active_stop_price,
        dynamic_stop_activated=dynamic_stop_activated,
    )


def _summarize(
    trades: pd.DataFrame,
    *,
    mode: Phase2Mode,
    gate: Phase2Gate,
    risk_policy: Phase2RiskPolicy | None = None,
) -> dict[str, Any]:
    if "trade_status" in trades.columns:
        trades = trades.loc[trades["trade_status"].eq("completed")]
    if trades.empty:
        initial_equity = risk_policy.initial_equity if risk_policy is not None else 1.0
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
            "initial_equity": initial_equity,
            "final_equity": initial_equity,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "max_single_trade_net_return": None,
            "min_single_trade_net_return": None,
            "return_basis": (
                "risk_sized_portfolio_return"
                if risk_policy is not None
                else "full_notional_trade_return"
            ),
        }
    return_column = (
        "portfolio_return" if "portfolio_return" in trades.columns else "net_return"
    )
    returns = pd.to_numeric(trades[return_column], errors="coerce").fillna(0.0)
    wins = returns.loc[returns > 0].sum()
    losses = -returns.loc[returns < 0].sum()
    compounded = (1.0 + returns).cumprod()
    running_peak = compounded.cummax().clip(lower=1.0)
    drawdown = compounded / running_peak - 1.0
    winning_trades = returns.loc[returns > 0]
    losing_trades = returns.loc[returns < 0]
    average_win = float(winning_trades.mean()) if not winning_trades.empty else None
    average_loss = float(losing_trades.mean()) if not losing_trades.empty else None
    initial_equity = (
        float(trades["equity_before"].iloc[0])
        if "equity_before" in trades.columns
        else 1.0
    )
    final_equity = (
        float(trades["equity_after"].iloc[-1])
        if "equity_after" in trades.columns
        else float(compounded.iloc[-1])
    )
    return {
        "mode": mode,
        "evidence_status": gate.evidence_status,
        "trade_count": int(len(trades)),
        "mean_net_return": float(returns.mean()),
        "sum_net_return": float(returns.sum()),
        "hit_rate": float((returns > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": (
            float(average_win / abs(average_loss))
            if average_win is not None and average_loss not in {None, 0.0}
            else None
        ),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "compounded_return": float(compounded.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "max_single_trade_net_return": float(returns.max()),
        "min_single_trade_net_return": float(returns.min()),
        "cost_scenario": str(trades["cost_scenario"].iloc[0]),
        "return_basis": (
            "risk_sized_portfolio_return"
            if risk_policy is not None
            else "full_notional_trade_return"
        ),
    }


def _equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "net_return", "drawdown"])
    if {"equity_before", "equity_after", "portfolio_return"}.issubset(trades.columns):
        peak = float(trades["equity_before"].iloc[0])
        rows: list[dict[str, Any]] = []
        for row in trades.itertuples(index=False):
            equity = float(row.equity_after)
            peak = max(peak, equity)
            rows.append(
                {
                    "timestamp": row.exit_time,
                    "equity": equity,
                    "net_return": float(row.portfolio_return),
                    "drawdown": equity / peak - 1.0,
                    "position_notional_fraction": float(
                        row.position_notional_fraction
                    ),
                }
            )
        return pd.DataFrame(rows)
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
    risk_policy: Phase2RiskPolicy | None = None,
    mode: Phase2Mode = "sandbox",
    funding_events: pd.DataFrame | None = None,
) -> Phase2BacktestResult:
    """Run the first pre-registered long-only Phase 2 sandbox backtest.

    Official mode is fail-closed and requires the Phase 2 gate to be fully open.
    Sandbox mode may run while Future-OOS is pending, but the metadata marks the
    result as non-promotable.
    """

    contract.validate()
    if risk_policy is not None:
        risk_policy.validate()
    gate.assert_mode_allowed(mode)
    scenario = cost_scenario or contract.cost_scenarios[1]
    scenario.validate()
    if mode == "official" and funding_events is None:
        raise ValueError("Official accounting requires historical funding events")
    bars = _prepare_bars(bars, contract)
    signals = _prepare_signals(signals, contract)
    bars, signals, legacy_times = normalize_execution_inputs(bars, signals, contract)
    if funding_events is not None:
        funding_events = funding_events.copy()
        funding_events["timestamp"] = pd.to_datetime(
            funding_events["timestamp"], utc=True
        )
        ordered = funding_events.timestamp.sort_values()
        tolerance = pd.Timedelta(hours=8, minutes=1)
        in_window = ordered.loc[
            (ordered >= bars.bar_open_time.min() - tolerance)
            & (ordered <= bars[contract.bar_time_column].max() + tolerance)
        ]
        if not bars.empty and (
            ordered.empty
            or ordered.min() > bars.bar_open_time.min() + tolerance
            or ordered.max() < bars[contract.bar_time_column].max() - tolerance
            or (in_window.diff().dropna() > tolerance).any()
        ):
            raise ValueError("Historical funding coverage is incomplete for these bars")
        funding_events = funding_events.loc[
            (funding_events.timestamp >= bars.bar_open_time.min())
            & (funding_events.timestamp < bars[contract.bar_time_column].max())
        ]
    funding_events = validate_funding_events(funding_events)
    bar_times = bars["bar_open_time"]

    trades: list[dict[str, Any]] = []
    min_entry_idx = 0
    selected_signal_count = 0
    skipped_no_next_bar_count = 0
    skipped_stale_entry_count = 0
    skipped_during_open_position_count = 0
    skipped_invalid_atr_count = 0
    entry_filter_passed_count = 0
    entry_filter_skipped_count = 0
    score_margin_filter_skipped_count = 0
    atr_fraction_filter_skipped_count = 0
    data_gap_forced_close_count = 0
    risk_daily_loss_skip_count = 0
    risk_drawdown_halt_skip_count = 0
    entry_delay_hours: list[float] = []
    portfolio_equity = risk_policy.initial_equity if risk_policy is not None else 1.0
    portfolio_peak = portfolio_equity
    portfolio_halted = False
    daily_start_equity: dict[Any, float] = {}
    daily_realized_pnl: dict[Any, float] = {}
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
        entry_time = bars.loc[entry_idx, "bar_open_time"]
        entry_delay = max((entry_time - decision_time).total_seconds() / 3600.0, 0.0)
        if entry_delay > contract.max_bar_gap_hours:
            skipped_stale_entry_count += 1
            continue
        if (
            entry_idx == 0
            or bars.loc[entry_idx - 1, contract.bar_time_column] != entry_time
        ):
            skipped_stale_entry_count += 1
            continue
        atr = float(bars.loc[entry_idx - 1, contract.atr_column])
        if atr <= 0:
            skipped_invalid_atr_count += 1
            continue
        entry_price = float(bars.loc[entry_idx, "open"])
        filter_rejection = _entry_filter_rejection_reason(
            contract,
            score=score,
            entry_price=entry_price,
            atr=atr,
        )
        if filter_rejection is not None:
            entry_filter_skipped_count += 1
            if filter_rejection == "score_margin_below_min":
                score_margin_filter_skipped_count += 1
            if filter_rejection.startswith("entry_atr_fraction"):
                atr_fraction_filter_skipped_count += 1
            continue
        entry_filter_passed_count += 1
        if not contract.allow_overlapping_positions and entry_idx < min_entry_idx:
            skipped_during_open_position_count += 1
            continue
        if risk_policy is not None:
            decision_day = decision_time.date()
            daily_start_equity.setdefault(decision_day, portfolio_equity)
            daily_realized_pnl.setdefault(decision_day, 0.0)
            if portfolio_halted:
                risk_drawdown_halt_skip_count += 1
                continue
            realized_loss_fraction = (
                daily_realized_pnl[decision_day] / daily_start_equity[decision_day]
            )
            if (
                realized_loss_fraction
                <= -risk_policy.daily_realized_loss_limit_fraction
            ):
                risk_daily_loss_skip_count += 1
                continue
        exit_resolution = _resolve_exit(
            bars,
            contract,
            entry_idx=entry_idx,
            entry_price=entry_price,
            atr=atr,
        )
        exit_idx = exit_resolution.exit_idx
        exit_price = exit_resolution.exit_price
        exit_reason = exit_resolution.exit_reason
        if exit_reason == "data_gap_censored":
            data_gap_forced_close_count += 1
        exit_time = bars.loc[
            exit_idx,
            "bar_open_time"
            if exit_resolution.exit_at_open
            else contract.bar_time_column,
        ]
        holding_hours = (exit_time - entry_time).total_seconds() / 3600.0
        censored = exit_reason.endswith("_censored")
        entry_delay_hours.append(entry_delay)
        trade_bars = bars.loc[entry_idx:exit_idx]
        returns = net_long_return(
            scenario,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_hours=holding_hours,
            funding_return=historical_funding_return(
                funding_events,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
            ),
            charge_exit=not censored,
        )
        risk_fields: dict[str, Any] = {}
        equity_before = portfolio_equity
        notional_fraction = 1.0
        if risk_policy is not None:
            stop_distance_fraction = max(
                (entry_price - exit_resolution.initial_stop_price) / entry_price,
                0.0,
            )
            risk_sizing_cost_fraction = scenario.round_trip_cost_fraction(
                holding_hours=(
                    contract.max_holding_bars * contract.expected_bar_interval_hours
                )
            )
            risk_sizing_loss_fraction = (
                stop_distance_fraction + risk_sizing_cost_fraction
            )
            notional_fraction = risk_policy.notional_fraction(
                stop_distance_fraction=risk_sizing_loss_fraction,
            )
            equity_before = portfolio_equity
            position_notional = equity_before * notional_fraction
            realized_pnl = position_notional * float(returns["net_return"])
            portfolio_equity = equity_before + realized_pnl
            portfolio_return = (
                realized_pnl / equity_before if equity_before > 0 else 0.0
            )
            realized_drawdown = (
                portfolio_equity / max(portfolio_peak, portfolio_equity) - 1.0
            )
            exit_day = exit_time.date()
            daily_start_equity.setdefault(exit_day, equity_before)
            daily_realized_pnl.setdefault(exit_day, 0.0)
            if not censored:
                daily_realized_pnl[exit_day] += realized_pnl
            if realized_drawdown <= -risk_policy.max_realized_drawdown_fraction:
                portfolio_halted = True
            risk_fields = {
                "risk_policy_id": risk_policy.policy_id,
                "risk_budget_fraction": risk_policy.risk_fraction_per_trade,
                "stop_distance_fraction": stop_distance_fraction,
                "risk_sizing_cost_fraction": risk_sizing_cost_fraction,
                "risk_sizing_loss_fraction": risk_sizing_loss_fraction,
                "position_notional_fraction": notional_fraction,
                "position_notional": position_notional,
                "equity_before": equity_before,
                "realized_portfolio_pnl": 0.0 if censored else realized_pnl,
                "unrealized_portfolio_pnl": realized_pnl if censored else 0.0,
                "portfolio_return": portfolio_return,
                "equity_after": portfolio_equity,
                "realized_drawdown_after": None if censored else realized_drawdown,
                "portfolio_halted_after_trade": portfolio_halted,
            }
        if risk_policy is None:
            position_notional = equity_before
            portfolio_equity = equity_before + position_notional * returns["net_return"]
        risk_fields.update(
            {
                "equity_before": equity_before,
                "equity_after": portfolio_equity,
                "position_notional": position_notional,
                "position_notional_fraction": notional_fraction,
                "portfolio_return": portfolio_equity / equity_before - 1,
            }
        )
        trades.append(
            {
                "strategy_id": contract.strategy_id,
                "candidate_id": contract.candidate_id,
                "decision_time": decision_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "trade_status": "censored" if censored else "completed",
                "exit_time_basis": "bar_open_gap_fill"
                if exit_resolution.exit_at_open
                else "bar_close_proxy_intrabar_time_unknown",
                "excursions_are_bar_bounds": True,
                "score": score,
                "threshold": contract.threshold,
                "score_margin": score - contract.threshold,
                "entry_delay_hours": entry_delay,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "atr": atr,
                "entry_atr_fraction": atr / entry_price if entry_price > 0 else None,
                "take_profit_atr": contract.take_profit_atr,
                "stop_loss_atr": contract.stop_loss_atr,
                "exit_policy": contract.exit_policy,
                "initial_stop_price": exit_resolution.initial_stop_price,
                "take_profit_price": exit_resolution.take_profit_price,
                "final_active_stop_price": exit_resolution.final_active_stop_price,
                "dynamic_stop_activated": exit_resolution.dynamic_stop_activated,
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
                **{
                    name: returns[name]
                    for name in (
                        "entry_fee_return",
                        "exit_fee_return",
                        "entry_slippage_return",
                        "exit_slippage_return",
                        "funding_return",
                        "funding_basis",
                        "entry_fill_price",
                        "exit_fill_price",
                    )
                },
                "cost_scenario": scenario.name,
                "phase2_mode": mode,
                "evidence_status": gate.evidence_status,
                **risk_fields,
            }
        )
        if risk_policy is not None:
            marks = marked_equity_curve(
                trade_bars,
                pd.DataFrame([trades[-1]]),
                contract=contract,
                scenario=scenario,
                initial_equity=equity_before,
                funding_events=funding_events,
            )
            # Realized guardrails retain their original meaning; also latch a
            # mark-to-market breach to block subsequent entries (not a live liquidation).
            for mark in marks["equity"]:
                portfolio_peak = max(portfolio_peak, float(mark))
                if (
                    mark / portfolio_peak - 1
                    <= -risk_policy.max_marked_drawdown_fraction
                ):
                    portfolio_halted = True
        if exit_reason == "data_gap_censored":
            bars = bars.iloc[: exit_idx + 1]
            break  # Balance after an unobserved interval is unknown, not flat.
        min_entry_idx = (
            exit_idx + 1 if not contract.allow_overlapping_positions else entry_idx + 1
        )

    trade_frame = pd.DataFrame(trades)
    equity = marked_equity_curve(
        bars,
        trade_frame,
        contract=contract,
        scenario=scenario,
        initial_equity=risk_policy.initial_equity if risk_policy else 1.0,
        funding_events=funding_events,
    )
    summary = _summarize(
        trade_frame,
        mode=mode,
        gate=gate,
        risk_policy=risk_policy,
    )
    summary["cost_scenario"] = scenario.name
    summary["completed_trade_compounded_return"] = summary["compounded_return"]
    if not equity.empty:
        summary["final_equity"] = float(equity.equity.iloc[-1])
        summary["compounded_return"] = (
            summary["final_equity"] / summary["initial_equity"] - 1
        )
        summary["max_drawdown"] = float(equity.drawdown.min())
    summary["censored_position_count"] = int(
        trade_frame.get("trade_status", pd.Series(dtype=str)).eq("censored").sum()
    )
    summary["accounting_version"] = "phase2_mtm_v2"
    summary["funding_basis"] = (
        "historical_events"
        if funding_events is not None
        else "fixed_rate_duration_estimate"
    )
    summary["data_contract_complete"] = data_gap_forced_close_count == 0
    execution_diagnostics = {
        "selected_signal_count": selected_signal_count,
        "skipped_no_next_bar_count": skipped_no_next_bar_count,
        "skipped_stale_entry_count": skipped_stale_entry_count,
        "skipped_during_open_position_count": skipped_during_open_position_count,
        "skipped_invalid_atr_count": skipped_invalid_atr_count,
        "entry_filter_passed_count": entry_filter_passed_count,
        "entry_filter_skipped_count": entry_filter_skipped_count,
        "score_margin_filter_skipped_count": score_margin_filter_skipped_count,
        "atr_fraction_filter_skipped_count": atr_fraction_filter_skipped_count,
        "entry_filter_pass_rate": (
            float(
                entry_filter_passed_count
                / (entry_filter_passed_count + entry_filter_skipped_count)
            )
            if entry_filter_passed_count + entry_filter_skipped_count > 0
            else None
        ),
        "data_gap_forced_close_count": data_gap_forced_close_count,
        "risk_daily_loss_skip_count": risk_daily_loss_skip_count,
        "risk_drawdown_halt_skip_count": risk_drawdown_halt_skip_count,
        "risk_portfolio_halted": portfolio_halted,
        "dynamic_stop_activation_count": (
            int(trade_frame["dynamic_stop_activated"].sum())
            if not trade_frame.empty
            else 0
        ),
        "dynamic_stop_activation_share": (
            float(trade_frame["dynamic_stop_activated"].mean())
            if not trade_frame.empty
            else 0.0
        ),
        "max_entry_delay_hours": max(entry_delay_hours) if entry_delay_hours else None,
        "mean_entry_delay_hours": (
            sum(entry_delay_hours) / len(entry_delay_hours)
            if entry_delay_hours
            else None
        ),
    }
    summary.update(execution_diagnostics)
    metadata = {
        "accounting_version": "phase2_mtm_v2",
        "equity_basis": "hourly_close_mark_to_market_not_intrabar_worst_case",
        "legacy_open_timestamps_normalized": legacy_times,
        "censored_positions_are_not_executed_exits": True,
        "funding_time_assumption": "entry_inclusive_exit_exclusive_bar_close_proxy",
        "mode": mode,
        "contract": {
            "strategy_id": contract.strategy_id,
            "candidate_id": contract.candidate_id,
            "score_column": contract.score_column,
            "threshold": contract.threshold,
            "entry_rule": contract.entry_rule,
            "same_bar_policy": contract.same_bar_policy,
            "take_profit_atr": contract.take_profit_atr,
            "stop_loss_atr": contract.stop_loss_atr,
            "max_holding_bars": contract.max_holding_bars,
            "min_score_margin": contract.min_score_margin,
            "min_entry_atr_fraction": contract.min_entry_atr_fraction,
            "max_entry_atr_fraction": contract.max_entry_atr_fraction,
            "exit_policy": contract.exit_policy,
            "breakeven_trigger_atr": contract.breakeven_trigger_atr,
            "trailing_stop_atr": contract.trailing_stop_atr,
            "expected_bar_interval_hours": contract.expected_bar_interval_hours,
            "max_bar_gap_hours": contract.max_bar_gap_hours,
        },
        "gate": gate.as_dict(),
        "cost_scenario": scenario.name,
        "risk_policy": asdict(risk_policy) if risk_policy is not None else None,
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
