"""No-fit, frozen-trade accounting with explicit funding-price uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yenibot.phase2.accounting import marked_equity_curve
from yenibot.phase2.costs import (
    historical_funding_return,
    net_long_return,
    validate_funding_events,
)


def validate_funding_grid(events, *, start, end):
    """Audit-specific BTCUSDT eight-hour coverage; do not round charge timestamps."""
    f = events.copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True, errors="raise")
    slots = f.timestamp.dt.floor("8h")
    expected = pd.date_range(
        pd.Timestamp(start).floor("8h"), pd.Timestamp(end).floor("8h"), freq="8h"
    )
    f = f.loc[slots.isin(expected)].copy()
    slots = f.timestamp.dt.floor("8h")
    if (
        f.timestamp.isna().any()
        or slots.duplicated().any()
        or set(slots) != set(expected)
        or (f.timestamp - slots).gt(pd.Timedelta(seconds=1)).any()
        or not np.isfinite(f.funding_rate).all()
        or ("symbol" in f and not f.symbol.eq("BTCUSDT").all())
    ):
        raise ValueError("Incomplete/ambiguous BTCUSDT historical funding grid")
    valid_marks = f.mark_price.dropna()
    if not np.isfinite(valid_marks).all() or (valid_marks <= 0).any():
        raise ValueError("Invalid reported settlement mark")
    return f.sort_values("timestamp").reset_index(drop=True)


def held_funding_mask(events, ledgers):
    mask = pd.Series(False, index=events.index)
    for ledger in ledgers:
        for trade in ledger.itertuples():
            mask |= events.timestamp.ge(trade.entry_time) & events.timestamp.lt(
                trade.exit_time
            )
    return mask


def funding_price_scenarios(events, minute_bars, *, needed_mask=None):
    """Rate signs determine cost endpoints. Interval prices are NEVER exact marks."""
    f = events.copy()
    missing = f.mark_price.isna()
    needed = (
        np.ones(len(f), dtype=bool)
        if needed_mask is None
        else np.asarray(needed_mask, dtype=bool)
    )
    if needed.shape != (len(f),):
        raise ValueError("Funding needed mask must align with events")
    low = f.mark_price.to_numpy(copy=True)
    high = low.copy()
    for i in np.flatnonzero(missing & needed):
        minute = f.timestamp.iloc[i].floor("min")
        if minute not in minute_bars:
            raise ValueError(f"Missing funding mark interval: {minute}")
        row = minute_bars[minute]
        lo, hi = float(row["low"]), float(row["high"])
        if not np.isfinite([lo, hi]).all() or lo <= 0 or hi < lo:
            raise ValueError("Invalid one-minute mark interval")
        low[i], high[i] = lo, hi
    result = {}
    positive = f.funding_rate.to_numpy() >= 0
    for name, mark in (
        ("favorable_charge", np.where(positive, low, high)),
        ("adverse_charge", np.where(positive, high, low)),
    ):
        part = f.copy()
        part["mark_price"] = mark
        part["price_basis"] = np.where(
            missing & needed,
            "minute_interval_sensitivity",
            np.where(
                missing, "missing_outside_held_intervals", "reported_settlement_mark"
            ),
        )
        result[name] = part
    return result


def validate_frozen_ledger(ledger, bars):
    t = ledger.copy()
    for c in ("decision_time", "entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c], utc=True, errors="raise")
    if t.empty:
        return t
    if (
        t[["decision_time", "entry_time", "exit_time", "evaluation_fold"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Frozen ledger identity/time is missing")
    if t.duplicated(["evaluation_fold", "decision_time"]).any():
        raise ValueError("Duplicate frozen trade")
    numeric = [
        "entry_price",
        "exit_price",
        "atr",
        "holding_hours",
        "net_return",
        "initial_stop_price",
    ]
    if (
        not np.isfinite(t[numeric]).all().all()
        or (t[["entry_price", "exit_price", "atr"]] <= 0).any().any()
    ):
        raise ValueError("Invalid frozen prices or returns")
    if (t.initial_stop_price >= t.entry_price).any():
        raise ValueError("Frozen initial stop must define positive downside risk")
    if not t.trade_status.isin(["completed", "censored"]).all():
        raise ValueError("Unexpected source trade status")
    if (t.entry_time < t.decision_time).any() or (t.exit_time <= t.entry_time).any():
        raise ValueError("Invalid frozen holding interval")
    if not np.allclose(
        (t.exit_time - t.entry_time).dt.total_seconds() / 3600,
        t.holding_hours,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Frozen holding duration does not match timestamps")
    if not set(t.evaluation_fold).issubset(set(bars.fold)):
        raise ValueError("Frozen fold has no price coverage")
    for fold, group in t.groupby("evaluation_fold"):
        g = group.sort_values("entry_time")
        b = bars.loc[bars.fold.eq(fold)].sort_values("bar_open_time")
        if (
            not b.bar_open_time.diff().dropna().eq(pd.Timedelta(hours=1)).all()
            or not g.entry_time.isin(b.bar_open_time).all()
            or not g.exit_time.isin(b.bar_close_time).all()
            or (
                g.entry_time.iloc[1:].to_numpy() < g.exit_time.iloc[:-1].to_numpy()
            ).any()
        ):
            raise ValueError("Frozen trade coverage/gap/overlap violation")
        entries = b.set_index("bar_open_time").loc[g.entry_time, "open"].to_numpy()
        if not np.allclose(entries, g.entry_price, rtol=1e-12, atol=1e-10):
            raise ValueError("Frozen entry price differs from source next-open")
        censored = g.loc[g.trade_status.eq("censored")]
        if len(censored) > 1 or (
            not censored.empty
            and not censored.exit_time.eq(b.bar_close_time.max()).all()
        ):
            raise ValueError("Censoring must occur at the original fold boundary")
    return t.sort_values(["evaluation_fold", "entry_time"]).reset_index(drop=True)


def replay_frozen_ledger(
    bars, ledger, *, contract, scenario, funding_events=None, liquidate_terminal=False
):
    """Reprice identical saved fills; no re-entry, search, new sizing, or fitting."""
    ledger = validate_frozen_ledger(ledger, bars)
    if funding_events is not None:
        funding_events = validate_funding_grid(
            funding_events,
            start=bars.bar_open_time.min(),
            end=bars.bar_close_time.max(),
        )
        held = held_funding_mask(funding_events, [ledger])
        funding_events = validate_funding_events(funding_events.loc[held].copy())
    curves, parts = [], []
    chained_equity = 1.0
    chained_cost = 0.0
    for fold, b in bars.groupby("fold", sort=True):
        b = b.sort_values("bar_close_time").reset_index(drop=True)
        local_equity, records = 1.0, []
        for row in ledger.loc[ledger.evaluation_fold.eq(fold)].to_dict("records"):
            censored = row["trade_status"] == "censored"
            funding = historical_funding_return(
                funding_events,
                entry_time=row["entry_time"],
                exit_time=row["exit_time"],
                entry_price=row["entry_price"],
            )
            pnl = net_long_return(
                scenario,
                entry_price=row["entry_price"],
                exit_price=row["exit_price"],
                holding_hours=row["holding_hours"],
                funding_return=funding,
                charge_exit=not censored or liquidate_terminal,
            )
            held_events = (
                None
                if funding_events is None
                else funding_events.loc[
                    funding_events.timestamp.ge(row["entry_time"])
                    & funding_events.timestamp.lt(row["exit_time"])
                ]
            )
            price_basis = (
                "fixed_duration_estimate"
                if held_events is None
                else (
                    "minute_mark_interval_sensitivity"
                    if "price_basis" in held_events
                    and held_events.price_basis.eq("minute_interval_sensitivity").any()
                    else "reported_settlement_marks_or_no_held_events"
                )
            )
            before = local_equity
            local_equity *= 1 + pnl["net_return"]
            if local_equity <= 0 or not np.isfinite(local_equity):
                raise ValueError("Insolvent/nonfinite reference-notional replay")
            records.append(
                {
                    **row,
                    **pnl,
                    "source_trade_status": row["trade_status"],
                    "cost_scenario": scenario.name,
                    "funding_price_basis": price_basis,
                    "trade_status": "hypothetical_terminal_liquidation"
                    if censored and liquidate_terminal
                    else row["trade_status"],
                    "equity_before": before,
                    "equity_after": local_equity,
                    "position_notional": before,
                    "position_notional_fraction": 1.0,
                    "portfolio_return": pnl["net_return"],
                    "terminal_liquidation_sensitivity": bool(
                        censored and liquidate_terminal
                    ),
                }
            )
        trades = pd.DataFrame(records) if records else ledger.iloc[:0].copy()
        curve = marked_equity_curve(
            b,
            trades,
            contract=contract,
            scenario=scenario,
            initial_equity=1.0,
            funding_events=funding_events,
        )
        if not np.isclose(float(curve.equity.iloc[-1]), local_equity, rtol=1e-12):
            raise RuntimeError("Marked curve does not reconcile with frozen cashflows")
        for c in ("equity", "exposure", "cash_equivalent"):
            curve[c] *= chained_equity
        curve["cumulative_cost"] = chained_cost + curve.cumulative_cost * chained_equity
        curve["fold"] = fold
        chained_equity *= local_equity
        chained_cost = float(curve.cumulative_cost.iloc[-1])
        curves.append(curve)
        parts.append(trades)
    curve = pd.concat(curves, ignore_index=True)
    curve["drawdown"] = curve.equity / curve.equity.cummax().clip(lower=1.0) - 1
    curve["net_return"] = curve.equity / curve.equity.shift(fill_value=1.0) - 1
    trades = pd.concat(parts, ignore_index=True)
    completed = (
        trades.loc[trades.source_trade_status.eq("completed")]
        if not trades.empty
        else trades
    )
    stop_risk = (
        (trades.entry_price - trades.initial_stop_price) / trades.entry_price
        if not trades.empty
        else pd.Series(dtype=float)
    )
    held_hours = float(trades.holding_hours.sum()) if not trades.empty else 0.0
    summary = {
        "chained_independent_fold_return": float(curve.equity.iloc[-1] - 1),
        "hourly_contract_close_max_drawdown": float(curve.drawdown.min()),
        "completed_trade_return": float(np.prod(1 + completed.net_return) - 1)
        if not completed.empty
        else 0.0,
        "completed_trades": len(completed),
        "original_censored_positions": int(ledger.trade_status.eq("censored").sum()),
        "observed_fold_hours": len(bars),
        "occupied_hours": held_hours,
        "occupied_fraction_of_observed_hours": held_hours / len(bars),
        "mean_initial_stop_risk_fraction": float(stop_risk.mean())
        if len(stop_risk)
        else None,
        "p95_initial_stop_risk_fraction": float(stop_risk.quantile(0.95))
        if len(stop_risk)
        else None,
        "mean_net_r_multiple": float((trades.net_return / stop_risk).mean())
        if len(stop_risk)
        else None,
        "simple_net_bps_per_occupied_hour": float(
            trades.net_return.sum() / held_hours * 1e4
        )
        if held_hours
        else None,
        "funding_return_sum_per_unit_trade_notional": float(trades.funding_return.sum())
        if not trades.empty
        else 0.0,
        "terminal_liquidation_sensitivity": liquidate_terminal,
        "equity_basis": "hourly_contract_price_close_not_exchange_mark_or_intrabar_worst_case",
        "no_fit_no_trade_selection": True,
    }
    return summary, curve, trades
