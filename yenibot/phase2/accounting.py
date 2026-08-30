"""Hourly marked-to-market, non-overlapping reference-notional accounting."""

from __future__ import annotations

import pandas as pd
from yenibot.phase2.costs import cost_breakdown, historical_funding_return


def marked_equity_curve(
    bars, trades, *, contract, scenario, initial_equity, funding_events=None
):
    equity = peak = float(initial_equity)
    cumulative_cost = 0.0
    rows = []
    positions = iter(trades.to_dict("records"))
    trade = next(positions, None)
    previous = equity
    for bar in bars.to_dict("records"):
        time = bar[contract.bar_time_column]
        exposure = 0.0
        costs = cumulative_cost
        while trade is not None and time > trade["exit_time"]:
            equity = float(trade["equity_after"])
            cumulative_cost += trade["position_notional"] * trade["total_cost_return"]
            costs = cumulative_cost
            trade = next(positions, None)
        mark = equity
        if trade is not None and time > trade["entry_time"]:
            if time >= trade["exit_time"]:
                mark = float(trade["equity_after"])
                costs = (
                    cumulative_cost
                    + trade["position_notional"] * trade["total_cost_return"]
                )
                if trade.get("trade_status") == "censored":
                    exposure = (
                        trade["position_notional"] * bar["close"] / trade["entry_price"]
                    )
            else:
                price = float(bar["close"])
                funding = historical_funding_return(
                    funding_events,
                    entry_time=trade["entry_time"],
                    exit_time=time,
                    entry_price=trade["entry_price"],
                )
                cost = cost_breakdown(
                    scenario,
                    entry_price=trade["entry_price"],
                    exit_price=price,
                    holding_hours=(time - trade["entry_time"]).total_seconds() / 3600,
                    funding_return=funding,
                    charge_exit=False,
                )["total_cost_return"]
                mark = trade["equity_before"] + trade["position_notional"] * (
                    price / trade["entry_price"] - 1 - cost
                )
                costs = cumulative_cost + trade["position_notional"] * cost
                exposure = trade["position_notional"] * price / trade["entry_price"]
        peak = max(peak, mark)
        rows.append(
            {
                "timestamp": time,
                "equity": mark,
                "net_return": mark / previous - 1,
                "drawdown": mark / peak - 1,
                "exposure": exposure,
                "cumulative_cost": costs,
                "cash_equivalent": mark - exposure,
            }
        )
        previous = mark
    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "equity",
            "net_return",
            "drawdown",
            "exposure",
            "cumulative_cost",
            "cash_equivalent",
        ],
    )
