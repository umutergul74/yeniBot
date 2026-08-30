from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math
import pandas as pd

from yenibot.phase2.contracts import CostScenario


def cost_breakdown(
    scenario: CostScenario,
    *,
    entry_price: float,
    exit_price: float,
    holding_hours: float,
    funding_return: float | None = None,
    charge_exit: bool = True,
) -> dict[str, Any]:
    """Return itemized long-trade costs as return fractions."""

    scenario.validate()
    if (
        not all(math.isfinite(v) for v in (entry_price, exit_price, holding_hours))
        or entry_price <= 0
        or exit_price <= 0
    ):
        raise ValueError("Entry and exit prices must be positive.")
    ratio = exit_price / entry_price
    entry_slippage = scenario.entry_slippage_bps / 10_000.0
    exit_slippage = (
        ratio * scenario.exit_slippage_bps / 10_000.0 if charge_exit else 0.0
    )
    entry_fee = scenario.entry_fee_bps / 10_000.0 * (1.0 + entry_slippage)
    exit_fee = (
        scenario.exit_fee_bps / 10_000.0 * (ratio - exit_slippage)
        if charge_exit
        else 0.0
    )
    funding = (
        max(float(holding_hours), 0.0) / 8.0 * scenario.funding_bps_per_8h / 10_000.0
        if funding_return is None
        else float(funding_return)
    )
    if not math.isfinite(funding):
        raise ValueError("Funding return must be finite")
    total = entry_fee + exit_fee + entry_slippage + exit_slippage + funding
    return {
        "scenario": scenario.name,
        "entry_fee_return": entry_fee,
        "exit_fee_return": exit_fee,
        "entry_slippage_return": entry_slippage,
        "exit_slippage_return": exit_slippage,
        "funding_return": funding,
        "funding_basis": "fixed_rate_duration_estimate"
        if funding_return is None
        else "historical_events",
        "entry_fill_price": entry_price * (1.0 + entry_slippage),
        "exit_fill_price": exit_price - entry_price * exit_slippage,
        "total_cost_return": total,
        "scenario_config": asdict(scenario),
    }


def net_long_return(
    scenario: CostScenario,
    *,
    entry_price: float,
    exit_price: float,
    holding_hours: float,
    funding_return: float | None = None,
    charge_exit: bool = True,
) -> dict[str, Any]:
    gross = (float(exit_price) - float(entry_price)) / float(entry_price)
    costs = cost_breakdown(
        scenario,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        holding_hours=float(holding_hours),
        funding_return=funding_return,
        charge_exit=charge_exit,
    )
    return {
        "gross_return": gross,
        "net_return": gross - costs["total_cost_return"],
        **costs,
    }


def validate_funding_events(events: pd.DataFrame | None) -> pd.DataFrame | None:
    if events is None:
        return None
    frame = events.copy()
    required = ["timestamp", "funding_rate", "mark_price"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Funding events require {required}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for name in required[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not frame[name].map(math.isfinite).all():
            raise ValueError("Funding events must contain finite rates and mark prices")
    if (
        frame["timestamp"].isna().any()
        or frame["timestamp"].duplicated().any()
        or (frame["mark_price"] <= 0).any()
    ):
        raise ValueError("Invalid funding timestamps/mark prices")
    return frame.sort_values("timestamp").reset_index(drop=True)


def historical_funding_return(events, *, entry_time, exit_time, entry_price):
    if events is None:
        return None
    # Boundary rule: entry at a funding timestamp pays/receives; exit at that
    # timestamp does not. Intrabar exits use the declared bar-close proxy.
    held = events.loc[(events.timestamp >= entry_time) & (events.timestamp < exit_time)]
    return float((held.funding_rate * held.mark_price / entry_price).sum())
