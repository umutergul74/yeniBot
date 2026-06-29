from __future__ import annotations

from dataclasses import asdict
from typing import Any

from yenibot.phase2.contracts import CostScenario


def cost_breakdown(
    scenario: CostScenario,
    *,
    entry_price: float,
    exit_price: float,
    holding_hours: float,
) -> dict[str, Any]:
    """Return itemized long-trade costs as return fractions."""

    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Entry and exit prices must be positive.")
    entry_fee = scenario.entry_fee_bps / 10_000.0
    exit_fee = scenario.exit_fee_bps / 10_000.0
    entry_slippage = scenario.entry_slippage_bps / 10_000.0
    exit_slippage = scenario.exit_slippage_bps / 10_000.0
    funding = max(float(holding_hours), 0.0) / 8.0 * scenario.funding_bps_per_8h / 10_000.0
    total = entry_fee + exit_fee + entry_slippage + exit_slippage + funding
    return {
        "scenario": scenario.name,
        "entry_fee_return": entry_fee,
        "exit_fee_return": exit_fee,
        "entry_slippage_return": entry_slippage,
        "exit_slippage_return": exit_slippage,
        "funding_return": funding,
        "total_cost_return": total,
        "scenario_config": asdict(scenario),
    }


def net_long_return(
    scenario: CostScenario,
    *,
    entry_price: float,
    exit_price: float,
    holding_hours: float,
) -> dict[str, Any]:
    gross = (float(exit_price) - float(entry_price)) / float(entry_price)
    costs = cost_breakdown(
        scenario,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        holding_hours=float(holding_hours),
    )
    return {
        "gross_return": gross,
        "net_return": gross - costs["total_cost_return"],
        **costs,
    }
