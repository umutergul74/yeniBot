from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal


Phase2Mode = Literal["sandbox", "official"]
SameBarPolicy = Literal["stop_first", "take_profit_first", "skip_ambiguous"]


@dataclass(frozen=True)
class CostScenario:
    """Explicit Phase 2 cost assumptions, expressed in basis points."""

    name: str = "base"
    entry_fee_bps: float = 4.0
    exit_fee_bps: float = 4.0
    entry_slippage_bps: float = 2.0
    exit_slippage_bps: float = 2.0
    funding_bps_per_8h: float = 0.0

    def round_trip_cost_fraction(self, holding_hours: float = 0.0) -> float:
        fee_bps = self.entry_fee_bps + self.exit_fee_bps
        slippage_bps = self.entry_slippage_bps + self.exit_slippage_bps
        funding_bps = max(float(holding_hours), 0.0) / 8.0 * self.funding_bps_per_8h
        return (fee_bps + slippage_bps + funding_bps) / 10_000.0


@dataclass(frozen=True)
class Phase2StrategyContract:
    """Pre-registered first Phase 2 long-only strategy contract."""

    candidate_id: str = "control_recent3_equal_v2"
    score_column: str = "prob_long"
    decision_time_column: str = "decision_time"
    bar_time_column: str = "bar_close_time"
    threshold: float = 0.42674046854178105
    side: Literal["long_only"] = "long_only"
    entry_rule: Literal["next_bar_open"] = "next_bar_open"
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 5.0
    max_holding_bars: int = 10
    atr_column: str = "atr_14"
    same_bar_policy: SameBarPolicy = "stop_first"
    allow_overlapping_positions: bool = False
    cost_scenarios: tuple[CostScenario, ...] = field(
        default_factory=lambda: (
            CostScenario(
                name="optimistic",
                entry_fee_bps=2.0,
                exit_fee_bps=2.0,
                entry_slippage_bps=1.0,
                exit_slippage_bps=1.0,
                funding_bps_per_8h=0.0,
            ),
            CostScenario(name="base"),
            CostScenario(
                name="adverse",
                entry_fee_bps=5.0,
                exit_fee_bps=5.0,
                entry_slippage_bps=5.0,
                exit_slippage_bps=5.0,
                funding_bps_per_8h=2.0,
            ),
        )
    )

    def validate(self) -> None:
        if self.side != "long_only":
            raise ValueError("The first Phase 2 contract is long-only.")
        if self.entry_rule != "next_bar_open":
            raise ValueError("The first Phase 2 contract must use next-bar open fills.")
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("Signal threshold must be a probability-like score in (0, 1).")
        if self.take_profit_atr <= 0 or self.stop_loss_atr <= 0:
            raise ValueError("ATR exits must be positive.")
        if self.max_holding_bars <= 0:
            raise ValueError("Maximum holding bars must be positive.")
        if self.same_bar_policy not in {"stop_first", "take_profit_first", "skip_ambiguous"}:
            raise ValueError(f"Unsupported same-bar ambiguity policy: {self.same_bar_policy}")


DEFAULT_PHASE2_CONTRACT = Phase2StrategyContract()
