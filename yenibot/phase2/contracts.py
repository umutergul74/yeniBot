from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal
import math


Phase2Mode = Literal["sandbox", "official"]
SameBarPolicy = Literal["stop_first", "take_profit_first", "skip_ambiguous"]
ExitPolicy = Literal["fixed_atr", "breakeven", "atr_trailing"]


@dataclass(frozen=True)
class CostScenario:
    """Explicit Phase 2 cost assumptions, expressed in basis points."""

    name: str = "base"
    entry_fee_bps: float = 4.0
    exit_fee_bps: float = 4.0
    entry_slippage_bps: float = 2.0
    exit_slippage_bps: float = 2.0
    funding_bps_per_8h: float = 0.0

    def validate(self) -> None:
        values = (
            self.entry_fee_bps,
            self.exit_fee_bps,
            self.entry_slippage_bps,
            self.exit_slippage_bps,
            self.funding_bps_per_8h,
        )
        if not all(math.isfinite(v) for v in values) or any(
            v < 0 or v >= 10_000 for v in values[:4]
        ):
            raise ValueError(
                "Costs must be finite; fees/slippage must be in [0, 10000) bps"
            )

    def round_trip_cost_fraction(self, holding_hours: float = 0.0) -> float:
        fee_bps = self.entry_fee_bps + self.exit_fee_bps
        slippage_bps = self.entry_slippage_bps + self.exit_slippage_bps
        funding_bps = max(float(holding_hours), 0.0) / 8.0 * self.funding_bps_per_8h
        return (fee_bps + slippage_bps + funding_bps) / 10_000.0


@dataclass(frozen=True)
class Phase2StrategyContract:
    """Pre-registered first Phase 2 long-only strategy contract."""

    strategy_id: str = "baseline_fixed_atr_v1"
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
    min_score_margin: float = 0.0
    min_entry_atr_fraction: float | None = None
    max_entry_atr_fraction: float | None = None
    exit_policy: ExitPolicy = "fixed_atr"
    breakeven_trigger_atr: float | None = None
    trailing_stop_atr: float | None = None
    expected_bar_interval_hours: float = 1.0
    max_bar_gap_hours: float = 1.5
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
        if self.allow_overlapping_positions:
            raise ValueError(
                "Overlapping positions require an event-driven multi-position ledger; unsupported"
            )
        for name in (
            "threshold",
            "take_profit_atr",
            "stop_loss_atr",
            "min_score_margin",
            "expected_bar_interval_hours",
            "max_bar_gap_hours",
            "min_entry_atr_fraction",
            "max_entry_atr_fraction",
            "breakeven_trigger_atr",
            "trailing_stop_atr",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
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
        if self.min_score_margin < 0:
            raise ValueError("Minimum score margin must be non-negative.")
        if self.threshold + self.min_score_margin >= 1:
            raise ValueError("Threshold plus minimum score margin must stay below 1.")
        if (
            self.min_entry_atr_fraction is not None
            and self.min_entry_atr_fraction <= 0
        ):
            raise ValueError("Minimum entry ATR fraction must be positive when set.")
        if (
            self.max_entry_atr_fraction is not None
            and self.max_entry_atr_fraction <= 0
        ):
            raise ValueError("Maximum entry ATR fraction must be positive when set.")
        if (
            self.min_entry_atr_fraction is not None
            and self.max_entry_atr_fraction is not None
            and self.min_entry_atr_fraction > self.max_entry_atr_fraction
        ):
            raise ValueError(
                "Minimum entry ATR fraction cannot exceed the maximum entry ATR fraction."
            )
        if self.exit_policy not in {"fixed_atr", "breakeven", "atr_trailing"}:
            raise ValueError(f"Unsupported exit policy: {self.exit_policy}")
        if self.exit_policy in {"breakeven", "atr_trailing"}:
            if self.breakeven_trigger_atr is None or self.breakeven_trigger_atr <= 0:
                raise ValueError(
                    "Dynamic exit policies require a positive breakeven trigger."
                )
        if self.exit_policy == "atr_trailing":
            if self.trailing_stop_atr is None or self.trailing_stop_atr <= 0:
                raise ValueError("ATR trailing exits require a positive trailing distance.")
        if self.expected_bar_interval_hours <= 0:
            raise ValueError("Expected bar interval must be positive.")
        if self.max_bar_gap_hours < self.expected_bar_interval_hours:
            raise ValueError(
                "Maximum bar gap cannot be shorter than the expected bar interval."
            )
        if self.same_bar_policy not in {"stop_first", "take_profit_first", "skip_ambiguous"}:
            raise ValueError(f"Unsupported same-bar ambiguity policy: {self.same_bar_policy}")


DEFAULT_PHASE2_CONTRACT = Phase2StrategyContract()
