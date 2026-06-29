from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase2RiskPolicy:
    """Pre-registered fixed-fractional portfolio guardrails.

    Scores are never interpreted as probabilities. Position size is derived
    only from the causal initial stop distance and a fixed equity risk budget.
    """

    policy_id: str = "fixed_fractional_guardrails_v1"
    initial_equity: float = 10_000.0
    risk_fraction_per_trade: float = 0.0025
    max_notional_fraction: float = 0.25
    daily_realized_loss_limit_fraction: float = 0.01
    max_realized_drawdown_fraction: float = 0.05
    allow_leverage: bool = False

    def validate(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Risk policy id cannot be empty.")
        if self.initial_equity <= 0:
            raise ValueError("Initial equity must be positive.")
        if not 0 < self.risk_fraction_per_trade < 1:
            raise ValueError("Risk fraction per trade must be in (0, 1).")
        if not 0 < self.max_notional_fraction:
            raise ValueError("Maximum notional fraction must be positive.")
        if not self.allow_leverage and self.max_notional_fraction > 1:
            raise ValueError("A non-leveraged risk policy cannot exceed 100% notional.")
        if not 0 < self.daily_realized_loss_limit_fraction < 1:
            raise ValueError("Daily realized loss limit must be in (0, 1).")
        if not 0 < self.max_realized_drawdown_fraction < 1:
            raise ValueError("Maximum realized drawdown must be in (0, 1).")

    def notional_fraction(self, *, stop_distance_fraction: float) -> float:
        """Return bounded notional/equity from the fixed loss budget."""

        self.validate()
        if stop_distance_fraction <= 0:
            raise ValueError("Stop distance fraction must be positive.")
        risk_sized = self.risk_fraction_per_trade / stop_distance_fraction
        return float(min(risk_sized, self.max_notional_fraction))


DEFAULT_PHASE2_RISK_POLICY = Phase2RiskPolicy()
