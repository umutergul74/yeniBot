from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.contracts import Phase2StrategyContract


@dataclass(frozen=True)
class Phase2StrategyVariantSpec:
    """Immutable metadata for a deliberately small Phase 2 strategy family."""

    contract: Phase2StrategyContract
    status: str
    rationale: str
    selection_allowed_on_current_test: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"]["cost_scenarios"] = [
            asdict(item) for item in self.contract.cost_scenarios
        ]
        return payload


def registered_phase2_strategy_variants(
    *,
    candidate_id: str,
    threshold: float,
) -> tuple[Phase2StrategyVariantSpec, ...]:
    """Return the bounded, audit-visible Phase 2 sandbox strategy family.

    The variants were registered after the first baseline result was reviewed.
    They are exploratory engineering hypotheses and cannot be selected or
    promoted on the already-seen test window.
    """

    common = replace(
        DEFAULT_PHASE2_CONTRACT,
        candidate_id=candidate_id,
        threshold=threshold,
    )
    return (
        Phase2StrategyVariantSpec(
            contract=common,
            status="baseline_reference",
            rationale="Preserve the Phase 1 label-mirroring exit as the immutable reference.",
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="breakeven_after_1atr_v1",
                exit_policy="breakeven",
                breakeven_trigger_atr=1.0,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "After a completed bar reaches +1 ATR, move the stop to entry "
                "for subsequent bars; never assume intrabar ordering."
            ),
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="atr_trailing_1atr_after_1atr_v1",
                exit_policy="atr_trailing",
                breakeven_trigger_atr=1.0,
                trailing_stop_atr=1.0,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "Activate a causal 1 ATR trailing stop only after a completed "
                "bar reaches +1 ATR, with breakeven as the minimum stop."
            ),
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="time_stop_6bar_v1",
                max_holding_bars=6,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "Test a shorter fixed signal lifetime without changing the "
                "frozen entry threshold or ATR barriers."
            ),
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="score_margin_05_fixed_atr_v1",
                min_score_margin=0.05,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "Require the frozen score to clear the baseline threshold by "
                "at least 0.05 before accepting a trade; this is a bounded "
                "entry-quality filter, not a threshold promotion."
            ),
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="score_margin_08_fixed_atr_v1",
                min_score_margin=0.08,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "Test a stricter score-margin buffer as a cost-pressure proxy "
                "while leaving the frozen model and official threshold unchanged."
            ),
        ),
        Phase2StrategyVariantSpec(
            contract=replace(
                common,
                strategy_id="score_margin_05_time_stop_6bar_v1",
                min_score_margin=0.05,
                max_holding_bars=6,
            ),
            status="exploratory_registered_after_baseline_review",
            rationale=(
                "Combine the first score-margin entry filter with the shorter "
                "signal lifetime suggested by max-holding forensics."
            ),
        ),
    )


def phase2_strategy_registry(
    *,
    candidate_id: str,
    threshold: float,
) -> dict[str, Any]:
    variants = registered_phase2_strategy_variants(
        candidate_id=candidate_id,
        threshold=threshold,
    )
    return {
        "registry_version": "phase2_strategy_registry_v1",
        "created_after_baseline_review": True,
        "selection_allowed_on_current_test": False,
        "clean_confirmation_window_required": True,
        "automatic_winner_selection": False,
        "trial_count": len(variants),
        "variants": [item.as_dict() for item in variants],
    }
