"""Phase 2 sandbox services.

The modules in this package implement the research backtest skeleton behind a
fail-closed Phase 2 gate.  They may be used to prepare execution, cost, and
portfolio accounting while the frozen Future-OOS gate is still pending, but
their outputs remain sandbox evidence until Phase 1 explicitly unlocks Phase 2.
"""

from yenibot.phase2.adapter import Phase2InputBuildResult
from yenibot.phase2.adapter import build_phase2_sandbox_inputs
from yenibot.phase2.adapter import phase2_inputs_from_predictions
from yenibot.phase2.contracts import CostScenario
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.engine import Phase2BacktestResult
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.readiness import load_phase2_gate
from yenibot.phase2.variants import Phase2StrategyVariantSpec
from yenibot.phase2.variants import phase2_strategy_registry
from yenibot.phase2.variants import registered_phase2_strategy_variants

__all__ = [
    "CostScenario",
    "Phase2BacktestResult",
    "Phase2Gate",
    "Phase2InputBuildResult",
    "Phase2StrategyContract",
    "Phase2StrategyVariantSpec",
    "build_phase2_sandbox_inputs",
    "load_phase2_gate",
    "phase2_inputs_from_predictions",
    "phase2_strategy_registry",
    "registered_phase2_strategy_variants",
    "run_long_only_backtest",
]
