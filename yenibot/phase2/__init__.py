"""Phase 2 sandbox services.

The modules in this package implement the research backtest skeleton behind a
fail-closed Phase 2 gate.  They may be used to prepare execution, cost, and
portfolio accounting while the frozen Future-OOS gate is still pending, but
their outputs remain sandbox evidence until Phase 1 explicitly unlocks Phase 2.
"""

from yenibot.phase2.contracts import CostScenario
from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.engine import Phase2BacktestResult
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.readiness import load_phase2_gate

__all__ = [
    "CostScenario",
    "Phase2BacktestResult",
    "Phase2Gate",
    "Phase2StrategyContract",
    "load_phase2_gate",
    "run_long_only_backtest",
]
