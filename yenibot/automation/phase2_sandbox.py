from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yenibot.phase2 import CostScenario
from yenibot.phase2 import Phase2StrategyContract
from yenibot.phase2 import load_phase2_gate
from yenibot.phase2 import run_long_only_backtest
from yenibot.phase2.reporting import write_phase2_sandbox_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the gated Phase 2 sandbox backtest. Official mode fails closed "
            "unless Phase 2 readiness and Future-OOS promotion gates pass."
        )
    )
    parser.add_argument("--report-dir", required=True, help="Phase 1 report directory.")
    parser.add_argument("--bars", required=True, help="CSV with causal OHLCV/ATR bars.")
    parser.add_argument("--signals", required=True, help="CSV with decision_time and prob_long.")
    parser.add_argument("--output-dir", required=True, help="Directory for sandbox outputs.")
    parser.add_argument("--mode", default="sandbox", choices=["sandbox", "official"])
    parser.add_argument("--threshold", type=float, default=0.42674046854178105)
    parser.add_argument("--cost-name", default="base")
    parser.add_argument("--entry-fee-bps", type=float, default=4.0)
    parser.add_argument("--exit-fee-bps", type=float, default=4.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=2.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=2.0)
    parser.add_argument("--funding-bps-per-8h", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate = load_phase2_gate(Path(args.report_dir))
    contract = Phase2StrategyContract(threshold=float(args.threshold))
    scenario = CostScenario(
        name=str(args.cost_name),
        entry_fee_bps=float(args.entry_fee_bps),
        exit_fee_bps=float(args.exit_fee_bps),
        entry_slippage_bps=float(args.entry_slippage_bps),
        exit_slippage_bps=float(args.exit_slippage_bps),
        funding_bps_per_8h=float(args.funding_bps_per_8h),
    )
    bars = pd.read_csv(args.bars)
    signals = pd.read_csv(args.signals)
    result = run_long_only_backtest(
        bars,
        signals,
        gate=gate,
        contract=contract,
        cost_scenario=scenario,
        mode=args.mode,
    )
    write_phase2_sandbox_report(args.output_dir, result)
    print(
        f"phase2_sandbox_written output_dir={args.output_dir} "
        f"mode={args.mode} evidence_status={result.summary.get('evidence_status')} "
        f"trades={result.summary.get('trade_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
