from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from yenibot.phase2 import CostScenario
from yenibot.phase2 import Phase2StrategyContract
from yenibot.phase2 import build_phase2_sandbox_inputs
from yenibot.phase2 import load_phase2_gate
from yenibot.phase2 import run_long_only_backtest
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.reporting import write_phase2_sandbox_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the gated Phase 2 sandbox backtest. Official mode fails closed "
            "unless Phase 2 readiness and Future-OOS promotion gates pass."
        )
    )
    parser.add_argument("--report-dir", required=True, help="Phase 1 report directory.")
    parser.add_argument(
        "--bars",
        help=(
            "CSV with causal OHLCV/ATR bars. If omitted, inputs are generated "
            "from the frozen candidate predictions."
        ),
    )
    parser.add_argument(
        "--signals",
        help=(
            "CSV with decision_time and prob_long. If omitted, inputs are "
            "generated from the frozen candidate predictions."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for sandbox outputs.")
    parser.add_argument("--mode", default="sandbox", choices=["sandbox", "official"])
    parser.add_argument(
        "--checkpoint-dir",
        help=(
            "Checkpoint root containing experiments/<source_run_id>. Required "
            "for automatic input generation unless --scope-dir is supplied."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        help="Optional explicit frozen candidate manifest path.",
    )
    parser.add_argument(
        "--scope-dir",
        help=(
            "Optional explicit profile scope directory containing "
            "predictions_all.parquet/csv."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Prediction split used by automatic input generation. Use 'all' for all rows.",
    )
    parser.add_argument(
        "--build-inputs",
        action="store_true",
        help="Generate phase2_bars.csv and phase2_signals.csv before running.",
    )
    parser.add_argument(
        "--candidate-id",
        help="Override candidate id in the Phase 2 contract.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override signal threshold. Defaults to the frozen manifest threshold "
            "when auto-building, otherwise the default Phase 2 contract threshold."
        ),
    )
    parser.add_argument("--cost-name", default="base")
    parser.add_argument("--entry-fee-bps", type=float, default=4.0)
    parser.add_argument("--exit-fee-bps", type=float, default=4.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=2.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=2.0)
    parser.add_argument("--funding-bps-per-8h", type=float, default=0.0)
    parser.add_argument(
        "--all-cost-scenarios",
        action="store_true",
        help=(
            "Run the preregistered optimistic, base, and adverse cost scenarios. "
            "The root report remains the base scenario for compatibility."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    gate = load_phase2_gate(report_dir)
    manual_inputs = bool(args.bars or args.signals)
    if manual_inputs and not (args.bars and args.signals):
        raise ValueError("--bars and --signals must be supplied together.")

    input_build = None
    bars_path = Path(args.bars) if args.bars else None
    signals_path = Path(args.signals) if args.signals else None
    if args.build_inputs or not manual_inputs:
        input_build = build_phase2_sandbox_inputs(
            report_dir=report_dir,
            checkpoint_dir=args.checkpoint_dir,
            manifest_path=args.manifest_path,
            scope_dir=args.scope_dir,
            output_dir=output_dir,
            split=str(args.split),
            threshold=args.threshold,
        )
        bars_path = input_build.bars_path
        signals_path = input_build.signals_path

    if bars_path is None or signals_path is None:
        raise ValueError(
            "No Phase 2 inputs resolved. Supply --bars/--signals or allow "
            "automatic generation with --checkpoint-dir/--scope-dir."
        )

    threshold = float(
        args.threshold
        if args.threshold is not None
        else (
            input_build.threshold
            if input_build is not None
            else DEFAULT_PHASE2_CONTRACT.threshold
        )
    )
    candidate_id = str(
        args.candidate_id
        or (input_build.candidate_id if input_build is not None else "")
        or DEFAULT_PHASE2_CONTRACT.candidate_id
    )
    contract = Phase2StrategyContract(
        candidate_id=candidate_id,
        threshold=threshold,
    )
    custom_scenario = CostScenario(
        name=str(args.cost_name),
        entry_fee_bps=float(args.entry_fee_bps),
        exit_fee_bps=float(args.exit_fee_bps),
        entry_slippage_bps=float(args.entry_slippage_bps),
        exit_slippage_bps=float(args.exit_slippage_bps),
        funding_bps_per_8h=float(args.funding_bps_per_8h),
    )
    bars = pd.read_csv(bars_path)
    signals = pd.read_csv(signals_path)
    scenarios = (
        contract.cost_scenarios
        if args.all_cost_scenarios
        else (custom_scenario,)
    )
    results = []
    for scenario in scenarios:
        result = run_long_only_backtest(
            bars,
            signals,
            gate=gate,
            contract=contract,
            cost_scenario=scenario,
            mode=args.mode,
        )
        if input_build is not None:
            result.metadata["input_adapter"] = input_build.as_dict()
        results.append(result)
        if args.all_cost_scenarios:
            write_phase2_sandbox_report(
                output_dir / "cost_scenarios" / scenario.name,
                result,
            )
            if scenario.name == "base":
                write_phase2_sandbox_report(output_dir, result)
        else:
            write_phase2_sandbox_report(output_dir, result)

    if args.all_cost_scenarios:
        summary_frame = pd.DataFrame([result.summary for result in results])
        summary_frame.to_csv(
            output_dir / "phase2_cost_scenario_summary.csv",
            index=False,
        )
        trade_frames = [
            result.trades
            for result in results
            if not result.trades.empty
        ]
        if trade_frames:
            pd.concat(trade_frames, ignore_index=True).to_csv(
                output_dir / "phase2_trade_ledger_all_costs.csv",
                index=False,
            )
        equity_frames = []
        for result in results:
            equity = result.equity.copy()
            equity["cost_scenario"] = result.summary.get("cost_scenario")
            equity_frames.append(equity)
        pd.concat(equity_frames, ignore_index=True).to_csv(
            output_dir / "phase2_equity_curve_all_costs.csv",
            index=False,
        )

    base_result = next(
        (
            result
            for result in results
            if result.summary.get("cost_scenario") == "base"
        ),
        results[0],
    )
    print(
        f"phase2_sandbox_written output_dir={output_dir} "
        f"mode={args.mode} evidence_status={base_result.summary.get('evidence_status')} "
        f"trades={base_result.summary.get('trade_count')} "
        f"cost_scenarios={len(results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
