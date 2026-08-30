"""One bounded full-history density audit, separate from frozen-candidate inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from yenibot.automation.phase2_attribution import _holm_adjust
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.economic_attribution import (
    EconomicAttributionSpec,
    run_economic_attribution,
    write_economic_attribution,
)
from yenibot.phase2.full_oof import build_full_oof_inputs, file_sha256
from yenibot.phase2.readiness import load_phase2_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--spec", default="configs/full_oof_attribution_v1.json")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["exit_contract"] != "baseline_fixed_atr_v1":
        raise ValueError("This bounded audit supports the unchanged baseline exit only")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing OOF attribution run")
    bars, signals, source = build_full_oof_inputs(args.scope_dir, spec=spec)
    source["spec_sha256"] = file_sha256(spec_path)
    output.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output / "phase2_oof_bars.csv", index=False)
    signals.to_csv(output / "phase2_oof_signals.csv", index=False)
    (output / "phase2_oof_input_manifest.json").write_text(
        json.dumps(source, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Verified full OOF: {len(signals)} test rows; {signals.fold.nunique()} folds",
        flush=True,
    )
    gate = load_phase2_gate(args.report_dir)
    rows = []
    for percentile in spec["percentile_thresholds"]:
        contract = replace(
            DEFAULT_PHASE2_CONTRACT,
            candidate_id=source["candidate_id"],
            strategy_id=f"validation_cdf_q{round(percentile * 100):02d}_fixed_atr_v1",
            threshold=float(percentile),
        )
        print(f"Running {contract.strategy_id} ...", flush=True)
        result = run_economic_attribution(
            bars,
            signals,
            gate=gate,
            contract=contract,
            spec=EconomicAttributionSpec(
                permutations=spec["permutations"],
                seed=spec["seed"],
                evidence_scope=spec["evidence_scope"],
            ),
            source_metadata=source,
        )
        write_economic_attribution(output / contract.strategy_id, result)
        report = result.report
        base = report["actual"]["base"]
        null = report["rank_destroyed_null"]
        rows.append(
            {
                "strategy_id": contract.strategy_id,
                "percentile_threshold": percentile,
                "base_return": base["compounded_return"],
                "adverse_return": report["actual"]["adverse"]["compounded_return"],
                "completed_trades": base["trade_count"],
                "mean_net_return": base["mean_net_return"],
                "profit_factor": base["profit_factor"],
                "completed_trade_close_max_drawdown": base["max_drawdown"],
                "positive_fold_share": report["fold_diagnostics"][
                    "positive_fold_share"
                ],
                "null_median_return": null["return_median"],
                "raw_p_value": null["one_sided_p_value_actual_not_better_than_null"],
                "always_on_base_return": report["deterministic_controls"][
                    "always_on_long_context"
                ]["base"]["compounded_return"],
                "inverted_base_return": report["deterministic_controls"][
                    "inverted_model_ranking"
                ]["base"]["compounded_return"],
                "diagnostic_gate_passed": report["assessment"][
                    "diagnostic_gate_passed"
                ],
                "failed_criteria": "|".join(report["assessment"]["failed_criteria"]),
            }
        )
        print(
            f"Completed {contract.strategy_id}: base={base['compounded_return']:.6f}",
            flush=True,
        )
    adjusted = _holm_adjust({row["strategy_id"]: row["raw_p_value"] for row in rows})
    for row in rows:
        row["holm_p_value"] = adjusted[row["strategy_id"]]
        row["familywise_diagnostic_gate_passed"] = (
            row["diagnostic_gate_passed"] and row["holm_p_value"] <= 0.05
        )
    pd.DataFrame(rows).to_csv(output / "full_oof_attribution_suite.csv", index=False)
    (output / "full_oof_attribution_suite.json").write_text(
        json.dumps(
            {
                "spec": spec,
                "spec_sha256": source["spec_sha256"],
                "rows": rows,
                "multiplicity": "holm_across_all_four_density_policies_not_all_historical_research",
                "automatic_winner_selection_allowed": False,
                "promotion_allowed": False,
                "live_trading_allowed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
