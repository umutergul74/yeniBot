"""Replay the whole pinned OOF density family under serial-preserving controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from yenibot.automation.phase2_attribution import _holm_adjust
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.economic_attribution import (
    ECONOMIC_ATTRIBUTION_VERSION,
    EconomicAttributionSpec,
    _run_fold_segmented_backtest,
    _stable_hash,
    _validate_inputs,
)
from yenibot.phase2.execution_cache import (
    assert_cache_matches_reference,
    build_fold_execution_cache,
)
from yenibot.phase2.full_oof import build_full_oof_inputs, file_sha256
from yenibot.phase2.readiness import load_phase2_gate
from yenibot.phase2.serial_attribution import (
    cache_with_score_threshold,
    circular_shift_scores,
    run_serial_null,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--attribution-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", default="configs/full_oof_attribution_v1.json")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    source_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_sha = file_sha256(spec_path)
    root, output = Path(args.attribution_dir), Path(args.output_dir)
    if not (root / "full_oof_attribution_suite.json").is_file():
        raise ValueError("The entire original density family must be completed first")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a serial-control audit")
    bars, signals, source = build_full_oof_inputs(args.scope_dir, spec=source_spec)
    spec = EconomicAttributionSpec(
        permutations=source_spec["permutations"],
        seed=source_spec["seed"],
        evidence_scope=source_spec["evidence_scope"],
    )
    contract = replace(
        DEFAULT_PHASE2_CONTRACT, candidate_id=source["candidate_id"], threshold=0.5
    )
    bars, signals, _ = _validate_inputs(bars, signals, contract=contract, spec=spec)
    scenarios = tuple(
        c for c in contract.cost_scenarios if c.name in ("base", "adverse")
    )
    base = next(c for c in scenarios if c.name == "base")
    cache = build_fold_execution_cache(bars, signals, contract=contract, scenario=base)
    groups = list(
        signals.groupby(["fold", "calendar_month"], sort=True).indices.values()
    )
    scores = signals[contract.score_column].to_numpy()
    gate = load_phase2_gate(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_reports = {}
    seed = source_spec["seed"] + 77111
    for threshold in source_spec["percentile_thresholds"]:
        strategy = f"validation_cdf_q{round(threshold * 100):02d}_fixed_atr_v1"
        variant = replace(contract, threshold=threshold, strategy_id=strategy)
        report_path = root / strategy / "phase2_economic_attribution.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_contract = _stable_hash(
            {
                "attribution_version": ECONOMIC_ATTRIBUTION_VERSION,
                "fold_segmentation": "independent_walk_forward_folds",
                "strategy": asdict(variant),
                "spec": asdict(spec),
                "scenarios": [asdict(c) for c in scenarios],
            }
        )
        _, _, diagnostics = _validate_inputs(bars, signals, contract=variant, spec=spec)
        if (
            report["contract_hash"] != expected_contract
            or report["source_metadata"]["spec_sha256"] != source_sha
            or any(
                report["input_diagnostics"][key] != diagnostics[key]
                for key in ("bars_hash", "signals_hash")
            )
        ):
            raise ValueError(
                f"Original attribution source/contract mismatch: {strategy}"
            )
        selected_cache = cache_with_score_threshold(cache, variant)
        assert_cache_matches_reference(
            selected_cache.evaluate(scores), report["actual"]["base"]
        )
        first_shift = circular_shift_scores(scores, groups, np.random.default_rng(seed))
        shifted_signals = signals.copy()
        shifted_signals[variant.score_column] = first_shift
        reference = _run_fold_segmented_backtest(
            bars, shifted_signals, gate=gate, contract=variant, cost_scenario=base
        )
        assert_cache_matches_reference(
            selected_cache.evaluate(first_shift), reference.summary
        )
        null = run_serial_null(
            selected_cache, scores, groups, permutations=spec.permutations, seed=seed
        )
        null.to_csv(output / f"{strategy}_serial_null.csv", index=False)
        actual = report["actual"]["base"]["compounded_return"]
        extreme = (null.compounded_return >= actual) | np.isclose(
            null.compounded_return, actual, rtol=1e-12, atol=1e-14
        )
        circular_p = (1 + int(extreme.sum())) / (1 + len(null))
        original_p = report["rank_destroyed_null"][
            "one_sided_p_value_actual_not_better_than_null"
        ]
        rows.append(
            {
                "strategy_id": strategy,
                "base_return": actual,
                "adverse_return": report["actual"]["adverse"]["compounded_return"],
                "actual_trade_count": report["actual"]["base"]["trade_count"],
                "serial_null_mean_trade_count": float(null.trade_count.mean()),
                "serial_null_median_return": float(null.compounded_return.median()),
                "serial_null_return_q025": float(
                    null.compounded_return.quantile(0.025)
                ),
                "serial_null_return_q975": float(
                    null.compounded_return.quantile(0.975)
                ),
                "serial_p_value": circular_p,
                "ordinary_p_value": original_p,
                "conservative_both_nulls_p_value": max(original_p, circular_p),
                "original_diagnostic_gate_passed": report["assessment"][
                    "diagnostic_gate_passed"
                ],
            }
        )
        source_reports[strategy] = file_sha256(report_path)
        print(
            f"{strategy}: serial p={circular_p:.4f}; null trades={null.trade_count.mean():.1f}",
            flush=True,
        )
    adjusted = _holm_adjust(
        {r["strategy_id"]: r["conservative_both_nulls_p_value"] for r in rows}
    )
    for row in rows:
        row["holm_both_nulls_p_value"] = adjusted[row["strategy_id"]]
        row["robust_diagnostic_gate_passed"] = (
            row["original_diagnostic_gate_passed"]
            and row["holm_both_nulls_p_value"] <= 0.05
        )
    pd.DataFrame(rows).to_csv(output / "full_oof_serial_control_suite.csv", index=False)
    (output / "full_oof_serial_control_suite.json").write_text(
        json.dumps(
            {
                "version": "full_oof_serial_control_v1",
                "spec_sha256": source_sha,
                "original_report_sha256": source_reports,
                "permutations": spec.permutations,
                "seed": seed,
                "rows": rows,
                "method": "independent_uniform_circular_shift_within_fold_calendar_month",
                "limitations": "Preserves cyclic score order with one wrap seam per group. Assumes local shift invariance; turnover is reported, not forced equal. Historical selection bias remains.",
                "multiplicity": "per_policy_max_of_two_null_p_values_then_Holm_across_four_policies",
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
