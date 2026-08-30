"""Run the single specified validation net-utility probe, never live trading."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.economic_attribution import (
    ECONOMIC_ATTRIBUTION_VERSION,
    EconomicAttributionSpec,
    _fold_outcomes,
    _run_fold_segmented_backtest,
    _stable_hash,
    _validate_inputs,
    run_economic_attribution,
    write_economic_attribution,
)
from yenibot.phase2.engine import Phase2BacktestResult
from yenibot.phase2.execution_cache import assert_cache_matches_reference
from yenibot.phase2.full_oof import (
    build_full_oof_inputs,
    file_sha256,
    load_pinned_full_oof_frame,
)
from yenibot.phase2.net_utility import (
    build_net_utility_signals,
    paired_fold_block_intervals,
)
from yenibot.phase2.readiness import load_phase2_gate


def _write(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--spec", default="configs/validation_net_utility_hurdle_v1.json"
    )
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (
        spec["version"] != "validation_net_utility_hurdle_v1"
        or spec["ridge_alpha"] != 10
        or spec["minimum_fit_rows"] != 200
        or spec["utility_cutoff"] != 0
        or spec["utility_cutoff_comparison"] != "strictly_greater"
        or spec["paired_confidence_level"] != 0.95
        or spec["paired_cost_scenarios"] != ["base", "adverse"]
        or spec["paired_fold_block_lengths"] != [3, 6]
        or spec["benchmark"] != "validation_cdf_q80_fixed_atr_v1"
        or spec["features"]
        != [
            "validation_score_percentile",
            "decision_atr_close_fraction",
            "score_atr_product",
        ]
        or spec["target"] != "adverse_cost_single_opportunity_net_return"
        or spec["permutations_per_null"] != 500
        or spec["paired_bootstrap_replicates"] != 5000
        or spec["seed"] != 20260830
    ):
        raise ValueError("Refusing a retuned v1 net-utility contract")
    source_spec_path = Path(spec["historical_source_spec"])
    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a net-utility probe")
    bars, reference_signals, source = build_full_oof_inputs(
        args.scope_dir, spec=source_spec
    )
    frame, _ = load_pinned_full_oof_frame(args.scope_dir, spec=source_spec)
    reference_path = (
        Path(args.reference_dir)
        / spec["benchmark"]
        / "phase2_economic_attribution.json"
    )
    original = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        candidate_id=source["candidate_id"],
        threshold=0.8,
        strategy_id=spec["benchmark"],
    )
    attribution_spec = EconomicAttributionSpec(
        permutations=spec["permutations_per_null"],
        seed=spec["seed"],
        evidence_scope="already_seen_full_cv_validation_fitted_policy_retrospective",
    )
    _, _, reference_diagnostics = _validate_inputs(
        bars, reference_signals, contract=reference_contract, spec=attribution_spec
    )
    original_spec = EconomicAttributionSpec(
        permutations=source_spec["permutations"],
        seed=source_spec["seed"],
        evidence_scope=source_spec["evidence_scope"],
    )
    expected_reference_hash = _stable_hash(
        {
            "attribution_version": ECONOMIC_ATTRIBUTION_VERSION,
            "fold_segmentation": "independent_walk_forward_folds",
            "strategy": asdict(reference_contract),
            "spec": asdict(original_spec),
            "scenarios": [
                asdict(c)
                for c in reference_contract.cost_scenarios
                if c.name in ("base", "adverse")
            ],
        }
    )
    if (
        original["source_metadata"]["spec_sha256"] != file_sha256(source_spec_path)
        or original["contract_hash"] != expected_reference_hash
        or original["strategy_id"] != spec["benchmark"]
        or any(
            original["input_diagnostics"][key] != reference_diagnostics[key]
            for key in ("bars_hash", "signals_hash")
        )
    ):
        raise ValueError("q80 reference source/identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "folds").mkdir()
    source.update(
        {
            "utility_spec": spec,
            "utility_spec_sha256": file_sha256(spec_path),
            "source_spec_sha256": file_sha256(source_spec_path),
            "reference_report_sha256": file_sha256(reference_path),
            "input_kind": "validation_fitted_payoff_model_not_frozen_candidate",
            "score_semantics": "monotone_predicted_adverse_net_payoff_not_probability",
            "implementation_sha256": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    Path("yenibot/phase2/net_utility.py"),
                    Path("yenibot/phase2/economic_attribution.py"),
                    Path("yenibot/phase2/engine.py"),
                    Path("yenibot/phase2/execution_cache.py"),
                    Path("yenibot/phase2/costs.py"),
                )
            },
        }
    )
    _write(output / "source_manifest.json", source)

    def on_fold(record, targets):
        fold = record["fold"]
        _write(output / "folds" / f"fold_{fold:02d}_fit.json", record)
        targets.to_csv(
            output / "folds" / f"fold_{fold:02d}_training_opportunities.csv",
            index=False,
        )
        _write(
            output / "run_checkpoint.json",
            {"stage": "fitting", "last_completed_fold": fold, "pid": os.getpid()},
        )
        if fold == 0 or (fold + 1) % 10 == 0 or fold == source_spec["fold_count"] - 1:
            print(
                f"Fit checkpoint fold={fold}; eligible={record['eligible_fit_rows']}",
                flush=True,
            )

    signals, fits, targets = build_net_utility_signals(
        frame, source_spec=source_spec, utility_spec=spec, on_fold=on_fold
    )
    fit_count = sum(record["fit_performed"] for record in fits)
    _write(output / "all_fold_fits.json", fits)
    signals.to_csv(output / "net_utility_signals.csv", index=False)
    targets.to_csv(output / "validation_opportunity_targets.csv", index=False)
    source.update(
        candidate_id=str(signals.candidate_id.iloc[0]),
        upstream_fit_operations=fit_count,
    )
    _write(output / "source_manifest.json", source)
    _write(
        output / "run_checkpoint.json",
        {"stage": "economic_evaluation", "fit_count": fit_count, "pid": os.getpid()},
    )
    print(
        f"Evaluating single candidate: fits={fit_count}, selected_rows={int(signals.utility_action.sum())}",
        flush=True,
    )
    gate = load_phase2_gate(args.report_dir)
    contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        candidate_id=source["candidate_id"],
        strategy_id=spec["version"],
        threshold=0.5,
        score_column="utility_score",
    )
    result = run_economic_attribution(
        bars,
        signals,
        gate=gate,
        contract=contract,
        spec=attribution_spec,
        source_metadata=source,
        upstream_fit_operations=fit_count,
        include_serial_control=True,
    )
    write_economic_attribution(output / "attribution", result)
    paired = {}
    q80_reference = {}
    for scenario_name in spec["paired_cost_scenarios"]:
        scenario = next(c for c in contract.cost_scenarios if c.name == scenario_name)
        reference = _run_fold_segmented_backtest(
            bars,
            reference_signals,
            gate=gate,
            contract=reference_contract,
            cost_scenario=scenario,
        )
        keys = ("compounded_return", "trade_count", "mean_net_return", "max_drawdown")
        assert_cache_matches_reference(
            {k: reference.summary[k] for k in keys}, original["actual"][scenario_name]
        )
        q80_reference[scenario_name] = {k: reference.summary[k] for k in keys}
        ref_folds = _fold_outcomes(
            reference, reference_signals, contract=reference_contract
        )
        ledger = result.trade_ledger
        candidate_trades = (
            ledger.loc[ledger.cost_scenario.eq(scenario_name)]
            if not ledger.empty
            else ledger
        )
        candidate_result = Phase2BacktestResult(
            candidate_trades, pd.DataFrame(), {}, {}
        )
        candidate_folds = _fold_outcomes(candidate_result, signals, contract=contract)
        intervals, comparison = paired_fold_block_intervals(
            candidate_folds,
            ref_folds,
            block_lengths=spec["paired_fold_block_lengths"],
            replicates=spec["paired_bootstrap_replicates"],
            seed=spec["seed"] + 9911,
        )
        comparison.to_csv(
            output / f"paired_fold_comparison_{scenario_name}.csv", index=False
        )
        paired[scenario_name] = intervals
    criteria = result.report["assessment"]["criteria"].copy()
    criteria["paired_q80_improvement_all_temporal_intervals"] = all(
        interval["lower_bound_positive"]
        for intervals in paired.values()
        for interval in intervals
    )
    passed = all(criteria.values())
    report = {
        "version": spec["version"],
        "status": "historical_probe_requires_clean_confirmation"
        if passed
        else "historical_probe_failed_family_closed",
        "economic_gate_passed": passed,
        "criteria": criteria,
        "failed_criteria": [key for key, value in criteria.items() if not value],
        "source_manifest_sha256": file_sha256(output / "source_manifest.json"),
        "attribution_sha256": file_sha256(
            output / "attribution" / "phase2_economic_attribution.json"
        ),
        "fit_count": fit_count,
        "tcn_gru_refits": 0,
        "test_labels_used_in_fits": False,
        "actual": result.report["actual"],
        "reference_q80": q80_reference,
        "no_trade_return": 0.0,
        "ordinary_null": result.report["rank_destroyed_null"],
        "serial_null": result.report["serial_shift_null"],
        "paired_fold_bootstrap": paired,
        "bootstrap_method": "paired_fold_return_delta_overlapping_moving_blocks_3_and_6_folds_95pct_percentile_intervals",
        "bootstrap_limitations": "Only 38 dependent historical folds; approximate temporal uncertainty, not independent confirmation. No overlapping hourly-label pseudo-replication.",
        "promotion_allowed": False,
        "live_trading_allowed": False,
    }
    _write(output / "net_utility_probe_result.json", report)
    _write(
        output / "run_checkpoint.json",
        {
            "stage": "evaluation_completed",
            "economic_gate_passed": passed,
            "fit_count": fit_count,
        },
    )
    print(
        f"Net-utility probe complete: passed={passed}; failed={report['failed_criteria']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
