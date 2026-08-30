"""One locked past-OOF utility probe and same-cohort controls; never live."""

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
from yenibot.phase2.full_oof import build_full_oof_inputs, file_sha256
from yenibot.phase2.net_utility import paired_fold_block_intervals
from yenibot.phase2.prequential_utility import (
    build_prequential_signals,
    select_common_cohort,
)
from yenibot.phase2.readiness import load_phase2_gate

# Pin the complete pre-fit protocol; canonical JSON tolerates OS line endings.
SPEC_SHA256 = "91c673a964ecd2771b349bc8e522c6d87a3cd03634fd1fb914eb472a6291b568"


def _load_locked_spec(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    if _stable_hash(spec) != SPEC_SHA256:
        raise ValueError("Pinned research protocol mismatch")
    return spec


def _write(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _require_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Pinned artifact mismatch: {path}")


def _policy_contract(signals, name, *, q80=False):
    return replace(
        DEFAULT_PHASE2_CONTRACT,
        candidate_id=str(signals.candidate_id.iloc[0]),
        strategy_id=name,
        threshold=0.8 if q80 else 0.5,
        score_column="prob_long" if q80 else "utility_score",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "scope-dir",
        "report-dir",
        "q80-dir",
        "validation-probe-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{arg}", required=True)
    parser.add_argument("--spec", default="configs/prequential_oof_utility_v1.json")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    spec = _load_locked_spec(spec_path)
    source_path = Path(spec["source_spec"])
    source_spec = json.loads(source_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing prequential probe")
    gate = load_phase2_gate(args.report_dir)
    q80_path = Path(args.q80_dir) / "phase2_economic_attribution.json"
    legacy_path = Path(args.validation_probe_dir) / "net_utility_signals.csv"
    legacy_result = Path(args.validation_probe_dir) / "net_utility_probe_result.json"
    for path, key in (
        (q80_path, "q80_reference_report_sha256"),
        (legacy_path, "closed_validation_probe_signals_sha256"),
        (legacy_result, "closed_validation_probe_result_sha256"),
    ):
        _require_hash(path, spec[key])
    original = json.loads(q80_path.read_text(encoding="utf-8"))
    if original["source_metadata"]["spec_sha256"] != file_sha256(source_path):
        raise ValueError("Source specification differs from pinned q80 reference")
    bars, frozen, source = build_full_oof_inputs(args.scope_dir, spec=source_spec)
    attribution_spec = EconomicAttributionSpec(
        permutations=spec["permutations_per_null"],
        seed=spec["seed"],
        evidence_scope="already_seen_historical_past_oof_fitted_policy_retrospective",
    )
    q80_contract = _policy_contract(frozen, "validation_cdf_q80_fixed_atr_v1", q80=True)
    reference_spec = EconomicAttributionSpec(
        permutations=source_spec["permutations"],
        seed=source_spec["seed"],
        evidence_scope=source_spec["evidence_scope"],
    )
    expected_contract_hash = _stable_hash(
        {
            "attribution_version": ECONOMIC_ATTRIBUTION_VERSION,
            "fold_segmentation": "independent_walk_forward_folds",
            "strategy": asdict(q80_contract),
            "spec": asdict(reference_spec),
            "scenarios": [
                asdict(c)
                for c in q80_contract.cost_scenarios
                if c.name in ("base", "adverse")
            ],
        }
    )
    if original["contract_hash"] != expected_contract_hash:
        raise ValueError("Execution/cost contract differs from locked q80 reference")
    bars, frozen, diagnostics = _validate_inputs(
        bars, frozen, contract=q80_contract, spec=attribution_spec
    )
    if any(
        original["input_diagnostics"][key] != diagnostics[key]
        for key in ("bars_hash", "signals_hash")
    ):
        raise ValueError("Full historical input differs from pinned q80 input")
    folds = list(range(spec["evaluation_fold_first"], spec["evaluation_fold_last"] + 1))
    keys = frozen.loc[frozen.fold.isin(folds), ["fold", "decision_time"]].copy()
    legacy = pd.read_csv(legacy_path, float_precision="round_trip")
    legacy = select_common_cohort(legacy, keys, folds=folds)
    q80 = select_common_cohort(frozen, keys, folds=folds)
    evaluation_bars = bars.loc[
        bars.bar_close_time.isin(keys.decision_time)
    ].reset_index(drop=True)
    # Identity and execution contract checks for ALL controls precede any fitting.
    for name, control in (("q80", q80), ("archived_validation", legacy)):
        _validate_inputs(
            evaluation_bars,
            control,
            contract=_policy_contract(control, name, q80=name == "q80"),
            spec=attribution_spec,
        )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "stage": "read_only_preflight_passed",
                    "fit_operations": 0,
                    "evaluation_folds": folds,
                    "evaluation_rows": len(keys),
                    "all_source_and_reference_hashes_verified": True,
                    "all_controls_have_same_cohort": True,
                }
            ),
            flush=True,
        )
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "folds").mkdir()
    source.update(
        {
            "utility_spec": spec,
            "utility_spec_sha256": file_sha256(spec_path),
            "source_spec_sha256": file_sha256(source_path),
            "full_input_diagnostics": diagnostics,
            "input_kind": "past_oof_fitted_payoff_policy_not_frozen_candidate",
            "score_semantics": "monotone_predicted_adverse_net_payoff_not_probability",
            "earlier_oof_test_outcomes_used_in_fit": True,
            "current_or_future_fold_outcomes_used_in_fit": False,
            "fit_scope": "36_candidate_and_36_atr_only_payoff_fits_no_tcn_gru_refit",
            "evaluation_fold_ids": folds,
            "evaluation_rows": len(keys),
            "evaluation_start": keys.decision_time.min().isoformat(),
            "evaluation_end": keys.decision_time.max().isoformat(),
            "implementation_sha256": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    Path("yenibot/phase2/prequential_utility.py"),
                    Path("yenibot/phase2/net_utility.py"),
                    Path("yenibot/phase2/full_oof.py"),
                    Path("yenibot/phase2/economic_attribution.py"),
                    Path("yenibot/phase2/serial_attribution.py"),
                    Path("yenibot/phase2/execution_cache.py"),
                    Path("yenibot/phase2/engine.py"),
                    Path("yenibot/phase2/contracts.py"),
                    Path("yenibot/phase2/costs.py"),
                    Path("yenibot/phase2/market_contract.py"),
                )
            },
        }
    )
    _write(output / "source_manifest.json", source)

    def checkpoint(stage, **details):
        _write(
            output / "run_checkpoint.json",
            {"stage": stage, "pid": os.getpid(), **details},
        )
        print(f"Checkpoint {stage}: {details}", flush=True)

    def on_fold(record):
        fold = record["fold"]
        _write(output / "folds" / f"fold_{fold:02d}_fits.json", record)
        if fold == folds[0] or (fold + 1) % 10 == 0 or fold == folds[-1]:
            checkpoint(
                "past_oof_fit",
                last_completed_fold=fold,
                history_rows=record["eligible_history_rows"],
            )

    checkpoint(
        "input_contract_verified",
        evaluation_rows=len(keys),
        evaluation_folds=len(folds),
    )
    candidate, atr_only, fits, targets = build_prequential_signals(
        bars, frozen, spec=spec, on_fold=on_fold
    )
    _write(output / "all_fold_fits.json", fits)
    targets.to_csv(output / "oof_opportunity_targets.csv", index=False)
    policies = {
        "candidate": candidate,
        "atr_only": atr_only,
        "q80": q80,
        "archived_validation": legacy,
    }
    contracts = {}
    for name, signals in policies.items():
        policies[name] = select_common_cohort(signals, keys, folds=folds)
        contracts[name] = _policy_contract(policies[name], name, q80=name == "q80")
        _validate_inputs(
            evaluation_bars,
            policies[name],
            contract=contracts[name],
            spec=attribution_spec,
        )
        policies[name].to_csv(output / f"{name}_signals.csv", index=False)
    source.update(
        candidate_id=spec["candidate_id"],
        candidate_fit_count=len(fits),
        total_payoff_fit_count=2 * len(fits),
    )
    _write(output / "source_manifest.json", source)
    checkpoint(
        "candidate_economic_evaluation",
        selected_rows=int(candidate.utility_action.sum()),
        fit_count=2 * len(fits),
    )
    result = run_economic_attribution(
        evaluation_bars,
        candidate,
        gate=gate,
        contract=contracts["candidate"],
        spec=attribution_spec,
        source_metadata=source,
        upstream_fit_operations=len(fits),
        include_serial_control=True,
    )
    write_economic_attribution(output / "attribution", result)
    summaries, paired = {}, {}
    for cost in spec["required_paired_costs"]:
        scenario = next(
            c for c in DEFAULT_PHASE2_CONTRACT.cost_scenarios if c.name == cost
        )
        ledger = result.trade_ledger
        trades = (
            ledger.loc[ledger.cost_scenario.eq(cost)] if not ledger.empty else ledger
        )
        candidate_folds = _fold_outcomes(
            Phase2BacktestResult(trades, pd.DataFrame(), {}, {}),
            candidate,
            contract=contracts["candidate"],
        )
        candidate_folds.to_csv(output / f"candidate_{cost}_folds.csv", index=False)
        for name in ("atr_only", "q80", "archived_validation"):
            reference = _run_fold_segmented_backtest(
                evaluation_bars,
                policies[name],
                gate=gate,
                contract=contracts[name],
                cost_scenario=scenario,
            )
            reference.trades.to_csv(output / f"{name}_{cost}_trades.csv", index=False)
            ref_folds = _fold_outcomes(
                reference, policies[name], contract=contracts[name]
            )
            ref_folds.to_csv(output / f"{name}_{cost}_folds.csv", index=False)
            summaries.setdefault(name, {})[cost] = reference.summary
            if name in spec["required_paired_controls"]:
                intervals, comparison = paired_fold_block_intervals(
                    candidate_folds,
                    ref_folds,
                    block_lengths=spec["bootstrap_block_folds"],
                    replicates=spec["bootstrap_replicates"],
                    seed=spec["seed"] + 9911,
                )
                comparison.to_csv(output / f"paired_{name}_{cost}.csv", index=False)
                paired[f"{name}_{cost}"] = intervals
            checkpoint("same_cohort_reference_evaluated", policy=name, cost=cost)
    criteria = result.report["assessment"]["criteria"].copy()
    for name in spec["required_paired_controls"]:
        criteria[f"paired_{name}_improvement_all_costs_and_blocks"] = all(
            item["lower_bound_positive"]
            for cost in spec["required_paired_costs"]
            for item in paired[f"{name}_{cost}"]
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
        "actual": result.report["actual"],
        "same_cohort_controls": summaries,
        "evaluation_fold_ids": folds,
        "evaluation_rows": len(keys),
        "warmup_folds": spec["warmup_folds"],
        "candidate_fit_count": len(fits),
        "atr_only_fit_count": len(fits),
        "total_payoff_fit_count": 2 * len(fits),
        "tcn_gru_refits": 0,
        "archived_policy_refits": 0,
        "earlier_oof_test_outcomes_used_in_fit": True,
        "current_or_future_fold_outcomes_used_in_fit": False,
        "ordinary_null": result.report["rank_destroyed_null"],
        "serial_null": result.report["serial_shift_null"],
        "paired_fold_bootstrap": paired,
        "whole_policy_null_not_isolated_tcn_contribution": True,
        "no_trade_return": 0.0,
        "return_semantics": "completed_trade_compounding_across_independent_folds_not_annualized_or_live_portfolio_return",
        "limitations": "Already-seen selected historical profile; 36 dependent folds; overlapping training opportunities; approximate moving-block uncertainty; terminal censored positions excluded; trade-close drawdown; fixed cost stress not actual historical funding.",
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "artifact_sha256": {
            str(p.relative_to(output)): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "run_checkpoint.json"
        },
    }
    _write(output / "prequential_probe_result.json", report)
    checkpoint(
        "evaluation_completed",
        economic_gate_passed=passed,
        failed_criteria=report["failed_criteria"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
