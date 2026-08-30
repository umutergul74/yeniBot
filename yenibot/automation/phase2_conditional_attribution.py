"""Run the fixed ATR-conditional score audit; no model fitting or promotion."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from yenibot.automation.phase2_ledger_risk_audit import PROBE_SHA256, artifact_path
from yenibot.phase2.conditional_attribution import (
    FrozenPayoffReplay,
    conditional_groups,
    conditional_tail_summary,
    draw_conditional_mapping,
    lag_one_within_fold,
    mapping_diagnostics,
)
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.economic_attribution import (
    EconomicAttributionSpec,
    _run_fold_segmented_backtest,
    _validate_inputs,
)
from yenibot.phase2.execution_cache import (
    assert_cache_matches_reference,
    build_fold_execution_cache,
)
from yenibot.phase2.full_oof import build_full_oof_inputs, file_sha256
from yenibot.phase2.readiness import load_phase2_gate

RATIOS = (1.10, 1.05)
METHODS = ("ordinary", "circular")
STATISTICS = ("compounded_return", "mean_net_return")


def _write(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _mean_or_none(values):
    mean = values.mean()
    return float(mean) if np.isfinite(mean) else None


@contextmanager
def output_lease(output):
    """OS lock releases after a crash; a stale file alone never means running."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".active.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save_trial_checkpoint(path, rows, *, ratio, method, seed):
    temporary = path.with_suffix(".json.tmp")
    _write(temporary, {"ratio": ratio, "method": method, "seed": seed, "rows": rows})
    temporary.replace(path)


def load_trial_checkpoint(path, *, ratio, method, seed):
    if not path.exists():
        return []
    saved = json.loads(path.read_text(encoding="utf-8"))
    if any(
        saved[k] != v
        for k, v in {"ratio": ratio, "method": method, "seed": seed}.items()
    ):
        raise ValueError("Checkpoint belongs to another conditional variant")
    rows = saved["rows"]
    if not rows:
        return rows
    f = pd.DataFrame(rows)
    count = len(f) // 2
    if (
        len(f) % 2
        or count > 500
        or f.duplicated(["trial", "cost"]).any()
        or sorted(f.trial.unique()) != list(range(count))
        or not f.groupby("trial")
        .cost.apply(lambda c: set(c) == {"base", "adverse"})
        .all()
    ):
        raise ValueError("Incomplete/duplicate conditional checkpoint trials")
    for key, value in (("ratio", ratio), ("method", method), ("seed", seed)):
        if not f[key].eq(value).all():
            raise ValueError("Checkpoint row identity differs")
    if not np.isfinite(f[list(STATISTICS)].to_numpy(dtype=float)).all():
        raise ValueError("Checkpoint has invalid statistics")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope-dir", "probe-dir", "report-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    with output_lease(Path(args.output_dir)):
        return execute(args)


def execute(args):
    probe, output = Path(args.probe_dir), Path(args.output_dir)
    root_path = probe / "prequential_probe_result.json"
    if file_sha256(root_path) != PROBE_SHA256:
        raise ValueError("Conditional audit requires the pinned closed probe")
    root = json.loads(root_path.read_text(encoding="utf-8"))
    for path, expected in root["artifact_sha256"].items():
        if file_sha256(artifact_path(probe, path)) != expected:
            raise ValueError(f"Changed frozen probe artifact: {path}")
    identity = {
        "version": "conditional_score_attribution_v1",
        "probe_sha256": PROBE_SHA256,
        "ratios": list(RATIOS),
        "methods": list(METHODS),
        "statistics": list(STATISTICS),
        "trials_per_variant": 500,
        "seed_base": 20260830,
        "protocol_sha256": file_sha256(
            Path("docs/conditional-score-attribution-v1.md")
        ),
        "implementation_sha256": {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                Path("yenibot/phase2/conditional_attribution.py"),
                Path("yenibot/phase2/execution_cache.py"),
                Path("yenibot/phase2/net_utility.py"),
                Path("yenibot/phase2/engine.py"),
                Path("yenibot/phase2/contracts.py"),
                Path("yenibot/phase2/costs.py"),
                Path("yenibot/phase2/economic_attribution.py"),
            )
        },
    }
    if (output / "conditional_attribution_result.json").exists():
        raise FileExistsError("Completed conditional audit is immutable")
    if any(p.name != ".active.lock" for p in output.iterdir()):
        saved = output / "source_identity.json"
        if not args.resume or not saved.exists():
            raise FileExistsError(
                "Inspect the checkpoint, then use --resume for this unfinished computation"
            )
        old = json.loads(saved.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Cannot resume a changed source/protocol/implementation")
    source = json.loads((probe / "source_manifest.json").read_text(encoding="utf-8"))
    bars, original, _ = build_full_oof_inputs(args.scope_dir, spec=source["spec"])
    bars = bars.loc[original.fold.isin(root["evaluation_fold_ids"])].reset_index(
        drop=True
    )
    signals = pd.read_csv(
        probe / "candidate_signals.csv",
        float_precision="round_trip",
        parse_dates=["decision_time"],
    )
    fits = json.loads((probe / "all_fold_fits.json").read_text(encoding="utf-8"))
    contract = replace(
        DEFAULT_PHASE2_CONTRACT,
        candidate_id=str(signals.candidate_id.iloc[0]),
        strategy_id="candidate",
        score_column="utility_score",
        threshold=0.5,
    )
    bars, signals, diagnostics = _validate_inputs(
        bars, signals, contract=contract, spec=EconomicAttributionSpec()
    )
    prior_attribution = json.loads(
        (probe / "attribution/phase2_economic_attribution.json").read_text(
            encoding="utf-8"
        )
    )
    if any(
        diagnostics[k] != prior_attribution["input_diagnostics"][k]
        for k in ("bars_hash", "signals_hash")
    ):
        raise ValueError("Conditional audit input/cohort changed")
    replay = FrozenPayoffReplay(signals, fits)
    raw = signals.frozen_score_percentile.to_numpy(dtype=float)
    predicted, actual_scores = replay.predict(raw)
    if not np.allclose(
        predicted, signals.predicted_adverse_net_return, rtol=1e-12, atol=1e-14
    ) or not np.array_equal(actual_scores >= 0.5, signals.utility_action):
        raise ValueError("Frozen predictions/actions do not reconstruct")
    groups = {ratio: conditional_groups(signals, ratio=ratio) for ratio in RATIOS}
    output.mkdir(parents=True, exist_ok=True)
    _write(
        output / "source_identity.json",
        {
            **identity,
            "input_diagnostics": diagnostics,
            "fit_operations": 0,
            "protocol_sha256": file_sha256(
                Path("docs/conditional-score-attribution-v1.md")
            ),
            "maximum_prediction_reconstruction_error": float(
                np.max(np.abs(predicted - signals.predicted_adverse_net_return))
            ),
        },
    )
    for ratio, (_, audit, table) in groups.items():
        table.to_csv(output / f"groups_{ratio}.csv", index=False)
        print(f"Conditional geometry {ratio}: {audit}", flush=True)
    gate = load_phase2_gate(args.report_dir)
    caches, actual = {}, {}
    cost_scenarios = {
        c.name: c for c in contract.cost_scenarios if c.name in ("base", "adverse")
    }
    for cost, scenario in cost_scenarios.items():
        print(f"Building reference execution cache: {cost}", flush=True)
        cache = build_fold_execution_cache(
            bars, signals, contract=contract, scenario=scenario
        )
        observed = cache.evaluate(actual_scores)
        reference = _run_fold_segmented_backtest(
            bars, signals, gate=gate, contract=contract, cost_scenario=scenario
        )
        assert_cache_matches_reference(observed, reference.summary)
        assert_cache_matches_reference(observed, root["actual"][cost])
        caches[cost], actual[cost] = cache, observed
    summaries, p_values = {}, []
    for gi, ratio in enumerate(RATIOS):
        for mi, method in enumerate(METHODS):
            seed = 20260830 + 1000 * gi + 100 * mi
            rng = np.random.default_rng(seed)
            checkpoint_path = output / f"checkpoint_{ratio}_{method}.json"
            rows = (
                load_trial_checkpoint(
                    checkpoint_path, ratio=ratio, method=method, seed=seed
                )
                if args.resume
                else []
            )
            completed = len(rows) // 2
            for trial in range(500):
                mapping = draw_conditional_mapping(
                    len(signals), groups[ratio][0], method=method, rng=rng
                )
                if trial < completed:
                    continue
                _, scores = replay.predict(raw[mapping])
                diag = mapping_diagnostics(signals, mapping, scores)
                if diag["maximum_donor_recipient_atr_ratio"] > ratio + 1e-12:
                    raise RuntimeError("Conditional donor crossed its ATR bound")
                for cost, cache in caches.items():
                    summary = cache.evaluate(scores)
                    if trial == 0:
                        perturbed = signals.copy()
                        perturbed["utility_score"] = scores
                        reference = _run_fold_segmented_backtest(
                            bars,
                            perturbed,
                            gate=gate,
                            contract=contract,
                            cost_scenario=cost_scenarios[cost],
                        )
                        assert_cache_matches_reference(summary, reference.summary)
                    rows.append(
                        {
                            "ratio": ratio,
                            "method": method,
                            "seed": seed,
                            "trial": trial,
                            "cost": cost,
                            **diag,
                            **summary,
                        }
                    )
                if (trial + 1) % 100 == 0:
                    # Completed deterministic trials are saved, never silently lost.
                    save_trial_checkpoint(
                        checkpoint_path, rows, ratio=ratio, method=method, seed=seed
                    )
                    pd.DataFrame(rows).to_csv(
                        output / f"trials_{ratio}_{method}.csv", index=False
                    )
                    _write(
                        output / "run_checkpoint.json",
                        {
                            "stage": "conditional_controls",
                            "ratio": ratio,
                            "method": method,
                            "completed_trials": trial + 1,
                            "seed": seed,
                            "fit_operations": 0,
                        },
                    )
                    print(
                        f"Conditional checkpoint {ratio}/{method}: {trial + 1}/500",
                        flush=True,
                    )
            frame = pd.DataFrame(rows)
            frame.to_csv(output / f"trials_{ratio}_{method}.csv", index=False)
            key = f"{ratio}_{method}"
            summaries[key] = {}
            for cost in caches:
                part = frame.loc[frame.cost.eq(cost)]
                values = {
                    stat: conditional_tail_summary(part, actual[cost], statistic=stat)
                    for stat in STATISTICS
                }
                p_values.extend(v["upper_tail_monte_carlo_p"] for v in values.values())
                summaries[key][cost] = {
                    "statistics": values,
                    "mean_trade_count": float(part.trade_count.mean()),
                    "trade_count_95_interval": part.trade_count.quantile(
                        [0.025, 0.975]
                    ).tolist(),
                    "mean_selected_rows": float(part.selected_signal_count.mean()),
                    "mean_raw_score_lag_one": _mean_or_none(part.raw_score_lag_one),
                    "mean_action_lag_one": _mean_or_none(part.action_lag_one),
                    "mean_changed_original_selected_score_fraction": float(
                        part.changed_original_selected_score_fraction.mean()
                    ),
                    "maximum_donor_recipient_atr_ratio": float(
                        part.maximum_donor_recipient_atr_ratio.max()
                    ),
                }
            _write(output / "completed_variant_summaries.json", summaries)
    covered = all(audit[1]["coverage_sufficient"] for audit in groups.values())
    passed = covered and max(p_values) <= 0.05
    result = {
        **identity,
        "status": "conditional_historical_diagnostic_passed_not_confirmation"
        if passed
        else "conditional_historical_diagnostic_failed_no_retune",
        "conditional_diagnostic_passed": passed,
        "selected_group_coverage_sufficient": covered,
        "conservative_maximum_monte_carlo_p": max(p_values),
        "test_combinations": len(p_values),
        "actual": actual,
        "group_audits": {str(k): v[1] for k, v in groups.items()},
        "conditional_controls": summaries,
        "actual_raw_score_lag_one": lag_one_within_fold(raw, signals.fold),
        "actual_action_lag_one": lag_one_within_fold(
            actual_scores >= 0.5, signals.fold
        ),
        "original_candidate_acceptance_passed": False,
        "original_failed_criteria": root["failed_criteria"],
        "fit_operations": 0,
        "new_candidate_selection_operations": 0,
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "limitations": [
            "Already-seen historically selected profile/policy",
            "Finite-width conditional exchangeability is approximate",
            "Circular order preserved only within irregular strata, not full hourly dependence",
            "Turnover and risk exposure not exactly equal",
            "Completed-trade compounding, terminal censored positions excluded",
            "Original fixed funding stress, not exact historical funding",
            "No genuinely unseen confirmation",
        ],
        "artifact_sha256": {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file()
            and p.name not in ("run_checkpoint.json", ".active.lock")
            and not p.name.endswith(".tmp")
        },
    }
    _write(output / "conditional_attribution_result.json", result)
    _write(
        output / "run_checkpoint.json",
        {
            "stage": "completed",
            "conditional_diagnostic_passed": passed,
            "original_candidate_acceptance_passed": False,
        },
    )
    print(
        f"Conditional audit complete: passed={passed}; maximum p={max(p_values)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
