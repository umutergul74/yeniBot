from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.phase2.economic_attribution import EconomicAttributionSpec
from yenibot.phase2.economic_attribution import run_economic_attribution
from yenibot.phase2.economic_attribution import write_economic_attribution
from yenibot.phase2.readiness import load_phase2_gate
from yenibot.phase2.variants import registered_phase2_strategy_variants


DEFAULT_INCUMBENT_STRATEGIES = (
    "baseline_fixed_atr_v1",
    "score_margin_07_time_stop_7bar_tp2_sl4_v1",
    "score_margin_04_atr_band_007_010_time_stop_6bar_tp15_sl4_v1",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_contract(input_manifest_path: Path) -> tuple[str, float, dict[str, Any]]:
    manifest = _read_json(input_manifest_path)
    candidate = manifest.get("candidate_manifest", {}) or {}
    result = manifest.get("result", {}) or {}
    candidate_id = str(
        candidate.get("candidate_id") or result.get("candidate_id") or ""
    ).strip()
    threshold_payload = candidate.get("threshold", {}) or {}
    threshold = threshold_payload.get("value", result.get("threshold"))
    if not candidate_id or threshold is None:
        raise ValueError(
            "Input manifest must identify the frozen candidate and threshold"
        )
    expected_hash = str(candidate.get("expected_manifest_hash") or "")
    actual_hash = str(candidate.get("manifest_hash") or "")
    if not expected_hash or not actual_hash or expected_hash != actual_hash:
        raise ValueError("Frozen candidate manifest hash is missing or mismatched")
    return candidate_id, float(threshold), manifest


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether frozen model ranking adds economic value relative "
            "to fold/month-matched rank-destroyed controls. Diagnostic only."
        )
    )
    parser.add_argument(
        "--report-dir", required=True, help="Directory for Phase 2 gate state"
    )
    parser.add_argument("--bars", required=True, help="Causal Phase 2 bars CSV")
    parser.add_argument(
        "--signals", required=True, help="Frozen test-split signals CSV"
    )
    parser.add_argument(
        "--input-manifest", required=True, help="Phase 2 input manifest JSON"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strategy-id",
        action="append",
        help="Registered strategy id; repeat for a bounded family",
    )
    parser.add_argument(
        "--incumbent-suite",
        action="store_true",
        help=(
            "Audit the immutable baseline plus the two previously documented "
            "seen-window incumbents. No winner is selected."
        ),
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report_dir = Path(args.report_dir)
    bars_path = Path(args.bars)
    signals_path = Path(args.signals)
    input_manifest_path = Path(args.input_manifest)
    output_dir = Path(args.output_dir)
    for required in (bars_path, signals_path, input_manifest_path):
        if not required.exists():
            raise FileNotFoundError(f"Attribution input not found: {required}")

    candidate_id, threshold, input_manifest = _candidate_contract(input_manifest_path)
    gate = load_phase2_gate(report_dir)
    bars = pd.read_csv(bars_path)
    signals = pd.read_csv(signals_path)
    registry = {
        item.contract.strategy_id: item
        for item in registered_phase2_strategy_variants(
            candidate_id=candidate_id,
            threshold=threshold,
        )
    }
    requested = (
        list(DEFAULT_INCUMBENT_STRATEGIES)
        if args.incumbent_suite
        else list(args.strategy_id or ["baseline_fixed_atr_v1"])
    )
    requested = list(dict.fromkeys(requested))
    unknown = sorted(set(requested).difference(registry))
    if unknown:
        raise ValueError(f"Unregistered strategy ids: {unknown}")

    source_metadata = {
        "bars_path": str(bars_path.resolve()),
        "bars_sha256": _sha256(bars_path),
        "signals_path": str(signals_path.resolve()),
        "signals_sha256": _sha256(signals_path),
        "input_manifest_path": str(input_manifest_path.resolve()),
        "input_manifest_sha256": _sha256(input_manifest_path),
        "source_run_id": (input_manifest.get("candidate_manifest", {}) or {}).get(
            "source_run_id"
        ),
        "candidate_manifest_hash": (
            input_manifest.get("candidate_manifest", {}) or {}
        ).get("manifest_hash"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    contract_hashes: dict[str, str] = {}
    for index, strategy_id in enumerate(requested):
        variant = registry[strategy_id]
        spec = EconomicAttributionSpec(
            permutations=int(args.permutations),
            seed=int(args.seed) + index * 100_000,
        )
        result = run_economic_attribution(
            bars,
            signals,
            gate=gate,
            contract=variant.contract,
            spec=spec,
            source_metadata={
                **source_metadata,
                "variant_status": variant.status,
                "variant_rationale": variant.rationale,
                "selection_allowed_on_current_test": False,
            },
        )
        write_economic_attribution(output_dir / strategy_id, result)
        report = result.report
        p_value = float(
            report["rank_destroyed_null"][
                "one_sided_p_value_actual_not_better_than_null"
            ]
        )
        p_values[strategy_id] = p_value
        contract_hashes[strategy_id] = str(report["contract_hash"])
        suite_rows.append(
            {
                "strategy_id": strategy_id,
                "variant_status": variant.status,
                "base_compounded_return": report["actual"]["base"]["compounded_return"],
                "adverse_compounded_return": report["actual"]["adverse"][
                    "compounded_return"
                ],
                "completed_trades": report["actual"]["base"]["trade_count"],
                "positive_fold_share": report["fold_diagnostics"][
                    "positive_fold_share"
                ],
                "score_forward_return_rankic": report["score_diagnostics"][
                    "score_forward_return_rankic"
                ],
                "null_median_return": report["rank_destroyed_null"]["return_median"],
                "actual_minus_null_median": report["rank_destroyed_null"][
                    "actual_minus_null_median"
                ],
                "raw_permutation_p_value": p_value,
                "diagnostic_gate_passed": report["assessment"][
                    "diagnostic_gate_passed"
                ],
                "failed_criteria": "|".join(report["assessment"]["failed_criteria"]),
            }
        )

    adjusted = _holm_adjust(p_values)
    for row in suite_rows:
        row["holm_adjusted_p_value"] = adjusted[row["strategy_id"]]
        row["familywise_beats_null_at_5pct"] = row["holm_adjusted_p_value"] <= 0.05
    summary = pd.DataFrame(suite_rows)
    summary.to_csv(output_dir / "phase2_economic_attribution_suite.csv", index=False)
    suite_report = {
        "version": "phase2_economic_attribution_suite_v1",
        "evidence_scope": "already_seen_phase2_test_window_retrospective",
        "candidate_id": candidate_id,
        "threshold": threshold,
        "strategies": requested,
        "contract_hashes": contract_hashes,
        "multiplicity_adjustment": "holm_familywise_across_requested_strategies",
        "source_metadata": source_metadata,
        "automatic_winner_selection_allowed": False,
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "rows": summary.to_dict(orient="records"),
    }
    (output_dir / "phase2_economic_attribution_suite.json").write_text(
        json.dumps(suite_report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(
        "phase2_economic_attribution_written "
        f"output_dir={output_dir} strategies={len(requested)} "
        f"permutations_per_strategy={args.permutations} promotion_allowed=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
