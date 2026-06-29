from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.phase2.confirmation import filter_clean_confirmation_inputs
from yenibot.phase2.confirmation import forward_lock_snapshot
from yenibot.phase2.confirmation import load_phase2_forward_lock
from yenibot.phase2.confirmation import locked_risk_policy
from yenibot.phase2.confirmation import locked_strategy_contracts
from yenibot.phase2.confirmation import validate_forward_input_manifest
from yenibot.phase2.adapter import phase2_inputs_from_predictions
from yenibot.phase2.engine import run_long_only_backtest
from yenibot.phase2.forensics import phase2_trade_forensics
from yenibot.phase2.forensics import write_phase2_forensics
from yenibot.phase2.readiness import Phase2Gate
from yenibot.phase2.readiness import load_phase2_gate
from yenibot.phase2.reporting import write_phase2_sandbox_report


FORWARD_RUNNER_VERSION = "phase2_forward_runner_v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename!r}")
        archive.extractall(output_dir)


def _resolve_phase2_dir(root: Path) -> Path:
    direct = root / "phase2_sandbox"
    if direct.is_dir():
        return direct
    matches = [path for path in root.rglob("phase2_sandbox") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one phase2_sandbox directory; "
            f"found {len(matches)}."
        )
    return matches[0]


def _find_single_file(root: Path, filename: str) -> Path | None:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected at most one {filename!r}; found {len(matches)}."
        )
    return matches[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _manifest_wrapper(root: Path) -> tuple[dict[str, Any], Path]:
    phase2_manifest = _find_single_file(root, "phase2_input_manifest.json")
    if phase2_manifest is not None:
        return (
            json.loads(phase2_manifest.read_text(encoding="utf-8")),
            phase2_manifest,
        )
    frozen_path = _find_single_file(root, "frozen_candidate_manifest.json")
    if frozen_path is None:
        raise FileNotFoundError(
            "Input is missing phase2_input_manifest.json and "
            "frozen_candidate_manifest.json."
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    threshold = (frozen.get("threshold", {}) or {}).get("value")
    return (
        {
            "candidate_manifest": frozen,
            "result": {
                "candidate_id": frozen.get("candidate_id"),
                "threshold": threshold,
            },
        },
        frozen_path,
    )


def _source_inputs(
    root: Path,
    *,
    lock: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frozen = lock["frozen_model"]
    future_path = _find_single_file(root, "future_oos_predictions.parquet")
    if future_path is None:
        future_path = _find_single_file(root, "future_oos_predictions.csv")
    if future_path is not None:
        predictions = _read_table(future_path)
        if "candidate_id" in predictions.columns:
            predictions = predictions.loc[
                predictions["candidate_id"].astype(str)
                == str(frozen["candidate_id"])
            ].copy()
        if predictions.empty:
            return (
                pd.DataFrame(
                    columns=[
                        "bar_close_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "atr_14",
                    ]
                ),
                pd.DataFrame(
                    columns=[
                        "decision_time",
                        "prob_long",
                        "candidate_id",
                        "threshold",
                    ]
                ),
                {
                    "source_mode": "future_oos_predictions_empty",
                    "source_path": str(future_path),
                },
            )
        required_market = {"open", "high", "low", "close", "atr_14"}
        missing = sorted(required_market.difference(predictions.columns))
        if missing:
            raise ValueError(
                "Future-OOS predictions do not contain Phase 2 market columns "
                f"{missing}. Re-run Notebook 05 with the current repository "
                "before clean-forward evaluation."
            )
        bars, signals, stats = phase2_inputs_from_predictions(
            predictions,
            candidate_id=str(frozen["candidate_id"]),
            threshold=float(frozen["threshold"]),
            split="all",
        )
        return (
            bars,
            signals,
            {
                "source_mode": "future_oos_predictions",
                "source_path": str(future_path),
                **stats,
            },
        )

    phase2_dir = _resolve_phase2_dir(root)
    bars_path = phase2_dir / "phase2_bars.csv"
    signals_path = phase2_dir / "phase2_signals.csv"
    for path in (bars_path, signals_path):
        if not path.exists():
            raise FileNotFoundError(f"Input is missing required file: {path}")
    return (
        pd.read_csv(bars_path),
        pd.read_csv(signals_path),
        {
            "source_mode": "phase2_sandbox_inputs",
            "bars_path": str(bars_path),
            "signals_path": str(signals_path),
        },
    )


def _bundle_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pending_gate(report_dir: Path) -> Phase2Gate:
    return Phase2Gate(
        report_dir=report_dir,
        ready_for_phase2=False,
        report_consistency_passed=True,
        future_oos_evaluation_completed=False,
        future_oos_candidate_passed=False,
        promotion_allowed=False,
        blockers=("future_unseen_oos_not_ready",),
        advisories=("clean_confirmation_runner_is_non_promotable",),
        next_action="collect_post_anchor_data_without_refit",
    )


def _risk_scaled_forensics_frame(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    if frame.empty or "portfolio_return" not in frame.columns:
        return frame
    fraction = pd.to_numeric(
        frame["position_notional_fraction"],
        errors="coerce",
    ).fillna(0.0)
    frame["gross_return"] = pd.to_numeric(
        frame["gross_return"],
        errors="coerce",
    ).fillna(0.0) * fraction
    frame["total_cost_return"] = pd.to_numeric(
        frame["total_cost_return"],
        errors="coerce",
    ).fillna(0.0) * fraction
    frame["net_return"] = pd.to_numeric(
        frame["portfolio_return"],
        errors="coerce",
    ).fillna(0.0)
    return frame


def _gate_checks(
    *,
    lock: dict[str, Any],
    boundary: dict[str, Any],
    base_result: Any,
    adverse_result: Any,
    base_forensics: dict[str, Any],
) -> dict[str, Any]:
    window = lock["confirmation_window"]
    gates = lock["success_gates"]
    summary = base_result.summary
    adverse_summary = adverse_result.summary
    forensic_summary = base_forensics["summary"]
    bootstrap = base_forensics["bootstrap"]
    checks = {
        "minimum_coverage_days": (
            float(boundary["coverage_days"])
            >= float(window["minimum_coverage_days"])
        ),
        "minimum_trade_count": (
            int(summary["trade_count"]) >= int(window["minimum_trade_count"])
        ),
        "positive_base_return": (
            float(summary["compounded_return"])
            > float(gates["base_compounded_return_min_exclusive"])
        ),
        "positive_adverse_return": (
            float(adverse_summary["compounded_return"])
            > float(gates["adverse_compounded_return_min_exclusive"])
        ),
        "base_profit_factor": (
            summary.get("profit_factor") is not None
            and float(summary["profit_factor"])
            >= float(gates["base_profit_factor_min"])
        ),
        "bootstrap_probability_positive": (
            bootstrap.get("probability_compounded_return_positive") is not None
            and float(bootstrap["probability_compounded_return_positive"])
            >= float(gates["bootstrap_probability_positive_min"])
        ),
        "maximum_drawdown": (
            float(summary["max_drawdown"])
            >= float(gates["max_drawdown_floor"])
        ),
        "best_month_removed_return": (
            float(
                forensic_summary["best_month_removed_compounded_return"]
            )
            >= float(gates["best_month_removed_return_min"])
        ),
    }
    evidence_ready = bool(
        checks["minimum_coverage_days"] and checks["minimum_trade_count"]
    )
    metrics_passed = bool(evidence_ready and all(checks.values()))
    if not evidence_ready:
        status = "collecting_clean_evidence"
    elif metrics_passed:
        status = "locked_metrics_passed_promotion_still_fail_closed"
    else:
        status = "locked_clean_confirmation_failed"
    return {
        "status": status,
        "evidence_ready": evidence_ready,
        "metrics_passed": metrics_passed,
        "promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _decision_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Clean Forward Confirmation",
        "",
        f"Runner: `{report['runner_version']}`",
        f"Status: `{report['status']}`",
        f"Lock hash: `{report['lock_hash']}`",
        "",
        "> Fail-closed: this runner cannot refit, reselect, or automatically promote.",
        "",
        "## Clean boundary",
        "",
        f"- Cutoff (exclusive): `{report['boundary']['cutoff_exclusive']}`",
        f"- Accepted signals: `{report['boundary']['accepted_signal_count']}`",
        f"- Coverage days: `{report['boundary']['coverage_days']}`",
        "",
        "## Candidate results",
        "",
    ]
    if not report.get("candidate_results"):
        lines.append("- No post-anchor signals are available yet.")
    for item in report.get("candidate_results", []):
        lines.extend(
            [
                f"### {item['role']}",
                "",
                f"- Strategy: `{item['strategy_id']}`",
                f"- Status: `{item['gate']['status']}`",
                f"- Trades: `{item['base']['trade_count']}`",
                f"- Base return: `{item['base']['compounded_return']}`",
                f"- Adverse return: `{item['adverse']['compounded_return']}`",
                f"- Failed checks: `{item['gate']['failed_checks']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable Phase 2 clean-forward confirmation from a bundle. "
            "Pre-anchor rows are excluded and automatic promotion is impossible."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", help="Phase 1/Phase 2 bundle zip.")
    source.add_argument(
        "--input-dir",
        help="Extracted Phase 1 report or bundle directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Forward-run output directory.")
    parser.add_argument(
        "--report-dir",
        help="Optional Phase 1 report directory used for the fail-closed gate.",
    )
    parser.add_argument(
        "--lock-path",
        help="Optional explicit forward lock manifest; defaults to the committed lock.",
    )
    parser.add_argument(
        "--candidate-role",
        default="all",
        choices=["all", "primary_balanced", "challenger_return"],
    )
    parser.add_argument("--keep-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle) if args.bundle else None
    if bundle is not None:
        if not bundle.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle}")
        source_root = output_dir / "bundle_extract"
        _safe_extract_zip(bundle, source_root)
    else:
        source_root = Path(args.input_dir)
        if not source_root.exists():
            raise FileNotFoundError(f"Input directory not found: {source_root}")

    lock = load_phase2_forward_lock(
        args.lock_path
        if args.lock_path
        else Path(__file__).parents[1] / "phase2" / "forward_lock.json"
    )
    input_manifest, manifest_path = _manifest_wrapper(source_root)
    manifest_audit = validate_forward_input_manifest(lock, input_manifest)
    source_bars, source_signals, source_audit = _source_inputs(
        source_root,
        lock=lock,
    )
    inputs = filter_clean_confirmation_inputs(
        source_bars,
        source_signals,
        lock=lock,
    )
    boundary = {
        **inputs.boundary_report,
        "input_manifest_audit": manifest_audit,
        "input_manifest_path": str(manifest_path),
        "source_audit": source_audit,
        "source_bundle_sha256": (
            _bundle_sha256(bundle) if bundle is not None else None
        ),
    }
    _write_json(output_dir / "phase2_forward_lock_snapshot.json", forward_lock_snapshot(lock))
    _write_json(output_dir / "phase2_forward_boundary_report.json", boundary)
    inputs.bars.to_csv(output_dir / "phase2_forward_bars.csv", index=False)
    inputs.signals.to_csv(output_dir / "phase2_forward_signals.csv", index=False)

    report: dict[str, Any] = {
        "runner_version": FORWARD_RUNNER_VERSION,
        "generated_at_utc": (
            datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        ),
        "status": "waiting_for_post_anchor_data",
        "lock_hash": lock["lock_hash"],
        "source_bundle": str(bundle) if bundle is not None else None,
        "source_input_dir": str(source_root) if bundle is None else None,
        "boundary": boundary,
        "risk_policy": asdict(locked_risk_policy(lock)),
        "candidate_results": [],
        "fit_operations_performed": 0,
        "selection_operations_performed": 0,
        "promotion_allowed": False,
    }
    if not inputs.signals.empty:
        gate = (
            load_phase2_gate(args.report_dir)
            if args.report_dir
            else _pending_gate(output_dir)
        )
        contracts = locked_strategy_contracts(lock)
        roles = (
            list(contracts)
            if args.candidate_role == "all"
            else [args.candidate_role]
        )
        risk_policy = locked_risk_policy(lock)
        for role in roles:
            contract = contracts[role]
            results: dict[str, Any] = {}
            diagnostics: dict[str, Any] = {}
            for scenario in (contract.cost_scenarios[1], contract.cost_scenarios[2]):
                result = run_long_only_backtest(
                    inputs.bars,
                    inputs.signals,
                    gate=gate,
                    contract=contract,
                    cost_scenario=scenario,
                    risk_policy=risk_policy,
                    mode="sandbox",
                )
                scenario_dir = output_dir / "candidates" / role / scenario.name
                write_phase2_sandbox_report(scenario_dir, result)
                scaled_trades = _risk_scaled_forensics_frame(result.trades)
                forensic = phase2_trade_forensics(
                    scaled_trades,
                    inputs.signals,
                    execution_summary=result.metadata["execution_diagnostics"],
                )
                write_phase2_forensics(scenario_dir, forensic)
                results[scenario.name] = result
                diagnostics[scenario.name] = forensic
            gate_result = _gate_checks(
                lock=lock,
                boundary=boundary,
                base_result=results["base"],
                adverse_result=results["adverse"],
                base_forensics=diagnostics["base"],
            )
            report["candidate_results"].append(
                {
                    "role": role,
                    "strategy_id": contract.strategy_id,
                    "base": results["base"].summary,
                    "adverse": results["adverse"].summary,
                    "base_forensics": {
                        "summary": diagnostics["base"]["summary"],
                        "bootstrap": diagnostics["base"]["bootstrap"],
                    },
                    "gate": gate_result,
                }
            )
        primary = next(
            (
                item
                for item in report["candidate_results"]
                if item["role"] == "primary_balanced"
            ),
            report["candidate_results"][0],
        )
        report["status"] = primary["gate"]["status"]

    _write_json(output_dir / "phase2_forward_decision.json", report)
    (output_dir / "phase2_forward_decision.md").write_text(
        _decision_markdown(report),
        encoding="utf-8",
    )
    print(
        "phase2_forward_written "
        f"status={report['status']} "
        f"accepted_signals={boundary['accepted_signal_count']} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
