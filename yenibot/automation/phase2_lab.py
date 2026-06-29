from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.automation.phase2_sandbox import main as phase2_sandbox_main
from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT


LOCAL_LAB_VERSION = "phase2_local_lab_v1"
DEFAULT_BUNDLE_GLOB = "phase2_latest_sandbox_bundle*.zip"
BASELINE_STRATEGY_ID = DEFAULT_PHASE2_CONTRACT.strategy_id


@dataclass(frozen=True)
class LocalLabPaths:
    run_dir: Path
    extracted_dir: Path
    source_phase2_dir: Path
    gate_dir: Path
    output_phase2_dir: Path
    decision_json_path: Path
    decision_md_path: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _find_latest_bundle(downloads_dir: Path) -> Path:
    candidates = sorted(
        downloads_dir.glob(DEFAULT_BUNDLE_GLOB),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {DEFAULT_BUNDLE_GLOB!r} files found in {downloads_dir}."
        )
    return candidates[0]


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename!r}")
        archive.extractall(output_dir)


def _resolve_source_phase2_dir(extracted_dir: Path) -> Path:
    direct = extracted_dir / "phase2_sandbox"
    if direct.exists():
        return direct
    matches = [
        item
        for item in extracted_dir.rglob("phase2_sandbox")
        if item.is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one phase2_sandbox directory in the bundle; "
            f"found {len(matches)}."
        )
    return matches[0]


def _candidate_and_threshold(input_manifest_path: Path) -> tuple[str, float]:
    manifest = _read_json(input_manifest_path)
    result = manifest.get("result", {}) or {}
    candidate_manifest = manifest.get("candidate_manifest", {}) or {}
    threshold_payload = candidate_manifest.get("threshold", {}) or {}
    candidate_id = str(
        result.get("candidate_id")
        or candidate_manifest.get("candidate_id")
        or DEFAULT_PHASE2_CONTRACT.candidate_id
    )
    threshold = float(
        result.get("threshold")
        or threshold_payload.get("value")
        or DEFAULT_PHASE2_CONTRACT.threshold
    )
    return candidate_id, threshold


def _write_local_pending_gate(gate_dir: Path) -> None:
    gate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        gate_dir / "phase2_readiness.json",
        {
            "ready_for_phase2": False,
            "blockers": [
                "local_lab_sandbox_only",
                "future_unseen_oos_not_ready",
            ],
            "advisories": [
                "Local lab runs are engineering diagnostics, not promotion evidence.",
            ],
            "next_action": "use_clean_confirmation_before_any_selection",
        },
    )
    _write_json(
        gate_dir / "future_oos_readiness.json",
        {
            "evaluation_completed": False,
            "primary_candidate_passed": None,
            "promotion_allowed": False,
        },
    )
    _write_json(
        gate_dir / "report_consistency_audit.json",
        {
            "operator_next_step": {
                "consistency_status": "passed",
                "failed_checks": [],
                "next_action": "local_phase2_lab_sandbox",
            }
        },
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _fold_stats(fold_frame: pd.DataFrame, strategy_id: str) -> dict[str, Any]:
    rows = fold_frame.loc[fold_frame["strategy_id"] == strategy_id].copy()
    if rows.empty:
        return {
            "fold_count": 0,
            "positive_fold_count": 0,
            "negative_fold_count": 0,
            "worst_fold_compounded_return": None,
            "best_fold_compounded_return": None,
            "fold_returns": {},
        }
    returns = pd.to_numeric(rows["compounded_net_return"], errors="coerce")
    return {
        "fold_count": int(len(rows)),
        "positive_fold_count": int((returns > 0).sum()),
        "negative_fold_count": int((returns < 0).sum()),
        "worst_fold_compounded_return": _float_or_none(returns.min()),
        "best_fold_compounded_return": _float_or_none(returns.max()),
        "fold_returns": {
            str(row.fold): _float_or_none(row.compounded_net_return)
            for row in rows.itertuples(index=False)
        },
    }


def _risk_flags(
    row: pd.Series,
    *,
    baseline_base_return: float,
    fold_stats: dict[str, Any],
) -> list[str]:
    flags = ["seen_test_window_only"]
    trade_count = _float_or_none(row.get("trade_count")) or 0.0
    compounded_return = _float_or_none(row.get("compounded_return"))
    bootstrap_prob = _float_or_none(
        row.get("bootstrap_probability_compounded_return_positive")
    )
    max_holding_share = _float_or_none(row.get("max_holding_exit_share"))
    best_month_removed = _float_or_none(
        row.get("best_month_removed_compounded_return")
    )
    if trade_count < 75:
        flags.append("low_trade_count")
    if fold_stats.get("positive_fold_count", 0) < 2:
        flags.append("single_or_zero_positive_fold")
    if bootstrap_prob is not None and bootstrap_prob < 0.50:
        flags.append("bootstrap_positive_probability_below_50pct")
    if compounded_return is not None and compounded_return < 0:
        flags.append("base_cost_return_still_negative")
    if compounded_return is not None and compounded_return <= baseline_base_return:
        flags.append("does_not_improve_baseline_base_return")
    if best_month_removed is not None and best_month_removed < 0:
        flags.append("best_month_removed_return_negative")
    elif (
        best_month_removed is not None
        and compounded_return is not None
        and best_month_removed < compounded_return - 0.03
    ):
        flags.append("best_month_concentration")
    if max_holding_share is not None and max_holding_share > 0.50:
        flags.append("high_max_holding_exit_share")
    return flags


def _hypothesis_status(
    row: pd.Series,
    *,
    baseline_base_return: float,
    fold_stats: dict[str, Any],
    flags: list[str],
) -> str:
    strategy_id = str(row.get("strategy_id"))
    if strategy_id == BASELINE_STRATEGY_ID:
        return "baseline_reference"
    compounded_return = _float_or_none(row.get("compounded_return"))
    adverse_delta = _float_or_none(row.get("adverse_delta_compounded_return_vs_baseline"))
    max_drawdown_delta = _float_or_none(row.get("delta_max_drawdown_vs_baseline"))
    trade_count = _float_or_none(row.get("trade_count")) or 0.0
    if compounded_return is None:
        return "insufficient_evidence"
    improves_base = compounded_return > baseline_base_return
    improves_adverse = adverse_delta is not None and adverse_delta > 0
    drawdown_not_worse = max_drawdown_delta is None or max_drawdown_delta >= -0.02
    enough_trades = trade_count >= 75
    if improves_base and improves_adverse and drawdown_not_worse and enough_trades:
        if any(
            flag in flags
            for flag in (
                "single_or_zero_positive_fold",
                "bootstrap_positive_probability_below_50pct",
                "base_cost_return_still_negative",
                "best_month_removed_return_negative",
                "best_month_concentration",
            )
        ):
            return "candidate_for_clean_confirmation_high_risk"
        return "carry_to_clean_confirmation"
    if improves_base and improves_adverse:
        return "watchlist_low_sample_or_fragile"
    if "does_not_improve_baseline_base_return" in flags:
        return "retire_for_now"
    return "watchlist_diagnostic_only"


def _build_decision_report(
    output_phase2_dir: Path,
    *,
    source_bundle: Path,
    source_phase2_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    summary = pd.read_csv(output_phase2_dir / "phase2_strategy_variant_summary.csv")
    forensics = pd.read_csv(output_phase2_dir / "phase2_strategy_forensics_summary.csv")
    folds = pd.read_csv(output_phase2_dir / "phase2_strategy_variant_by_fold.csv")

    base_rows = summary.loc[summary["cost_scenario"] == "base"].copy()
    adverse_rows = summary.loc[summary["cost_scenario"] == "adverse"].copy()
    adverse_delta = adverse_rows[
        ["strategy_id", "delta_compounded_return_vs_baseline"]
    ].rename(
        columns={
            "delta_compounded_return_vs_baseline": (
                "adverse_delta_compounded_return_vs_baseline"
            )
        }
    )
    merged = (
        base_rows.merge(forensics, on=["strategy_id", "cost_scenario"], suffixes=("", "_forensics"))
        .merge(adverse_delta, on="strategy_id", how="left")
    )
    baseline_row = merged.loc[merged["strategy_id"] == BASELINE_STRATEGY_ID]
    if baseline_row.empty:
        raise ValueError("Baseline strategy row is missing from local lab outputs.")
    baseline_base_return = float(baseline_row.iloc[0]["compounded_return"])

    hypotheses: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        row_series = pd.Series(row)
        stats = _fold_stats(folds, str(row["strategy_id"]))
        flags = _risk_flags(
            row_series,
            baseline_base_return=baseline_base_return,
            fold_stats=stats,
        )
        status = _hypothesis_status(
            row_series,
            baseline_base_return=baseline_base_return,
            fold_stats=stats,
            flags=flags,
        )
        hypotheses.append(
            {
                "strategy_id": row["strategy_id"],
                "status": status,
                "base_compounded_return": _float_or_none(row.get("compounded_return")),
                "base_delta_compounded_return_vs_baseline": _float_or_none(
                    row.get("delta_compounded_return_vs_baseline")
                ),
                "adverse_delta_compounded_return_vs_baseline": _float_or_none(
                    row.get("adverse_delta_compounded_return_vs_baseline")
                ),
                "profit_factor": _float_or_none(row.get("profit_factor")),
                "max_drawdown": _float_or_none(row.get("max_drawdown")),
                "trade_count": int(row.get("trade_count") or 0),
                "net_edge_bps_per_trade": _float_or_none(
                    row.get("net_edge_bps_per_trade")
                ),
                "bootstrap_probability_compounded_return_positive": _float_or_none(
                    row.get("bootstrap_probability_compounded_return_positive")
                ),
                "max_holding_exit_share": _float_or_none(
                    row.get("max_holding_exit_share")
                ),
                "best_month_removed_compounded_return": _float_or_none(
                    row.get("best_month_removed_compounded_return")
                ),
                "fold_stats": stats,
                "overfit_risk_flags": flags,
                "rationale": row.get("rationale"),
            }
        )
    risk_weights = {
        "low_trade_count": 3,
        "single_or_zero_positive_fold": 4,
        "bootstrap_positive_probability_below_50pct": 3,
        "base_cost_return_still_negative": 4,
        "does_not_improve_baseline_base_return": 3,
        "best_month_removed_return_negative": 3,
        "best_month_concentration": 1,
        "high_max_holding_exit_share": 1,
    }
    hypotheses = sorted(
        hypotheses,
        key=lambda item: (
            item["status"] != "carry_to_clean_confirmation",
            item["status"] != "candidate_for_clean_confirmation_high_risk",
            item["status"] != "watchlist_low_sample_or_fragile",
            sum(risk_weights.get(flag, 0) for flag in item["overfit_risk_flags"]),
            -(item["base_delta_compounded_return_vs_baseline"] or -999.0),
        ),
    )
    clean_candidates = [
        item
        for item in hypotheses
        if item["status"]
        in {
            "carry_to_clean_confirmation",
            "candidate_for_clean_confirmation_high_risk",
        }
    ]
    watchlist = [
        item
        for item in hypotheses
        if item["status"] == "watchlist_low_sample_or_fragile"
    ]
    return {
        "lab_version": LOCAL_LAB_VERSION,
        "generated_at_utc": (
            datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        ),
        "source_bundle": str(source_bundle),
        "source_phase2_dir": str(source_phase2_dir),
        "run_dir": str(run_dir),
        "evidence_scope": "already_seen_phase2_sandbox_test_window",
        "automatic_policy_selection_allowed": False,
        "promotion_allowed": False,
        "clean_confirmation_required": True,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "baseline_base_compounded_return": baseline_base_return,
        "clean_confirmation_candidates": clean_candidates,
        "watchlist_candidates": watchlist,
        "hypotheses": hypotheses,
        "operator_summary": {
            "top_clean_confirmation_candidate": (
                clean_candidates[0]["strategy_id"] if clean_candidates else None
            ),
            "top_watchlist_candidate": (
                watchlist[0]["strategy_id"] if watchlist else None
            ),
            "do_not_promote_reason": (
                "Local lab evidence uses an already-seen sandbox window; "
                "selection remains locked until clean Future-OOS or another "
                "pre-registered clean confirmation window."
            ),
        },
    }


def _decision_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Local Lab Decision Report",
        "",
        f"Lab version: `{report['lab_version']}`",
        f"Evidence scope: `{report['evidence_scope']}`",
        "",
        "> Diagnostic only. Automatic winner selection and promotion are disabled.",
        "",
        "## Decision",
        "",
        f"- Promotion allowed: `{report['promotion_allowed']}`",
        f"- Clean confirmation required: `{report['clean_confirmation_required']}`",
        "- Reason: "
        f"{report['operator_summary']['do_not_promote_reason']}",
        "",
        "## Clean-confirmation candidates",
        "",
    ]
    candidates = report.get("clean_confirmation_candidates", [])
    if not candidates:
        lines.append("- None met the stricter carry criteria.")
    for item in candidates:
        lines.append(
            "- "
            f"`{item['strategy_id']}`: base return "
            f"`{item['base_compounded_return']}`, delta vs baseline "
            f"`{item['base_delta_compounded_return_vs_baseline']}`, "
            f"bootstrap positive probability "
            f"`{item['bootstrap_probability_compounded_return_positive']}`."
        )
    lines.extend(["", "## Watchlist", ""])
    watchlist = report.get("watchlist_candidates", [])
    if not watchlist:
        lines.append("- None.")
    for item in watchlist:
        lines.append(
            "- "
            f"`{item['strategy_id']}`: status `{item['status']}`, "
            f"base delta `{item['base_delta_compounded_return_vs_baseline']}`, "
            f"flags `{', '.join(item['overfit_risk_flags'])}`."
        )
    lines.extend(["", "## All hypotheses", ""])
    for item in report.get("hypotheses", []):
        lines.append(
            "- "
            f"`{item['strategy_id']}` -> `{item['status']}`; "
            f"base `{item['base_compounded_return']}`; "
            f"PF `{item['profit_factor']}`; "
            f"trades `{item['trade_count']}`; "
            f"flags `{', '.join(item['overfit_risk_flags'])}`."
        )
    return "\n".join(lines) + "\n"


def _prepare_paths(output_dir: Path, run_id: str | None) -> LocalLabPaths:
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / resolved_run_id
    return LocalLabPaths(
        run_dir=run_dir,
        extracted_dir=run_dir / "bundle_extract",
        source_phase2_dir=run_dir / "bundle_extract" / "phase2_sandbox",
        gate_dir=run_dir / "local_gate",
        output_phase2_dir=run_dir / "phase2_sandbox",
        decision_json_path=run_dir / "phase2_lab_decision_report.json",
        decision_md_path=run_dir / "phase2_lab_decision_report.md",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repeatable local Phase 2 lab from an existing sandbox bundle. "
            "This is diagnostic-only and cannot promote a strategy."
        )
    )
    parser.add_argument(
        "--bundle",
        help=(
            "Path to phase2_latest_sandbox_bundle*.zip. If omitted, the newest "
            "matching file under --downloads-dir is used."
        ),
    )
    parser.add_argument(
        "--downloads-dir",
        default=str(Path.home() / "Downloads"),
        help="Directory searched when --bundle is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/phase2_local_lab",
        help="Local lab output directory. reports/ is gitignored.",
    )
    parser.add_argument("--run-id", help="Optional deterministic local run id.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove an existing output run directory before writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = Path(args.bundle) if args.bundle else _find_latest_bundle(Path(args.downloads_dir))
    if not bundle.exists():
        raise FileNotFoundError(f"Phase 2 bundle not found: {bundle}")
    paths = _prepare_paths(Path(args.output_dir), args.run_id)
    if paths.run_dir.exists() and not args.keep_existing:
        shutil.rmtree(paths.run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    _safe_extract_zip(bundle, paths.extracted_dir)
    source_phase2_dir = _resolve_source_phase2_dir(paths.extracted_dir)
    bars_path = source_phase2_dir / "phase2_bars.csv"
    signals_path = source_phase2_dir / "phase2_signals.csv"
    input_manifest_path = source_phase2_dir / "phase2_input_manifest.json"
    for required in (bars_path, signals_path, input_manifest_path):
        if not required.exists():
            raise FileNotFoundError(f"Bundle is missing required file: {required}")
    candidate_id, threshold = _candidate_and_threshold(input_manifest_path)
    _write_local_pending_gate(paths.gate_dir)

    phase2_sandbox_main(
        [
            "--report-dir",
            str(paths.gate_dir),
            "--bars",
            str(bars_path),
            "--signals",
            str(signals_path),
            "--output-dir",
            str(paths.output_phase2_dir),
            "--mode",
            "sandbox",
            "--candidate-id",
            candidate_id,
            "--threshold",
            str(threshold),
            "--all-cost-scenarios",
            "--strategy-suite",
        ]
    )
    report = _build_decision_report(
        paths.output_phase2_dir,
        source_bundle=bundle,
        source_phase2_dir=source_phase2_dir,
        run_dir=paths.run_dir,
    )
    _write_json(paths.decision_json_path, report)
    paths.decision_md_path.write_text(_decision_markdown(report), encoding="utf-8")
    print(
        "phase2_local_lab_written "
        f"run_dir={paths.run_dir} "
        f"bundle={bundle} "
        f"decision_report={paths.decision_json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
