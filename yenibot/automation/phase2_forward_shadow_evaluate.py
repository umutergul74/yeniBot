"""Evaluate one registered forward-shadow block only after all outcomes mature."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd

from yenibot.phase2.forward_evaluator import (
    evaluate_complete_shadow_block,
    shadow_block_preflight,
)
from yenibot.phase2.forward_shadow import (
    load_shadow_manifest,
    read_shadow_ledger,
    validate_shadow_registration,
)
from yenibot.phase2.full_oof import file_sha256


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def evaluate_forward_shadow(
    *,
    block_dir: str | Path,
    registration_path: str | Path,
    ledger_path: str | Path,
    market_path: str | Path,
    output_dir: str | Path,
) -> dict:
    block_root = Path(block_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a forward-shadow evaluation")
    manifest = load_shadow_manifest(
        block_root / "forward_shadow_manifest.json", artifact_root=block_root
    )
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    validate_shadow_registration(registration, manifest=manifest)
    ledger, ledger_head = read_shadow_ledger(ledger_path, with_hash=True)
    market = pd.read_parquet(market_path)
    preflight = shadow_block_preflight(
        ledger,
        market,
        manifest=manifest,
        registration=registration,
    )
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        if not preflight["ready_for_complete_block_evaluation"]:
            report = {
                "status": "waiting_for_complete_timely_block_and_mature_outcomes",
                "block_id": manifest["block"]["block_id"],
                "manifest_hash": manifest["manifest_hash"],
                "registration_hash": registration["registration_hash"],
                "ledger_head_hash": ledger_head,
                "preflight": preflight,
                "fit_operations_performed": 0,
                "selection_operations_performed": 0,
                "promotion_allowed": False,
                "live_trading_allowed": False,
            }
        else:
            evaluated = evaluate_complete_shadow_block(
                ledger,
                market,
                manifest=manifest,
                registration=registration,
            )
            artifact_hashes = {}
            for key, frame in evaluated.trades.items():
                path = staging / f"trades_{key}.csv"
                frame.to_csv(path, index=False)
                artifact_hashes[path.name] = file_sha256(path)
            for key, frame in evaluated.equity.items():
                path = staging / f"equity_{key}.csv"
                frame.to_csv(path, index=False)
                artifact_hashes[path.name] = file_sha256(path)
            opportunities_path = staging / "score_payoff_opportunities.csv"
            evaluated.opportunities.to_csv(opportunities_path, index=False)
            artifact_hashes[opportunities_path.name] = file_sha256(opportunities_path)
            report = {
                **evaluated.report,
                "ledger_head_hash": ledger_head,
                "evaluation_artifact_sha256": artifact_hashes,
            }
        _write(staging / "forward_shadow_evaluation.json", report)
        os.replace(staging, output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dir", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = evaluate_forward_shadow(
        block_dir=args.block_dir,
        registration_path=args.registration,
        ledger_path=args.ledger,
        market_path=args.market,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
