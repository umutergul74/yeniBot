"""Aggregate immutable complete-block reports under the preregistered v2 gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from yenibot.automation.phase2_forward_shadow_prepare import _canonical_spec
from yenibot.phase2.forward_evaluator import evaluate_forward_confirmation
from yenibot.phase2.full_oof import file_sha256


def confirm_forward_shadow(
    *,
    evaluation_dirs: list[str | Path],
    output_path: str | Path,
    spec_path: str | Path = "configs/forward_shadow_v2.json",
) -> dict:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a forward confirmation report")
    spec, spec_hash = _canonical_spec(Path(spec_path))
    loaded = []
    for value in evaluation_dirs:
        root = Path(value)
        report_path = root / "forward_shadow_evaluation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete_forward_shadow_block_evaluated":
            raise ValueError(f"Incomplete block evaluation cannot enter confirmation: {root}")
        expected = report.get("evaluation_artifact_sha256", {}) or {}
        for name, sha256 in expected.items():
            path = root / name
            if not path.is_file() or file_sha256(path) != sha256:
                raise ValueError(f"Changed confirmation input artifact: {path}")
        opportunities = pd.read_csv(
            root / "score_payoff_opportunities.csv", float_precision="round_trip"
        )
        equity = pd.read_csv(
            root / "equity_candidate_adverse.csv", float_precision="round_trip"
        )
        loaded.append((report, opportunities, equity))
    loaded.sort(key=lambda item: int(item[0]["block_ordinal"]))
    result = evaluate_forward_confirmation(
        [item[0] for item in loaded],
        [item[1] for item in loaded],
        [item[2] for item in loaded],
        spec=spec,
    )
    result.update(
        process_id=spec["version"],
        contract_canonical_sha256=spec_hash,
        source_evaluation_count=len(loaded),
        source_manifest_hashes=[item[0]["manifest_hash"] for item in loaded],
        source_registration_hashes=[
            item[0]["registration_hash"] for item in loaded
        ],
        fit_operations_performed=0,
        selection_operations_performed=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", default="configs/forward_shadow_v2.json")
    args = parser.parse_args(argv)
    result = confirm_forward_shadow(
        evaluation_dirs=args.evaluation_dir,
        output_path=args.output,
        spec_path=args.spec,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
