"""Append label-free, causally timed predictions for one registered shadow block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yenibot.config import load_config
from yenibot.phase2.forward_shadow import (
    append_shadow_predictions,
    build_shadow_scores,
    load_shadow_manifest,
    predict_label_free_artifacts,
    validate_shadow_registration,
)


def score_forward_shadow(
    *,
    features_path: str | Path,
    block_dir: str | Path,
    registration_path: str | Path,
    ledger_path: str | Path,
    config_path: str | Path = "config.yaml",
    device: str | None = None,
) -> dict:
    """Score only closed source bars; labels and returns are neither read nor required."""

    features_path = Path(features_path)
    block_root = Path(block_dir)
    manifest = load_shadow_manifest(
        block_root / "forward_shadow_manifest.json", artifact_root=block_root
    )
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    validate_shadow_registration(registration, manifest=manifest)
    frame = pd.read_parquet(features_path)
    forbidden = {"label", "fwd_return_10h", "forward_return", "tb_return"}
    inference = frame.drop(
        columns=[column for column in forbidden if column in frame], errors="ignore"
    )
    raw = predict_label_free_artifacts(
        inference,
        manifest=manifest,
        artifact_root=block_root,
        config=dict(load_config(config_path)),
        device=device,
    )
    generated_at = pd.Timestamp.now(tz="UTC")
    raw = raw.loc[
        pd.to_datetime(raw.timestamp, utc=True) + pd.Timedelta(hours=1)
        <= generated_at
    ].reset_index(drop=True)
    if raw.empty:
        return {
            "status": "no_closed_evidence_bar_available",
            "block_id": manifest["block"]["block_id"],
            "generated_at_utc": generated_at.isoformat(),
            "appended_rows": 0,
            "fit_operations_performed": 0,
            "selection_operations_performed": 0,
        }
    cdf_record = manifest["artifacts"]["validation_cdf"]
    reference = np.load(block_root / cdf_record["path"], allow_pickle=False)
    scored = build_shadow_scores(
        raw,
        manifest=manifest,
        cdf_reference=reference,
        generated_at=generated_at,
    )
    result = append_shadow_predictions(
        ledger_path,
        scored,
        manifest=manifest,
        registration=registration,
    )
    result.update(
        status="shadow_predictions_recorded",
        generated_at_utc=generated_at.isoformat(),
        label_columns_consumed=False,
        forward_return_columns_consumed=False,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--block-dir", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    result = score_forward_shadow(
        features_path=args.features,
        block_dir=args.block_dir,
        registration_path=args.registration,
        ledger_path=args.ledger,
        config_path=args.config,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
