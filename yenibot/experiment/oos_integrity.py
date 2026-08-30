"""Content integrity for one-shot OOS evidence; never fit or select a model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEAL_NAME = "future_oos_artifact_integrity.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_record(row, outcome) -> bool:
    if not outcome or any(k not in outcome for k in ("rows", "data_start", "data_end")):
        return False
    if int(row.get("rows", -1)) != int(outcome["rows"]):
        return False
    for key in ("data_start", "data_end"):
        if pd.Timestamp(row.get(key)) != pd.Timestamp(outcome[key]):
            return False
    for key, expected in outcome.items():
        if (
            isinstance(expected, (float, int))
            and not isinstance(expected, bool)
            and key != "rows"
        ):
            observed = row.get(key)
            if observed is None or not math.isclose(
                float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-9
            ):
                return False
    return True


def verify_family(root: Path, names, *, primary_id: str, outcome: dict) -> dict:
    """Verify seals when present; admit legacy evidence only against committed pins."""
    names = [n for n in names if n != "future_oos_readiness.json"]
    hashes = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty immutable OOS artifact: {name}")
        hashes[name] = file_sha256(path)
    seal_path = root / SEAL_NAME
    if seal_path.exists():
        sealed = json.loads(seal_path.read_text(encoding="utf-8"))
        if sealed.get("files") != hashes or sealed.get("candidate_id") != primary_id:
            raise ValueError("Immutable OOS artifact hash mismatch")
    frame = pd.read_parquet(root / "future_oos_predictions.parquet")
    if "candidate_id" not in frame:
        raise ValueError("OOS predictions have no candidate identity")
    frame = frame.loc[frame.candidate_id.astype(str).eq(primary_id)]
    times = pd.to_datetime(frame["timestamp"], utc=True)
    if (
        len(frame) != int(outcome["rows"])
        or times.duplicated().any()
        or times.isna().any()
    ):
        raise ValueError("OOS prediction row contract mismatch")
    if times.min() != pd.Timestamp(
        outcome["data_start"]
    ) or times.max() != pd.Timestamp(outcome["data_end"]):
        raise ValueError("OOS prediction window mismatch")
    numeric = frame[["prob_long", "label", "forward_return"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not frame.prob_long.between(0, 1).all():
        raise ValueError("Invalid OOS predictions")
    # Historical admission also checks the pinned ranking statistic; the
    # original gate decision and bootstrap results are never regenerated.
    if "rank_ic" in outcome:
        observed = frame.prob_long.corr(frame.forward_return, method="spearman")
        expected = float(outcome["rank_ic"])
        if not (
            (math.isnan(observed) and math.isnan(expected))
            or math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-9)
        ):
            raise ValueError("OOS predictions disagree with recorded Rank IC")
    return {
        "version": "oos_artifact_integrity_v1",
        "candidate_id": primary_id,
        "rows": int(outcome["rows"]),
        "data_start": outcome["data_start"],
        "data_end": outcome["data_end"],
        "files": hashes,
    }
