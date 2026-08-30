"""Separate append-only forward predictions from one-shot OOS evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def append_forward_predictions(
    path: Path, frame: pd.DataFrame, *, candidate_id: str, manifest_hash: str
) -> dict:
    """Idempotent append with exact overlap verification and a hash chain.

    Call after frozen-model inference, never with refitted/reselected scores.
    Missing previous hours and changed historical rows fail closed. A lock file
    excludes concurrent writers; an interrupted lock requires manual review.
    """
    if not candidate_id or not manifest_hash:
        raise ValueError("Forward predictions require candidate and manifest identity")
    required = [
        "timestamp",
        "candidate_id",
        "manifest_hash",
        "prob_long",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
    ]
    if not set(required).issubset(frame):
        raise ValueError(f"Forward predictions require {required}")
    incoming = frame[required].copy()
    incoming.timestamp = pd.to_datetime(incoming.timestamp, utc=True)
    if incoming.timestamp.isna().any() or incoming.timestamp.duplicated().any():
        raise ValueError("Forward timestamps must be unique and finite")
    if (
        not incoming.candidate_id.eq(candidate_id).all()
        or not incoming.manifest_hash.eq(manifest_hash).all()
    ):
        raise ValueError("Forward model identity mismatch")
    numeric = required[3:]
    incoming[numeric] = incoming[numeric].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(incoming[numeric].to_numpy()).all()
        or not incoming.prob_long.between(0, 1).all()
    ):
        raise ValueError("Invalid forward numeric values")
    incoming = incoming.sort_values("timestamp")
    incoming.timestamp = incoming.timestamp.map(lambda t: t.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        old, previous = read_forward_ledger(path, with_hash=True)
        if not old.empty and (
            not old.candidate_id.eq(candidate_id).all()
            or not old.manifest_hash.eq(manifest_hash).all()
        ):
            raise ValueError("Cannot append a different model to a forward ledger")
        by_time = {row["timestamp"]: row for row in old.to_dict("records")}
        additions = []
        for row in incoming.to_dict("records"):
            time = row["timestamp"]
            if time in by_time:
                if row != by_time[time]:
                    raise ValueError("Previously recorded forward prediction changed")
                continue
            last = (
                additions[-1]["timestamp"]
                if additions
                else (max(by_time) if by_time else None)
            )
            if last is not None and pd.Timestamp(time) - pd.Timestamp(
                last
            ) != pd.Timedelta(hours=1):
                raise ValueError(
                    "Forward appends must be contiguous and strictly later"
                )
            additions.append(row)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in additions:
                payload = {"previous_hash": previous, "prediction": row}
                previous = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                handle.write(
                    json.dumps({**payload, "hash": previous}, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "appended_rows": len(additions),
            "total_rows": len(old) + len(additions),
            "head_hash": previous,
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "fit_operations_performed": 0,
        }
    finally:
        os.close(fd)
        lock.unlink()


def read_forward_ledger(path: Path, *, with_hash=False):
    rows, previous = [], ""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            stored = payload.pop("hash")
            actual = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if payload["previous_hash"] != previous or stored != actual:
                raise ValueError("Forward ledger hash chain is invalid")
            rows.append(payload["prediction"])
            previous = stored
    frame = pd.DataFrame(rows)
    return (frame, previous) if with_hash else frame
