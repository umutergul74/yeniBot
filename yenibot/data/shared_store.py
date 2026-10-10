"""Immutable raw partitions shared by dated research snapshots (one writer only).

There is no mutable 'latest' parquet: extending the store adds partitions without
changing the bytes or the partition list consumed by an older experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _bytes(record):
    return json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")


def snapshot_digest(record):
    return hashlib.sha256(_bytes({k: v for k, v in record.items() if k != "sha256"})).hexdigest()


def _utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _manifest(path):
    return path.with_suffix(path.suffix + ".manifest.json")


def _combine(frames, start, end):
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    result = result.loc[(result.timestamp >= start) & (result.timestamp < end)].reset_index(drop=True)
    if result.empty or result.timestamp.isna().any() or result.timestamp.duplicated().any():
        raise ValueError("Shared raw snapshot has empty, missing or duplicate timestamps")
    if not result.timestamp.is_monotonic_increasing:
        raise ValueError("Shared raw snapshot is not chronological")
    return result


def read_snapshot(path, record):
    """Verify each referenced partition and its pinned identity before combining."""
    from yenibot.notebook_runtime import file_sha256, verified_table

    path = Path(path)
    if record.get("sha256") != snapshot_digest(record) or record.get("kind") != "shared_raw_snapshot":
        raise ValueError(f"Dataset identity mismatch: {path.name}")
    root = (path.parent / record["store_relative"]).resolve()
    start, end = _utc(record["start"]), _utc(record["end_exclusive"])
    frames, cursor = [], start
    for part in record["partitions"]:
        source = (root / part["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Partition path escapes shared raw store")
        lo, hi = _utc(part["start"]), _utc(part["end_exclusive"])
        if lo > cursor or hi <= cursor or (frames and lo != cursor):
            raise ValueError("Shared raw partition coverage overlaps or has a gap")
        metadata = json.loads(_manifest(source).read_text(encoding="utf-8"))
        if (metadata.get("schema") != 1 or metadata.get("sha256") != part["sha256"]
                or file_sha256(source) != part["sha256"]
                or metadata.get("provenance", {}).get("partition_start") != part["start"]
                or metadata.get("provenance", {}).get("partition_end_exclusive") != part["end_exclusive"]
                or metadata.get("provenance", {}).get("store_contract") != record["store_contract"]):
            raise ValueError("Shared raw partition identity or contract mismatch")
        frame = verified_table(source)
        if list(frame.columns) != record["columns"]:
            raise ValueError("Shared raw partition column mismatch")
        frames.append(frame)
        cursor = hi
    if not frames or cursor < end:
        raise ValueError("Shared raw snapshot does not cover its cutoff")
    frame = _combine(frames, start, end)
    if len(frame) != record["rows"]:
        raise ValueError("Shared raw snapshot row mismatch")
    return frame


def prepare_shared_table(path, store, contract, start, end, download, validate, provenance,
                         *, validate_complete=None):
    """Fetch missing monthly/range partitions and pin a lightweight workspace manifest."""
    from yenibot.notebook_runtime import publish_table, verified_table

    path, store = Path(path), Path(store).resolve()
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("Shared raw end must be after start")
    identity = hashlib.sha256(_bytes(contract)).hexdigest()
    directory = store / identity
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for manifest in directory.glob("*.parquet.manifest.json"):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        info = metadata["provenance"]
        if info.get("store_contract") != contract:
            raise ValueError("Shared store source contract mismatch")
        lo, hi = _utc(info["partition_start"]), _utc(info["partition_end_exclusive"])
        if hi <= lo:
            raise ValueError("Invalid shared partition interval")
        if lo < end and hi > start:
            entries.append((lo, hi, manifest.with_name(manifest.name.removesuffix(".manifest.json"))))
    entries.sort(key=lambda item: item[0])
    for previous, following in zip(entries, entries[1:]):
        if previous[1] > following[0]:
            raise ValueError("Shared raw partitions overlap")

    def fill(lo, hi):
        while lo < hi:
            boundary = min(hi, lo.normalize().replace(day=1) + pd.offsets.MonthBegin(1))
            # Microsecond precision is explicit in names; never silently reuse a rounded interval.
            name = f"{lo:%Y%m%dT%H%M%S%f}_{boundary:%Y%m%dT%H%M%S%f}.parquet"
            destination = directory / name
            if destination.exists() or _manifest(destination).exists():
                raise ValueError("Incomplete shared partition; preserve and inspect before retrying")
            frame = validate(download(lo.isoformat(), boundary.isoformat()), lo.isoformat(), boundary.isoformat())
            info = {"store_contract": contract, "partition_start": lo.isoformat(),
                    "partition_end_exclusive": boundary.isoformat()}
            if "funding_sources" in frame.attrs:
                info["sources"] = frame.attrs["funding_sources"]
            publish_table(frame, destination, provenance=info)
            entries.append((lo, boundary, destination))
            print("Added shared raw range:", path.name, lo.isoformat(), boundary.isoformat(), len(frame))
            lo = boundary

    cursor = start
    for lo, hi, _ in list(entries):
        if lo > cursor:
            fill(cursor, lo)
        cursor = max(cursor, hi)
    if cursor < end:
        fill(cursor, end)
    entries.sort(key=lambda item: item[0])
    frames, parts = [], []
    for lo, hi, source in entries:
        frame = verified_table(source)
        metadata = json.loads(_manifest(source).read_text(encoding="utf-8"))
        frames.append(frame)
        parts.append({"path": source.relative_to(store).as_posix(), "sha256": metadata["sha256"],
                      "start": lo.isoformat(), "end_exclusive": hi.isoformat()})
    result = _combine(frames, start, end)
    if validate_complete is not None:
        validate_complete(result)
    record = {"schema": 2, "kind": "shared_raw_snapshot", "store_contract": contract,
              "store_relative": Path(os.path.relpath(store, path.parent)).as_posix(),
              "start": start.isoformat(), "end_exclusive": end.isoformat(),
              "rows": len(result), "columns": list(result.columns), "partitions": parts,
              "provenance": provenance}
    record["sha256"] = snapshot_digest(record)
    read_snapshot(path, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This logical dataset has a manifest only, not another full parquet copy.
    if path.exists() or _manifest(path).exists():
        raise ValueError("Refusing to replace an existing research snapshot")
    with _manifest(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result
