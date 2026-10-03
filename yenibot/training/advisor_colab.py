"""Colab I/O helpers. No training, architecture selection or test evaluation."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from yenibot.training.advisor import atomic_json, sha256


def freeze_market_inputs(source_dir: Path, destination: Path, *, start: str, end: str) -> dict:
    """Pin inputs once; never silently replace a previous experiment's snapshot."""
    manifest_path = destination / "snapshot_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["start"] != start or manifest["end"] != end:
            raise ValueError("Input snapshot dates changed; use a new snapshot directory")
        for entry in manifest["files"].values():
            if sha256(destination / entry["file"]) != entry["sha256"]:
                raise ValueError("Frozen input checksum mismatch")
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "advisor_market_snapshot_v1", "start": start, "end": end,
                "created_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    for interval in ("1h", "4h"):
        source = source_dir / f"btc_{interval}.parquet"
        if not source.exists():
            raise FileNotFoundError(f"Required raw input is missing: {source}")
        frame = pd.read_parquet(source)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[frame.timestamp.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if frame.empty or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError("Empty, duplicate or unordered source timestamps")
        if not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=int(interval[:-1]))).all():
            raise ValueError("Raw input contains missing bars")
        if frame.timestamp.iloc[0] != pd.Timestamp(start):
            raise ValueError("Raw history starts too late; cannot silently shorten the experiment")
        expected_last = pd.Timestamp(end).floor(f"{interval[:-1]}h")
        if frame.timestamp.iloc[-1] != expected_last:
            raise ValueError("Raw history ends too early for target-label maturation")
        target = destination / source.name
        temporary = target.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        manifest["files"][interval] = {"file": target.name, "sha256": sha256(target),
                                      "rows": len(frame), "source_file": str(source),
                                      "first": frame.timestamp.iloc[0].isoformat(),
                                      "last": frame.timestamp.iloc[-1].isoformat()}
    atomic_json(manifest_path, manifest)
    return manifest


def cache_frozen_inputs(snapshot: Path, local_dir: Path) -> None:
    """Read big data from runtime SSD; persistent source/checkpoints stay in Drive."""
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    for entry in manifest["files"].values():
        source = snapshot / entry["file"]
        target = local_dir / entry["file"]
        if not target.exists() or sha256(target) != entry["sha256"]:
            shutil.copyfile(source, target)
        if sha256(target) != entry["sha256"]:
            raise ValueError("Local data copy failed checksum verification")
    shutil.copyfile(snapshot / "snapshot_manifest.json", local_dir / "snapshot_manifest.json")


def create_review_bundle(root: Path, session_metadata: dict) -> Path:
    """Small review ZIP: metrics, validation predictions and provenance; no weights."""
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = reports / f"advisor_review_{stamp}.zip"
    files = []
    allowed = {".json", ".yaml", ".yml", ".csv", ".md", ".log", ".txt"}
    for folder in (root / "runs", root / "sessions", root / "data"):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and (path.suffix in allowed or path.name == "validation_predictions.parquet"):
                files.append(path)
    inventory = [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path),
                  "size": path.stat().st_size} for path in files]
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())
        archive.writestr("review_manifest.json", json.dumps({"session": session_metadata, "files": inventory,
                         "weights_included": False, "role": "validation_development_only"}, indent=2))
    os.replace(temporary, destination)
    latest = reports / "advisor_latest_review_bundle.zip"
    temporary_latest = latest.with_suffix(".tmp.zip")
    shutil.copyfile(destination, temporary_latest)
    os.replace(temporary_latest, latest)
    return latest
