"""Colab I/O helpers. No training, architecture selection or test evaluation."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from yenibot.features.advisor_oi import normalize_oi_source

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


def pin_completed_a_reference(root: Path) -> Path:
    """Select by data identity/completion, never by validation performance."""
    target = root / "data/A_reference.manifest.json"
    candidate = target if target.exists() else root / "data/development_wavelet_off.manifest.json"
    if not candidate.exists():
        raise FileNotFoundError("Completed A data manifest not found in Drive; use the same colab_v1 folder")
    metadata = json.loads(candidate.read_text())
    found = []
    for directory in sorted((root / "runs").glob("A_*")):
        if not (directory / "status.json").exists():
            continue
        status = json.loads((directory / "status.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        expected = {(seed, fold) for seed in protocol["seeds"] for fold in protocol["folds"]}
        actual = {(entry["seed"], entry["fold"]) for entry in status["completed"]}
        if status["status"] == "complete" and actual == expected and protocol["data_sha256"] == metadata["frame_sha256"]:
            found.append(directory.name)
    if not found:
        raise ValueError("No completed A scope matches the saved base data manifest")
    if not target.exists():
        atomic_json(target, metadata)
    print(f"Pinned A data reference: {found}", flush=True)
    return target


def freeze_oi_inputs(source: Path | None, destination: Path, *, start: str, end: str,
                     downloader=None) -> Path:
    """Reuse Drive source; otherwise cache downloads per month before freezing."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "btc_futures_metrics.parquet"
    manifest_path = target.with_suffix(".manifest.json")
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["start"] != start or saved["end"] != end or sha256(target) != saved["sha256"]:
            raise ValueError("Frozen OI date/checksum mismatch")
        return target
    origin = str(source) if source else "Binance Vision monthly cached downloads"
    if source is not None:
        frame = normalize_oi_source(pd.read_parquet(source))
    else:
        if downloader is None:
            raise FileNotFoundError("OI source not found and no archive downloader configured")
        cache = destination / "monthly_cache"
        cache.mkdir(exist_ok=True)
        chunks = []
        month = pd.Timestamp(start).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        finish = pd.Timestamp(end) + pd.Timedelta(minutes=5)
        while month < finish:
            next_month = month + pd.offsets.MonthBegin(1)
            lower, upper = max(month, pd.Timestamp(start)), min(next_month, finish)
            path = cache / f"{month:%Y%m}.parquet"
            marker = path.with_suffix(".manifest.json")
            if marker.exists():
                info = json.loads(marker.read_text())
                if info["lower"] != lower.isoformat() or info["upper"] != upper.isoformat() or sha256(path) != info["sha256"]:
                    raise ValueError("OI monthly cache checksum/date mismatch")
                chunk = pd.read_parquet(path)
                print(f"OI month {month:%Y-%m}: reused ({len(chunk)} rows)", flush=True)
            else:
                chunk = normalize_oi_source(downloader(lower, upper))
                temporary = path.with_suffix(".tmp.parquet")
                chunk.to_parquet(temporary, index=False)
                os.replace(temporary, path)
                atomic_json(marker, {"lower": lower.isoformat(), "upper": upper.isoformat(), "sha256": sha256(path)})
                print(f"OI month {month:%Y-%m}: downloaded ({len(chunk)} rows)", flush=True)
            chunks.append(chunk)
            month = next_month
        frame = normalize_oi_source(pd.concat(chunks, ignore_index=True))
    frame = frame.loc[frame.timestamp.between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
    frame = normalize_oi_source(frame)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    atomic_json(manifest_path, {"start": start, "end": end, "sha256": sha256(target), "rows": len(frame),
                              "origin": origin, "first": frame.timestamp.iloc[0].isoformat(),
                              "last": frame.timestamp.iloc[-1].isoformat(), "immutable": True})
    return target


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
