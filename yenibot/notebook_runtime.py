"""Prospective notebook workspaces and verified, write-once dataset artifacts.

These checks never import legacy data automatically or modify frozen evidence.
Single writer per workspace is required; a Drive mount is not a distributed lock.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def environment_identity() -> dict:
    # Record the complete runtime inventory without URLs, credentials or local paths.
    packages = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()
                if dist.metadata.get("Name")}
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "packages": dict(sorted(packages.items()))}


def resolve_data_settings(cutoff_value: str, research_id: str, commit: str, config: dict,
                          *, now: datetime | None = None, environment: dict | None = None):
    """Resolve automatic ingestion settings once; downstream notebooks use the printed values."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")
    if cutoff_value == "latest_complete_day":
        cutoff = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        try:
            cutoff = datetime.fromisoformat(cutoff_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Set DATA_END_UTC or use latest_complete_day in notebook 01") from exc
    if cutoff.tzinfo is None or cutoff > now:
        raise ValueError("DATA_END_UTC must be a past timezone-aware cutoff")
    cutoff = cutoff.astimezone(timezone.utc)
    if research_id == "auto":
        identity = {"commit": commit, "config": config, "cutoff": cutoff.isoformat(),
                    "environment": environment if environment is not None else environment_identity()}
        digest = hashlib.sha256(_json_bytes(identity)).hexdigest()[:16]
        research_id = f"data_{cutoff:%Y%m%d}_{digest}"
    return cutoff.isoformat(), research_id


def initialize_workspace(base: Path, research_id: str, repository: Path,
                         expected_commit: str, config: dict, *, environment: dict | None = None) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", research_id):
        raise ValueError("Set a nonempty research_id using letters, digits, dash or underscore")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Pin the full lowercase 40-character Git commit before running")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                    cwd=repository, text=True).strip()
    if actual != expected_commit or dirty:
        raise ValueError("Repository revision differs or checkout is dirty; use a clean pinned checkout")
    root = Path(base) / research_id
    contract = {"schema": 1, "commit": actual, "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
                "environment": environment if environment is not None else environment_identity()}
    manifest = root / "workspace.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != contract:
            raise ValueError("Workspace code/config/environment changed; restore it or choose a new research_id")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Unmanifested workspace is nonempty; no automatic adoption of legacy artifacts")
        root.mkdir(parents=True, exist_ok=True)
        with manifest.open("xb") as stream:
            stream.write(_json_bytes(contract))
    for name in ("data/raw", "data/processed", "checkpoints", "reports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def verified_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("sha256") != file_sha256(path):
        raise ValueError(f"Dataset identity mismatch: {path.name}")
    frame = pd.read_parquet(path)
    if len(frame) != record.get("rows") or list(frame.columns) != record.get("columns"):
        raise ValueError(f"Dataset contract mismatch: {path.name}")
    return frame


def publish_table(frame: pd.DataFrame, path: Path, *, parents: list[Path] = (),
                  provenance: dict | None = None) -> dict:
    """Publish without replacement; interrupted/unmanifested artifacts fail closed.

    Provenance records caller-provided origin claims, not provider authentication.
    Parent identities are checked before publication and recorded by value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_records = []
    for parent in parents:
        verified_table(parent)
        parent_records.append({"name": Path(parent).name, "sha256": file_sha256(parent)})
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        frame.to_parquet(temporary, index=False)
        record = {"schema": 1, "sha256": file_sha256(temporary), "rows": len(frame),
                  "columns": list(frame.columns), "parents": parent_records,
                  "provenance": provenance or {}}
        if path.exists() or manifest_path.exists():
            verified_table(path)
            if json.loads(manifest_path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"Refusing to replace frozen dataset: {path.name}; use a new research_id")
            return record
        # Exclusive creation prevents overwriting an existing writer's result.
        # A crash before manifest publication requires explicit recovery, never adoption.
        with path.open("xb") as destination, temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(chunk)
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(record))
        verified_table(path)
        return record
    finally:
        temporary.unlink(missing_ok=True)
