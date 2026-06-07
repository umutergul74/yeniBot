"""Append-only experiment decision registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yenibot.experiment.common import _hash_payload, _json_ready

__all__ = ["append_experiment_registry"]


def append_experiment_registry(
    *,
    registry_path: str | Path,
    snapshot_path: str | Path,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Append one content-addressed event and copy the immutable history snapshot."""

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stable_event = _json_ready(event)
    event_id = _hash_payload(stable_event)
    record = {
        "event_id": event_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **stable_event,
    }

    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing_ids.add(str(json.loads(line).get("event_id", "")))
            except json.JSONDecodeError as error:
                raise ValueError(f"Experiment registry contains invalid JSON: {path}") from error
    if event_id not in existing_ids:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    snapshot = Path(snapshot_path)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return record
