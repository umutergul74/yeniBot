"""Experiment memory registry reports.

These reports expose the config-level rejected/reference profile memory in every
diagnostics bundle so stale or already-failed ideas do not quietly re-enter the
research queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _cfg, _json_ready, _table_markdown, _write_json

__all__ = [
    "_experiment_memory_registry_frame",
    "_write_experiment_memory_registry",
]


def _memory_reason(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("reason") or "")
    return str(value or "")


def _experiment_memory_registry_frame(config: dict[str, Any]) -> pd.DataFrame:
    memory = _cfg(config, ["experiments", "experiment_memory"], default={}) or {}
    rejected = memory.get("rejected_profiles", {}) or {}
    references = memory.get("reference_notes", {}) or {}
    allow_retest = {str(item) for item in memory.get("allow_retest_profiles", []) or []}
    rows: list[dict[str, Any]] = []

    for profile in sorted(str(key) for key in rejected):
        rows.append(
            {
                "profile": profile,
                "memory_status": "rejected",
                "reason": _memory_reason(rejected.get(profile)),
                "allow_retest": profile in allow_retest,
                "auto_retest_blocked": profile not in allow_retest,
                "source": "config.experiments.experiment_memory.rejected_profiles",
            }
        )

    rejected_names = {str(key) for key in rejected}
    for profile in sorted(str(key) for key in references):
        if profile in rejected_names:
            continue
        rows.append(
            {
                "profile": profile,
                "memory_status": "reference",
                "reason": str(references.get(profile) or ""),
                "allow_retest": profile in allow_retest,
                "auto_retest_blocked": False,
                "source": "config.experiments.experiment_memory.reference_notes",
            }
        )

    columns = [
        "profile",
        "memory_status",
        "reason",
        "allow_retest",
        "auto_retest_blocked",
        "source",
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_experiment_memory_registry(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "experiment_memory_registry.csv", index=False)
    (path / "experiment_memory_registry.md").write_text(
        _table_markdown("Experiment Memory Registry", frame),
        encoding="utf-8",
    )
    _write_json(
        path / "experiment_memory_registry.json",
        {
            "rejected_count": int((frame["memory_status"] == "rejected").sum())
            if not frame.empty
            else 0,
            "reference_count": int((frame["memory_status"] == "reference").sum())
            if not frame.empty
            else 0,
            "rows": _json_ready(frame.to_dict(orient="records")),
        },
    )
