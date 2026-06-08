"""Versioned validation-charter governance reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from yenibot.experiment.common import _cfg, _table_markdown, _write_json

__all__ = ["write_validation_charter_status"]


def _charter_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    charter = _cfg(config, ["validation", "charter"], {}) or {}
    active_version = str(charter.get("active_version", "v3_legacy"))
    versions = charter.get("versions", {}) or {}
    rows: list[dict[str, Any]] = []
    for version, definition in versions.items():
        definition = definition if isinstance(definition, dict) else {}
        configured_status = str(definition.get("status", "unknown"))
        is_active = str(version) == active_version
        rows.append(
            {
                "version": str(version),
                "configured_status": configured_status,
                "active_for_phase1_readiness": is_active,
                "official_gate_unchanged": active_version == "v3_legacy",
                "activation_policy": str(
                    charter.get(
                        "activation_policy",
                        "explicit_reviewed_config_and_documentation_commit_only",
                    )
                ),
                "description": str(definition.get("description", "")),
            }
        )
    if not rows:
        rows.append(
            {
                "version": active_version,
                "configured_status": "active",
                "active_for_phase1_readiness": True,
                "official_gate_unchanged": active_version == "v3_legacy",
                "activation_policy": str(charter.get("activation_policy", "")),
                "description": "",
            }
        )
    if sum(bool(row["active_for_phase1_readiness"]) for row in rows) != 1:
        raise ValueError("Exactly one validation charter version must be active")
    active = next(row for row in rows if row["active_for_phase1_readiness"])
    if str(active["configured_status"]) != "active":
        raise ValueError(
            "The selected validation charter version must also have status=active; "
            "draft charters cannot be activated by changing active_version alone"
        )
    return rows


def write_validation_charter_status(
    report_dir: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Write the explicitly configured validation-charter status."""

    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(_charter_rows(config))
    active = frame.loc[frame["active_for_phase1_readiness"].astype(bool)].iloc[0]
    charter = _cfg(config, ["validation", "charter"], {}) or {}
    versions = charter.get("versions", {}) or {}
    active_definition = versions.get(str(active["version"]), {}) or {}
    payload = {
        "active_version": str(active["version"]),
        "active_definition": active_definition,
        "official_gate_unchanged": bool(active["official_gate_unchanged"]),
        "automatic_activation_allowed": False,
        "versions": frame.to_dict(orient="records"),
    }
    frame.to_csv(path / "validation_charter_status.csv", index=False)
    governance_note = (
        "The active charter was selected explicitly in committed config and documentation. "
        "No draft charter can activate itself automatically."
    )
    (path / "validation_charter_status.md").write_text(
        _table_markdown("Validation Charter Status", frame)
        + f"\n\n{governance_note}\n",
        encoding="utf-8",
    )
    _write_json(path / "validation_charter_status.json", payload)
    return frame
