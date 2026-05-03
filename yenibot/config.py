from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access for YAML configuration."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: ConfigNode


def load_config(path: str | Path = "config.yaml") -> ConfigNode:
    """Load project configuration from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return _to_node(raw)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
