from pathlib import Path

import pytest
import yaml

from yenibot.config import load_config


@pytest.mark.parametrize("text", ["seed: 42\nseed: 43\n", "fit:\n  seed: 42\n  seed: 43\n"])
def test_duplicate_policy_keys_fail_closed(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate configuration key"):
        load_config(path)


def test_yaml_merge_override_and_attribute_access_remain_supported(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("base: &base\n  seed: 42\nfit:\n  <<: *base\n  seed: 43\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.base.seed == 42
    assert cfg.fit.seed == 43


def test_python_yaml_tags_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("x: !!python/tuple [1, 2]\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(path)
