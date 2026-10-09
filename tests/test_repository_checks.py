import pytest

from scripts.check_repository import secret_kinds, validate_skill


def test_high_confidence_credentials_are_detected_without_returning_values() -> None:
    token = b"ghp_" + b"a" * 36
    key = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert secret_kinds(token + b"\n" + key) == ["private-key", "github-token"]


def test_ordinary_source_and_placeholders_do_not_match() -> None:
    assert secret_kinds(b"API_KEY=<set-locally>\nseed=42") == []


def test_skill_validator_rejects_empty_description_and_duplicate_names(tmp_path):
    directory = tmp_path / "research-review"
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text('---\nname: research-review\ndescription: ""\n---\nInstructions', encoding="utf-8")
    with pytest.raises(ValueError, match="description"):
        validate_skill(path, set())
    path.write_text('---\nname: research-review\ndescription: Review research\n---\nInstructions', encoding="utf-8")
    names = set()
    validate_skill(path, names)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_skill(path, names)
