"""Offline config/skill validation and high-confidence tracked-secret checks.

Not a full secret detector or a vulnerability audit. Never print matched values.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yenibot.config import load_config  # noqa: E402

SECRET_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "aws-access-key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
}


def secret_kinds(content: bytes) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(content)]


def validate_skill(path: Path, names: set[str]) -> None:
    parts = path.read_text(encoding="utf-8").split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("Missing skill frontmatter")
    data = yaml.safe_load(parts[1])
    name = data["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9-]{1,64}", name):
        raise ValueError("Invalid skill name")
    if name != path.parent.name or name in names:
        raise ValueError("Duplicate or mismatched skill name")
    if not isinstance(data["description"], str) or not data["description"].strip():
        raise ValueError("Missing skill description")
    if not parts[2].strip():
        raise ValueError("Empty skill instructions")
    names.add(name)


def main() -> int:
    errors = []
    config_paths = [ROOT / "config.yaml", *sorted((ROOT / "configs").glob("*.yaml"))]
    for path in config_paths:
        try:
            load_config(path)
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid config ({type(exc).__name__})")
    for path in [ROOT / "pyproject.toml", *sorted((ROOT / ".codex").rglob("*.toml"))]:
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid TOML ({type(exc).__name__})")
    names = set()
    skills = sorted((ROOT / ".agents" / "skills").glob("*/SKILL.md"))
    for path in skills:
        try:
            validate_skill(path, names)
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid skill ({type(exc).__name__})")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    for raw in filter(None, tracked):
        relative = raw.decode("utf-8")
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            errors.append(f"{relative}: tracked environment secret file")
        # Scan source/config/docs, including notebook JSON, without deserialization.
        if path.suffix.lower() not in {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".json", ".ipynb", ".ps1", ".pem", ".key"}:
            continue
        if path.stat().st_size > 5_000_000:
            errors.append(f"{relative}: exceeds scan size limit; review before committing")
            continue
        for kind in secret_kinds(path.read_bytes()):
            errors.append(f"{relative}: potential {kind} (value redacted)")
    for message in errors:
        print(message)
    print(f"Repository checks: {len(config_paths)} YAML configs, {len(skills)} skills; {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
