"""Verify the target-specific dependency contract without installing packages."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def text_hash(path: Path) -> str:
    """Canonical UTF-8/LF hash, stable across Windows Git line-ending conversion."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def pins(text: str) -> dict[str, str]:
    result = {}
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)==([^\s;]+)(?:\s+--hash=sha256:[0-9a-f]{64})+", line)
        if match is None:
            raise ValueError("Lock must contain only exact pins with SHA-256 hashes")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in result:
            raise ValueError(f"Duplicate lock dependency: {name}")
        result[name] = match[2]
    if not result:
        raise ValueError("Dependency lock is empty")
    return result


def verify(root: Path = ROOT, *, installed: bool = False, target: bool = True) -> dict:
    contract = tomllib.loads((root / "requirements/environment.toml").read_text(encoding="utf-8"))
    if contract.get("schema") != 1:
        raise ValueError("Unsupported environment contract")
    required_files = {"requirements.txt", "requirements-dev.txt", contract["lock"]}
    if not required_files <= contract["files"].keys():
        raise ValueError("Contract must hash the lock and both source requirements")
    for name, expected in contract["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or text_hash(path) != expected:
            raise ValueError(f"Environment input changed: {name}; regenerate and review the lock")
    lock_path = contract["lock"]
    if lock_path not in contract["files"]:
        raise ValueError("Lock file must be included in the contract hashes")
    dependencies = pins((root / lock_path).read_text(encoding="utf-8"))
    expected_target = contract["target"]
    if target:
        actual = {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
                  "system": platform.system(), "machine": platform.machine()}
        if actual != expected_target:
            raise ValueError(f"Unsupported runtime: {actual}; required: {expected_target}")
    if installed:
        if not target:
            raise ValueError("Installed verification requires target verification")
        for name, expected in dependencies.items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError as exc:
                raise ValueError(f"Missing locked dependency: {name}") from exc
            if actual != expected:
                raise ValueError(f"Locked dependency drift: {name}: {actual} != {expected}")
    return {"schema": 1, "target": expected_target, "packages": dependencies,
            "lock_sha256_lf": contract["files"][lock_path], "installed_verified": installed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    result = verify(installed=args.installed, target=not args.structure_only)
    print(json.dumps({"target": result["target"], "package_count": len(result["packages"]),
                      "installed_verified": result["installed_verified"]}, indent=2))
