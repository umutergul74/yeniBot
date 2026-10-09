"""Print installed project dependency closure; no credentials or local paths.

This is a host-specific inventory, not a cross-platform lockfile. Pipe to a new
file for reproduction/audit. --audit strips wheel local tags for advisory lookup.
"""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def collect(requirements: list[str]) -> dict[str, str]:
    roots = [Requirement(line) for line in requirements]
    pending = [item for item in roots if item.marker is None or item.marker.evaluate()]
    visited: set[tuple[str, tuple[str, ...]]] = set()
    versions = {}
    while pending:
        request = pending.pop()
        name = canonicalize_name(request.name)
        key = (name, tuple(sorted(request.extras)))
        distribution = metadata.distribution(name)
        version = distribution.version
        if request.specifier and not request.specifier.contains(version, prereleases=True):
            raise ValueError(f"Installed {name} does not satisfy its dependency constraint")
        if key in visited:
            continue
        visited.add(key)
        versions[name] = version
        for raw in distribution.requires or []:
            dependency = Requirement(raw)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra}) for extra in {"", *request.extras}
            ):
                pending.append(dependency)
    return dict(sorted(versions.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    versions = collect([line for line in lines if line and not line.startswith("#")])
    print("# Host-specific installed project dependency closure; not a portable lock.")
    for name, version in versions.items():
        print(f"{name}=={Version(version).public if args.audit else version}")


if __name__ == "__main__":
    main()
