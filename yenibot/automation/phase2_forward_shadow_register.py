"""Register an exact sealed forward-shadow manifest from the pushed Git history."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pandas as pd

from yenibot.phase2.forward_shadow import (
    SHADOW_REGISTRATION_VERSION,
    _canonical_json,
    load_shadow_manifest,
    seal_shadow_registration,
    validate_shadow_registration,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def register_forward_shadow(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path,
    registry_path: str,
    output_path: str | Path,
    repo_dir: str | Path = ".",
) -> dict:
    """Prove that origin contains the exact manifest, then seal registration."""

    manifest_path = Path(manifest_path)
    output = Path(output_path)
    repo = Path(repo_dir).resolve()
    registry = PurePosixPath(registry_path)
    if registry.is_absolute() or ".." in registry.parts or not registry.parts:
        raise ValueError("Manifest registry path must be repository-relative")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a forward-shadow registration")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_shadow_manifest(manifest_path, artifact_root=artifact_root)
    if subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"], check=False
    ).returncode or subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False
    ).returncode:
        raise ValueError("Tracked repository changes must be committed before registration")
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    committed_text = _git(repo, "show", f"{head}:{registry.as_posix()}")
    committed_manifest = json.loads(committed_text)
    if _canonical_json(committed_manifest) != _canonical_json(raw_manifest):
        raise ValueError("Git does not contain the exact sealed manifest")
    remote_lines = _git(
        repo, "ls-remote", "origin", f"refs/heads/{branch}"
    ).splitlines()
    remote_commit = remote_lines[0].split()[0] if remote_lines else ""
    if remote_commit != head:
        raise ValueError("Exact manifest commit has not been pushed to origin")
    registered_at = pd.Timestamp.now(tz="UTC")
    payload = {
        "registration_version": SHADOW_REGISTRATION_VERSION,
        "process_id": manifest["process_id"],
        "candidate_id": manifest["candidate_id"],
        "block_id": manifest["block"]["block_id"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_git_commit": head,
        "manifest_registry_path": registry.as_posix(),
        "registered_at_utc": registered_at.isoformat(),
        "git_branch": branch,
        "git_remote": "origin",
        "remote_commit_verified": True,
    }
    registration = seal_shadow_registration(payload)
    validate_shadow_registration(registration, manifest=manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(registration, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return registration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-dir", default=".")
    args = parser.parse_args(argv)
    result = register_forward_shadow(
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        registry_path=args.registry_path,
        output_path=args.output,
        repo_dir=args.repo_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
