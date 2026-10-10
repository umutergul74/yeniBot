"""Explicit lock regeneration; never run automatically during installation or CI."""
from pathlib import Path
import argparse
import json
import subprocess

from verify_environment import ROOT, text_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True, help="Isolated uv 0.13.0 executable")
    args = parser.parse_args()
    uv = str(args.uv.resolve())
    version = subprocess.check_output([uv, "--version"], text=True).strip()
    if version.split()[:2] != ["uv", "0.13.0"]:
        raise ValueError("Use the reviewed generator uv 0.13.0")
    lock = "requirements/locks/linux-py313.txt"
    subprocess.run([uv, "pip", "compile", "requirements-dev.txt", "--python-version", "3.13",
                    "--python-platform", "x86_64-unknown-linux-gnu", "--generate-hashes",
                    "--only-binary", ":all:", "--index-url", "https://pypi.org/simple",
                    "--output-file", lock, "--quiet"], cwd=ROOT, check=True)
    lines = ['schema = 1', f'lock = "{lock}"', 'generator = "uv 0.13.0"',
             'hash_normalization = "UTF-8 with LF line endings"', '', '[target]',
             'python = "3.13"', 'system = "Linux"', 'machine = "x86_64"', '', '[files]']
    for name in ("requirements.txt", "requirements-dev.txt", lock):
        lines.append(f'{json.dumps(name)} = "{text_hash(ROOT / name)}"')
    (ROOT / "requirements/environment.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
