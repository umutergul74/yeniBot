"""Create a NEW isolated Linux/Python 3.12 environment and run a synthetic smoke."""
from pathlib import Path
import argparse
import subprocess
import venv

from verify_environment import ROOT, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Use an already authorized GPU runtime")
    args = parser.parse_args()
    verify()
    if args.directory.exists() or args.report.exists():
        parser.error("Directory/report already exists; use new paths without replacing existing evidence")
    directory = args.directory.resolve()
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(directory)
    python = str(directory / "bin/python")
    subprocess.run([python, "-m", "pip", "--isolated", "install", "--require-hashes",
                    "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                    "-r", str(ROOT / "requirements/locks/linux-py312.txt")], check=True)
    command = [python, str(ROOT / "scripts/environment_smoke.py"), "--report", str(args.report.resolve())]
    if args.gpu:
        command.append("--gpu")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Verified isolated interpreter: {python}")


if __name__ == "__main__":
    main()
