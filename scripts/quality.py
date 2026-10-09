"""Bounded, offline project checks; run with the project Python interpreter."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("static", "quick", "full"), default="quick", nargs="?")
    parser.add_argument("--report", type=Path, help="Optional new JSON report (refuses overwrite)")
    parser.add_argument("--gpu", action="store_true", help="Also exercise available CUDA hardware")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report already exists; choose a new path")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if not args.gpu:
        # Empty CUDA_VISIBLE_DEVICES is unreliable with this Windows/CUDA build.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    commands = [
        ("lint", [sys.executable, "-m", "ruff", "check", "yenibot", "scripts", "tests"], 120),
        ("repository", [sys.executable, "scripts/check_repository.py"], 120),
    ]
    if args.scope != "static":
        command = [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"]
        if args.scope == "quick":
            command += ["-m", "not slow"]
        commands.append(("tests", command, 5400 if args.scope == "full" else 1800))
    results = []
    for name, command, timeout in commands:
        print(f"[{name}] starting", flush=True)
        start = time.monotonic()
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append({"check": name, "exit_code": code, "seconds": round(time.monotonic() - start, 3)})
        print(f"[{name}] exit={code}, seconds={results[-1]['seconds']}", flush=True)
        if code:
            break
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        packages = ("numpy", "pandas", "torch", "scikit-learn", "pytest", "ruff")
        payload = {
            "started_scope": args.scope,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "gpu_requested": args.gpu,
            "checks": results,
            "token_usage": None,
            "model_cost": None,
        }
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    return int(any(result["exit_code"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
