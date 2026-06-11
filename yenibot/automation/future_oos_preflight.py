"""Command-line entry point for the read-only future-OOS preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from yenibot.config import load_config
from yenibot.experiment.oos_preflight import future_oos_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect frozen future-OOS readiness without fitting or writing artifacts."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or config["paths"]["checkpoint_dir"]
    result = future_oos_preflight(
        checkpoint_dir=Path(checkpoint_dir),
        config=config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["invariants_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
