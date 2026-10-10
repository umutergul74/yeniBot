"""Synthetic runtime check only: no market data, frozen artifacts or model selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_environment import verify  # noqa: E402


def smoke(*, gpu: bool = False) -> dict:
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import pandas as pd
    import torch
    from yenibot.models import HybridEncoder
    from yenibot.notebook_runtime import publish_table, verified_table

    torch.manual_seed(42)
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU explicitly requested but CUDA is unavailable")
    device = torch.device("cuda" if gpu else "cpu")
    model = HybridEncoder(3, seq_len=8, tcn_channels=4, tcn_dilations=[1],
                          gru_hidden=4, gru_layers=1, fusion_hidden=4, dropout=0.).to(device)
    x = torch.randn(2, 8, 3, device=device)
    logits = model(x, return_logits=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.], device=device))
    loss.backward()
    if not torch.isfinite(loss).item() or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    ):
        raise RuntimeError("Nonfinite synthetic model forward/backward result")
    with tempfile.TemporaryDirectory(prefix="yenibot-smoke-") as directory:
        path = Path(directory) / "synthetic.parquet"
        frame = pd.DataFrame({"value": [1., 2.]})
        publish_table(frame, path, provenance={"kind": "synthetic_smoke"})
        pd.testing.assert_frame_equal(verified_table(path), frame)
    return {"device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if gpu else None,
            "synthetic_forward_backward": "passed", "parquet_roundtrip": "passed",
            "market_data_read": False, "holdout_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Use an already authorized GPU runtime")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; preserve it and use a new name")
    environment = verify(installed=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    result = {"environment": environment, "checks": smoke(gpu=args.gpu),
              "python": sys.version.split()[0],
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Synthetic environment smoke passed; no research eligibility or GPU claim beyond the recorded device")
    print(json.dumps({"python": result["python"], "commit": result["commit"],
                      "lock_sha256_lf": environment["lock_sha256_lf"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
