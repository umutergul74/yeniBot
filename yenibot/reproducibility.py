"""Reproducibility fingerprints for fitted Phase 1 artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any


TRAINING_CODE_RELATIVE_PATHS = (
    "yenibot/losses.py",
    "yenibot/models/__init__.py",
    "yenibot/models/hybrid.py",
    "yenibot/models/tcn.py",
    "yenibot/regime/__init__.py",
    "yenibot/regime/hmm.py",
    "yenibot/training/__init__.py",
    "yenibot/training/dataset.py",
    "yenibot/training/multitask.py",
    "yenibot/training/preprocessing.py",
    "yenibot/training/sample_weights.py",
    "yenibot/training/trainer.py",
    "yenibot/training/walk_forward.py",
    "yenibot/training/weight_averaging.py",
    "yenibot/experiment/training.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return ""


def training_code_signature_payload(
    *,
    root: str | Path | None = None,
    include_files: bool = True,
) -> dict[str, Any]:
    """Hash the source files that can change fitted model outputs."""

    base = Path(root).resolve() if root is not None else _repo_root()
    file_rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    missing: list[str] = []
    for relative in TRAINING_CODE_RELATIVE_PATHS:
        path = base / relative
        digest.update(relative.encode("utf-8"))
        if not path.exists():
            missing.append(relative)
            digest.update(b"<missing>")
            continue
        data = path.read_bytes()
        file_hash = _sha256_bytes(data)
        digest.update(file_hash.encode("utf-8"))
        file_rows.append(
            {
                "path": relative,
                "sha256": file_hash,
                "bytes": int(len(data)),
            }
        )
    payload: dict[str, Any] = {
        "signature_version": "training_code_v1",
        "code_hash": digest.hexdigest(),
        "tracked_file_count": len(TRAINING_CODE_RELATIVE_PATHS),
        "missing_files": missing,
    }
    if include_files:
        payload["files"] = file_rows
    return payload


def runtime_signature_payload(
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Capture the runtime knobs needed to interpret same-seed retrains."""

    torch_info: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": str(torch.version.cuda or ""),
            "cudnn_version": str(torch.backends.cudnn.version() or ""),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        }
        if cuda_available:
            current_device = torch.cuda.current_device()
            torch_info["cuda_device_name"] = str(torch.cuda.get_device_name(current_device))
            torch_info["cuda_device_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(current_device)
            )
            torch_info["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - defensive metadata only
        torch_info = {"available": False, "error": type(exc).__name__}

    payload: dict[str, Any] = {
        "signature_version": "runtime_v1",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "hmmlearn": _package_version("hmmlearn"),
        },
        "torch": torch_info,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "seed": seed,
        "deterministic_requested": deterministic,
        "device_requested": str(device or ""),
    }
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    payload["runtime_hash"] = digest
    return payload
