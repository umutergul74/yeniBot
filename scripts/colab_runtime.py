"""Colab host bridge: mount Drive in Colab, execute research in a locked kernel.

This module deliberately imports no research packages into the host kernel.
It never installs, removes or checks packages in the Colab host environment.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "IPYTHONDIR", "JUPYTER_PATH"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def run_logged(command: list[str], log: Path, *, cwd: Path) -> None:
    """Stream both output channels, preserve errors, and fail before the next step."""
    with log.open("x", encoding="utf-8") as stream:
        print(f"Full setup log: {log}", flush=True)
        with subprocess.Popen(command, cwd=cwd, env=child_environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if code:
        raise RuntimeError(f"Environment command failed (exit {code}); see {log}")


def prepare(repository: Path, directory: Path, *, gpu: bool = False) -> Path:
    from scripts.verify_environment import verify

    verify(repository)
    directory = directory.absolute()
    python = directory / "bin/python"
    token = uuid.uuid4().hex
    report = directory.parent / f"yenibot_smoke_{token}.json"
    log = directory.parent / f"yenibot_setup_{token}.log"
    if directory.exists():
        # A failed install must not be silently repaired or adopted as ready.
        config = directory / "pyvenv.cfg"
        if not config.is_file() or not python.is_file():
            raise RuntimeError("Incomplete environment; preserve logs and use a fresh runtime")
        fields = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if fields.get("include-system-site-packages") != "false":
            raise RuntimeError("Research environment must exclude host site-packages")
        command = [str(python), "-u", str(repository / "scripts/environment_smoke.py"),
                   "--report", str(report)]
    else:
        command = [sys.executable, "-u", str(repository / "scripts/prepare_environment.py"),
                   "--directory", str(directory), "--report", str(report)]
    if gpu:
        command.append("--gpu")
    run_logged(command, log, cwd=repository)
    print(f"Verified environment report: {report}")
    return python


class ResearchKernel:
    """One persistent isolated kernel per notebook; errors block subsequent cells."""

    def __init__(self, python: Path, repository: Path, *, output_hook=None):
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager

        self.failed = False
        self.closed = False
        self.client = None
        self.output_hook = output_hook
        self.scratch = tempfile.TemporaryDirectory(prefix="yenibot-kernel-")
        scratch = Path(self.scratch.name)
        spec = scratch / "kernels/yenibot"
        spec.mkdir(parents=True)
        # Temporary, private specification: no registration or global kernel change.
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [str(python.absolute()), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "yeniBot isolated research", "language": "python",
        }), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[str(spec.parent)], ensure_native_kernel=False)
        self.manager = KernelManager(kernel_name="yenibot", kernel_spec_manager=manager,
                                     connection_file=str(scratch / "connection.json"),
                                     transport="tcp" if os.name == "nt" else "ipc",
                                     ip="127.0.0.1" if os.name == "nt" else str(scratch / "channel"))
        environment = child_environment()
        environment["IPYTHONDIR"] = str(scratch / "ipython")
        try:
            self.manager.start_kernel(cwd=str(repository), env=environment)
            self.client = self.manager.client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=90)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def execute(self, source: str):
        if self.failed or self.closed:
            raise RuntimeError("Research kernel stopped; rerun setup before any further cells")
        try:
            reply = self.client.execute_interactive(source, allow_stdin=False,
                                                    timeout=None, output_hook=self.output_hook)
            content = reply["content"]
            if content["status"] != "ok":
                raise RuntimeError(f"Research cell failed: {content.get('ename', 'error')}: "
                                   f"{content.get('evalue', 'see cell output')}")
            return None
        except BaseException:
            self.failed = True
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.manager.has_kernel:
                self.manager.shutdown_kernel(now=True)
        finally:
            if self.client is not None:
                self.client.stop_channels()
            self.scratch.cleanup()
            atexit.unregister(self.close)

    def finish(self, *, release: bool = False):
        if self.failed or self.closed:
            raise RuntimeError("Cannot finish a failed or closed research kernel")
        self.close()
        if release:
            from google.colab import runtime
            runtime.unassign()
