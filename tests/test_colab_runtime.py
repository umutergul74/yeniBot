"""Synthetic kernel/host boundary tests: no downloads, Drive or real training."""
import json
from pathlib import Path
import sys
import subprocess

import pytest

from scripts import colab_runtime as runtime
from scripts.harden_phase1_notebooks import INSTALL, remote_source, wrap_remote


def test_child_environment_excludes_injected_python_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/host/incompatible-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    environment = runtime.child_environment()
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_existing_environment_is_checked_without_host_pip(tmp_path, monkeypatch):
    from scripts import verify_environment
    monkeypatch.setattr(verify_environment, "verify", lambda repo: {})
    directory = tmp_path / "env"
    (directory / "bin").mkdir(parents=True)
    (directory / "bin/python").touch()
    (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    calls = []
    monkeypatch.setattr(runtime, "run_logged", lambda command, *a, **k: calls.append(command))
    python = runtime.prepare(tmp_path, directory)
    assert calls[0][0] == str(python)
    assert "environment_smoke.py" in calls[0][2]
    (directory / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(RuntimeError, match="exclude host"):
        runtime.prepare(tmp_path, directory)
    assert len(calls) == 1


def test_setup_failure_preserves_combined_log_and_stops(tmp_path, capsys):
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="exit 3"):
        runtime.run_logged([sys.executable, "-c", "import sys; print('root cause', file=sys.stderr); sys.exit(3)"],
                           log, cwd=tmp_path)
    assert "root cause" in log.read_text()
    assert "root cause" in capsys.readouterr().out


def test_notebook_installer_never_changes_host_packages():
    assert '"pip"' not in INSTALL
    assert "prepare(REPO_DIR, ENV_DIR" in INSTALL
    assert "ResearchKernel(ISOLATED_PYTHON" in INSTALL


def test_wrapper_preserves_code_with_quotes_and_backslashes():
    source = 'value = """line\\nquoted"""\nprint(value)\n'
    assert remote_source(wrap_remote(source)) == source


def test_real_kernel_retains_state_renders_output_and_stops_on_failure(tmp_path, monkeypatch):
    # Windows development environment can omit notebook tools. Linux CI must run this test.
    if sys.platform == "win32":
        pytest.importorskip("jupyter_client")
        pytest.importorskip("ipykernel")
    else:
        __import__("jupyter_client")
        __import__("ipykernel")
    hostile = tmp_path / "host-only"
    hostile.mkdir()
    (hostile / "host_only_dependency.py").write_text("raise AssertionError('host contamination')")
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    # Colab's machine-wide config selects a class absent from the isolated venv.
    bad_config = tmp_path / "colab_config.py"
    bad_config.write_text("c = get_config()\nc.IPKernelApp.kernel_class = 'google.colab._kernel.Kernel'\n")
    baseline = subprocess.run([sys.executable, "-m", "ipykernel_launcher", "--config=" + str(bad_config)],
                              env=runtime.child_environment(), capture_output=True, text=True, timeout=20)
    assert baseline.returncode != 0
    assert "google.colab._kernel.Kernel" in baseline.stderr
    from jupyter_client import KernelManager
    original_start = KernelManager.start_kernel
    def start_with_colab_config(manager, **kwargs):
        return original_start(manager, extra_arguments=["--config=" + str(bad_config)], **kwargs)
    monkeypatch.setattr(KernelManager, "start_kernel", start_with_colab_config)
    messages = []
    python = Path(sys.executable)
    kernel = runtime.ResearchKernel(python, tmp_path, output_hook=messages.append)
    try:
        kernel.execute("assert type(get_ipython().kernel).__module__ == 'ipykernel.ipkernel'\nimport importlib.util; assert importlib.util.find_spec('host_only_dependency') is None\nvalue = 40")
        result = tmp_path / "result.json"
        kernel.execute("import sys, json; from pathlib import Path\n"
                       f"Path({str(result)!r}).write_text(json.dumps({{'value': value + 2, 'python': sys.executable}}))\n"
                       "from IPython.display import display, HTML; display(HTML('<b>kernel output</b>'))")
        record = json.loads(result.read_text())
        assert record["value"] == 42
        assert Path(record["python"]).samefile(python)
        assert any(m["msg_type"] == "display_data" and "text/html" in m["content"]["data"] for m in messages)
        if sys.platform != "win32":
            kernel.execute("%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.plot([1, 2]); plt.show()")
            assert any(m["msg_type"] == "display_data" and "image/png" in m["content"]["data"] for m in messages)
        with pytest.raises(RuntimeError, match="synthetic failure"):
            kernel.execute("raise ValueError('synthetic failure')")
        with pytest.raises(RuntimeError, match="stopped"):
            kernel.execute("value = 0")
        with pytest.raises(RuntimeError, match="failed or closed"):
            kernel.finish(release=True)
        assert not kernel.manager.is_alive()
    finally:
        kernel.close()


def test_no_research_imports_in_host_domain_cells():
    import ast
    root = Path(__file__).resolve().parents[1]
    for path in (root / "notebooks").glob("0[0-5]_*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"][6:]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
