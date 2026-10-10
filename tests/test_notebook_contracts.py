"""Exercise notebook boundaries offline; never start Colab, downloads or training."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0[0-5]_*.ipynb"))


def code(path, index):
    from scripts.harden_phase1_notebooks import remote_source
    source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][index]["source"])
    return remote_source(source) if index >= 6 else source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_compile_without_execution_outputs_and_rebuild_identically(path, tmp_path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:{index}", "exec")
            assert not cell["outputs"]
            assert cell["execution_count"] is None
    spec = importlib.util.spec_from_file_location("notebook_builder", ROOT / "scripts/harden_phase1_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    copy = tmp_path / path.name
    copy.write_bytes(path.read_bytes())
    module.update_notebook(copy)
    assert copy.read_bytes() == path.read_bytes()


def test_features_stop_before_build_when_configured_context_is_missing(tmp_path, monkeypatch):
    import yenibot.features

    def unexpected(*args, **kwargs):
        pytest.fail("Feature build must not run with missing configured input")

    monkeypatch.setattr(yenibot.features, "build_feature_matrix", unexpected)
    namespace = {"DATA_DIR": str(tmp_path), "os": os, "Path": Path,
                 "verified_table": lambda path: pd.DataFrame(),
                 "cfg": {"binance": {"intrabar_intervals": ["15m"]}}}
    with pytest.raises(FileNotFoundError, match="btc_15m"):
        exec(code(ROOT / "notebooks/02_feature_engineering.ipynb", 6), namespace)


def test_label_notebook_validates_configured_horizon(tmp_path, monkeypatch):
    import yenibot.labeling

    frame = pd.DataFrame({"label": [0, 1], "hit_type": ["time", "tp"], "fwd_return_24h": [0., .1]})
    monkeypatch.setattr(yenibot.labeling, "add_long_only_labels", lambda *a, **k: frame)
    seen = []

    def validate(result, **kwargs):
        assert kwargs["forward_return_column"] == "fwd_return_24h"
        seen.append("validated")

    monkeypatch.setattr(yenibot.labeling, "validate_label_quality", validate)
    settings = dict(atr_column="atr_14", tp_multiplier=2., sl_multiplier=5., max_holding_bars=24,
                    min_long_forward_return=0., max_not_long_forward_return=.01, min_long_pct=.2, max_long_pct=.8)
    namespace = {"DATA_DIR": str(tmp_path), "Path": Path, "cfg": {"labeling": settings},
                 "verified_table": lambda path: frame, "publish_table": lambda *a, **k: seen.append("published")}
    exec(code(ROOT / "notebooks/03_labeling.ipynb", 6), namespace)
    assert seen == ["validated", "published"]


def test_checkout_stops_after_failed_clone(tmp_path, monkeypatch):
    commands = []

    def fail(command, **kwargs):
        assert kwargs["check"] is True
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    namespace = {"REPO_COMMIT": "a" * 40, "REPO_DIR": tmp_path / "absent",
                 "REPO_URL": "unused.invalid", "sys": SimpleNamespace(modules={})}
    with pytest.raises(subprocess.CalledProcessError):
        exec(code(NOTEBOOKS[0], 3), namespace)
    assert len(commands) == 1
    assert commands[0][:2] == ["git", "clone"]


def test_diagnostics_uses_explicit_run_and_preserves_failure(tmp_path, monkeypatch):
    import yenibot.experiment

    releases = []
    google = ModuleType("google")
    google.colab = ModuleType("google.colab")
    google.colab.runtime = SimpleNamespace(unassign=lambda: releases.append(True))
    ipython = ModuleType("IPython")
    ipython.display = ModuleType("IPython.display")
    ipython.display.Image = lambda **kwargs: None
    ipython.display.display = lambda *args: None
    for name, module in {"google": google, "google.colab": google.colab,
                         "IPython": ipython, "IPython.display": ipython.display}.items():
        monkeypatch.setitem(sys.modules, name, module)
    run_id = "selected_run"
    (tmp_path / "experiments" / run_id).mkdir(parents=True)
    monkeypatch.setattr(yenibot.experiment, "future_oos_preflight", lambda **kwargs: {
        "state": "blocked", "data": {"fresh_labeled_rows": 0, "min_rows": 720}, "failed_checks": []})
    received = []

    def diagnostics(**kwargs):
        received.append(kwargs["run_id"])
        raise RuntimeError("synthetic diagnostics failure")

    monkeypatch.setattr(yenibot.experiment, "write_experiment_diagnostics", diagnostics)
    namespace = {"WORKSPACE": tmp_path, "CHECKPT_DIR": str(tmp_path), "re": re,
                 "EXPERIMENT_RUN_ID": run_id, "cfg": {}, "AUTO_UNASSIGN": True}
    with pytest.raises(RuntimeError, match="synthetic diagnostics failure"):
        exec(code(ROOT / "notebooks/05_diagnostics_validation.ipynb", 6), namespace)
    assert received == [run_id]
    assert releases == []
