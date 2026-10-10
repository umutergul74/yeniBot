import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_environment as contract


def fixture_contract(root):
    (root / "requirements").mkdir()
    lock = root / "requirements/lock.txt"
    lock.write_text("example==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    source = root / "requirements.txt"
    source.write_text("example>=1\n", encoding="utf-8")
    dev = root / "requirements-dev.txt"
    dev.write_text("-r requirements.txt\n", encoding="utf-8")
    text = 'schema = 1\nlock = "requirements/lock.txt"\n[target]\npython = "3.12"\nsystem = "Linux"\nmachine = "x86_64"\n[files]\n'
    for path in (lock, source, dev):
        text += f'{json.dumps(path.relative_to(root).as_posix())} = "{contract.text_hash(path)}"\n'
    (root / "requirements/environment.toml").write_text(text, encoding="utf-8")


def target(monkeypatch):
    monkeypatch.setattr(contract.sys, "version_info", SimpleNamespace(major=3, minor=12))
    monkeypatch.setattr(contract.platform, "system", lambda: "Linux")
    monkeypatch.setattr(contract.platform, "machine", lambda: "x86_64")


def test_installed_version_drift_fails(tmp_path, monkeypatch):
    fixture_contract(tmp_path)
    target(monkeypatch)
    monkeypatch.setattr(contract.metadata, "version", lambda name: "1.0")
    assert contract.verify(tmp_path, installed=True)["installed_verified"]
    monkeypatch.setattr(contract.metadata, "version", lambda name: "1.1")
    with pytest.raises(ValueError, match="dependency drift"):
        contract.verify(tmp_path, installed=True)


@pytest.mark.parametrize("path", ["requirements.txt", "requirements/lock.txt"])
def test_source_or_lock_change_requires_review(tmp_path, path):
    fixture_contract(tmp_path)
    (tmp_path / path).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        contract.verify(tmp_path, target=False)


def test_runtime_target_is_not_silently_overridden(tmp_path, monkeypatch):
    fixture_contract(tmp_path)
    target(monkeypatch)
    monkeypatch.setattr(contract.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="Unsupported runtime"):
        contract.verify(tmp_path)
    with pytest.raises(ValueError, match="requires target"):
        contract.verify(tmp_path, installed=True, target=False)


@pytest.mark.parametrize("bad", ["", "example>=1", "example==1", "--index-url https://unused.invalid"])
def test_unpinned_or_unhashed_lock_rejected(bad):
    with pytest.raises(ValueError):
        contract.pins(bad)


def test_hash_is_stable_across_git_newlines(tmp_path):
    path = tmp_path / "text"
    path.write_bytes(b"a\nb\n")
    expected = contract.text_hash(path)
    path.write_bytes(b"a\r\nb\r\n")
    assert contract.text_hash(path) == expected


def test_repository_contract_is_current():
    result = contract.verify(Path(__file__).resolve().parents[1], target=False)
    assert {"torch", "setuptools", "numpy", "pandas"} <= result["packages"].keys()


def test_cpu_smoke_does_not_use_market_data():
    from scripts.environment_smoke import smoke
    result = smoke()
    assert result["device"] == "cpu"
    assert result["synthetic_forward_backward"] == "passed"
    assert result["market_data_read"] is False
    assert result["holdout_evaluated"] is False


def test_colab_smoke_notebook_compiles_without_saved_outputs():
    notebook = json.loads((contract.ROOT / "notebooks/environment_smoke.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "environment_smoke", "exec")
            assert not cell["outputs"]
