from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from yenibot.training.advisor_colab import cache_frozen_inputs, create_review_bundle, freeze_market_inputs
from yenibot.training.advisor import exclusive_run


def test_frozen_inputs_reused_and_modified_copy_rejected(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for interval in (1, 4):
        frame = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=24//interval,
                                                       freq=f"{interval}h", tz="UTC"), "close": 100.})
        frame.to_parquet(source / f"btc_{interval}h.parquet")
    snapshot = tmp_path / "snapshot"
    kwargs = dict(start="2022-01-01T00:00:00Z", end="2022-01-01T23:00:00Z")
    first = freeze_market_inputs(source, snapshot, **kwargs)
    assert freeze_market_inputs(source, snapshot, **kwargs) == first
    cache_frozen_inputs(snapshot, tmp_path / "cache")
    (snapshot / "btc_1h.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        freeze_market_inputs(source, snapshot, **kwargs)


def test_review_bundle_includes_provenance_not_weights_or_training_frame(tmp_path):
    folder = tmp_path / "runs" / "A_test" / "seed_42" / "fold_000"
    folder.mkdir(parents=True)
    (folder / "validation_metrics.json").write_text('{"test_evaluations":0}')
    (folder / "last.pt").write_bytes(b"weights")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "development.parquet").write_bytes(b"large data")
    path = create_review_bundle(tmp_path, {"commit": "test", "status": "interrupted"})
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert any(name.endswith("validation_metrics.json") for name in names)
        assert not any(name.endswith(".pt") or name.endswith("development.parquet") for name in names)
        assert json.loads(archive.read("review_manifest.json"))["session"]["status"] == "interrupted"


def test_runtime_lock_is_separate_and_prevents_duplicate_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVISOR_LOCK_ROOT", str(tmp_path / "runtime_locks"))
    run = tmp_path / "drive" / "runs" / "A_test"
    with exclusive_run(run):
        assert not (run / "run.lock").exists()
        with pytest.raises(OSError):
            with exclusive_run(run):
                pass
    with exclusive_run(run):
        pass  # Released locks must allow resume, even after an exception.
