"""Synthetic regression tests for incremental storage and immutable research views."""
import json

import pandas as pd
import pytest

from yenibot.data.shared_store import prepare_shared_table
from yenibot.notebook_runtime import publish_table, table_exists, table_identity, verified_table


def source(calls):
    def download(start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        times = pd.date_range(start, end, freq="1h", inclusive="left")
        return pd.DataFrame({"timestamp": times, "value": [stamp.value / 1e9 for stamp in times]})
    return download


def prepare(tmp_path, name, start, end, calls, contract=None, **kwargs):
    path = tmp_path / "research" / name / "data/raw/btc_1h.parquet"
    prepare_shared_table(path, tmp_path / "raw_store", contract or {"dataset": "btc_1h", "version": 1},
                         start, end, source(calls), lambda frame, lo, hi: frame,
                         {"kind": "test"}, **kwargs)
    return path


def test_new_workspace_reuses_partitions_and_next_week_downloads_only_delta(tmp_path):
    calls = []
    first = prepare(tmp_path, "one", "2026-09-01", "2026-10-10", calls)
    assert len(calls) == 2  # full September and October through the cutoff
    before = verified_table(first)
    before_hash = table_identity(first)
    partition_bytes = {p: p.read_bytes() for p in (tmp_path / "raw_store").rglob("*.parquet")}
    second = prepare(tmp_path, "new_code", "2026-09-01", "2026-10-10", calls)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(verified_table(second), before)
    third = prepare(tmp_path, "next_week", "2026-09-01", "2026-10-17", calls)
    assert calls[-1] == (pd.Timestamp("2026-10-10", tz="UTC"), pd.Timestamp("2026-10-17", tz="UTC"))
    assert len(calls) == 3
    assert len(verified_table(third)) == len(before) + 7 * 24
    assert table_identity(first) == before_hash
    pd.testing.assert_frame_equal(verified_table(first), before)
    assert all(path.read_bytes() == data for path, data in partition_bytes.items())
    assert table_exists(first) and not first.exists()  # no duplicate parquet in any workspace
    assert not list((tmp_path / "research").rglob("*.parquet"))
    subset = prepare(tmp_path, "earlier_cutoff", "2026-09-05", "2026-10-05", calls)
    assert len(calls) == 3
    assert len(verified_table(subset)) == 30 * 24


def test_processed_parent_identity_pins_shared_snapshot_and_detects_corruption(tmp_path):
    raw = prepare(tmp_path, "one", "2026-10-01", "2026-10-10", [])
    processed = raw.parent.parent / "processed/features.parquet"
    record = publish_table(pd.DataFrame({"feature": [1.]}), processed, parents=[raw])
    assert record["parents"][0]["sha256"] == table_identity(raw)
    part = next((tmp_path / "raw_store").rglob("*.parquet"))
    part.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        verified_table(raw)
    with pytest.raises(ValueError, match="identity"):
        publish_table(pd.DataFrame({"feature": [2.]}), processed.with_name("next.parquet"), parents=[raw])


def test_failed_extension_does_not_modify_old_snapshot_and_can_resume(tmp_path):
    calls = []
    first = prepare(tmp_path, "one", "2026-09-01", "2026-10-10", calls)
    original = table_identity(first)
    new = tmp_path / "research/two/data/raw/btc_1h.parquet"
    downloads = source(calls)

    def fail_in_november(lo, hi):
        if pd.Timestamp(lo).month == 11:
            raise RuntimeError("Provider unavailable")
        return downloads(lo, hi)

    with pytest.raises(RuntimeError, match="Provider"):
        prepare_shared_table(new, tmp_path / "raw_store", {"dataset": "btc_1h", "version": 1},
                             "2026-09-01", "2026-11-08", fail_in_november,
                             lambda frame, lo, hi: frame, {})
    assert not table_exists(new)
    assert table_identity(first) == original
    completed = len(calls)
    prepare(tmp_path, "two", "2026-09-01", "2026-11-08", calls)
    assert len(calls) == completed + 1
    assert calls[-1][0] == pd.Timestamp("2026-11-01", tz="UTC")


def test_raw_policy_change_uses_a_separate_store_contract(tmp_path):
    calls = []
    prepare(tmp_path, "one", "2026-10-01", "2026-10-10", calls)
    prepare(tmp_path, "two", "2026-10-01", "2026-10-10", calls,
            contract={"dataset": "btc_1h", "version": 2})
    assert len(calls) == 2
    assert len(list((tmp_path / "raw_store").iterdir())) == 2


def test_manifest_tampering_and_missing_partition_stop_read(tmp_path):
    raw = prepare(tmp_path, "one", "2026-10-01", "2026-10-10", [])
    manifest = raw.with_suffix(".parquet.manifest.json")
    original = manifest.read_bytes()
    record = json.loads(original)
    record["end_exclusive"] = "2026-10-17T00:00:00+00:00"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        verified_table(raw)
    manifest.write_bytes(original)
    next((tmp_path / "raw_store").rglob("*.parquet")).unlink()
    with pytest.raises(FileNotFoundError):
        verified_table(raw)


def test_cross_partition_validation_runs_before_publishing_snapshot(tmp_path):
    def reject(frame):
        raise ValueError("Synthetic gap crossing partition boundary")

    with pytest.raises(ValueError, match="boundary"):
        prepare(tmp_path, "one", "2026-09-01", "2026-10-10", [], validate_complete=reject)
    assert not table_exists(tmp_path / "research/one/data/raw/btc_1h.parquet")


def test_legacy_parquets_remain_readable(tmp_path):
    path = tmp_path / "old.parquet"
    frame = pd.DataFrame({"x": [1, 2]})
    record = publish_table(frame, path)
    assert table_exists(path)
    assert table_identity(path) == record["sha256"]
    pd.testing.assert_frame_equal(verified_table(path), frame)
