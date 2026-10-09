import json
import subprocess

import pandas as pd
import pytest

from yenibot.notebook_runtime import initialize_workspace, publish_table, verified_table


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-m", "fixture"], cwd=root, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return root, sha


def test_workspace_refuses_code_environment_config_drift_and_legacy_data(tmp_path, repo):
    root, sha = repo
    base = tmp_path / "workspaces"
    args = (base, "candidate1", root, sha, {"seed": 42})
    workspace = initialize_workspace(*args, environment={"python": "3.11"})
    assert initialize_workspace(*args, environment={"python": "3.11"}) == workspace
    for environment, config in [({"python": "3.12"}, {"seed": 42}), ({"python": "3.11"}, {"seed": 43})]:
        with pytest.raises(ValueError, match="changed"):
            initialize_workspace(base, "candidate1", root, sha, config, environment=environment)
    with pytest.raises(ValueError, match="revision differs"):
        initialize_workspace(base, "candidate2", root, "0" * 40, {}, environment={})
    (root / "local.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        initialize_workspace(*args, environment={})
    (root / "local.py").unlink()
    (base / "legacy").mkdir()
    (base / "legacy" / "data").mkdir()
    with pytest.raises(ValueError, match="legacy"):
        initialize_workspace(base, "legacy", root, sha, {}, environment={})


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "a\\b"])
def test_workspace_name_cannot_escape_base(tmp_path, repo, name):
    root, sha = repo
    with pytest.raises(ValueError, match="research_id"):
        initialize_workspace(tmp_path, name, root, sha, {}, environment={})


def test_tables_are_verified_linked_and_never_replaced(tmp_path):
    raw = tmp_path / "raw.parquet"
    processed = tmp_path / "processed.parquet"
    frame = pd.DataFrame({"value": [1., 2.]})
    original = publish_table(frame, raw, provenance={"source": "synthetic"})
    assert publish_table(frame, raw, provenance={"source": "synthetic"}) == original
    child = publish_table(frame, processed, parents=[raw])
    assert child["parents"][0]["sha256"] == original["sha256"]
    pd.testing.assert_frame_equal(verified_table(processed), frame)
    before = raw.read_bytes()
    with pytest.raises(ValueError, match="Refusing to replace"):
        publish_table(frame + 1, raw)
    assert raw.read_bytes() == before
    raw.write_bytes(b"modified")
    with pytest.raises(ValueError, match="identity mismatch"):
        verified_table(raw)
    with pytest.raises(ValueError, match="identity mismatch"):
        publish_table(frame, tmp_path / "next.parquet", parents=[raw])
    assert not (tmp_path / "next.parquet").exists()


def test_unmanifested_interrupted_write_is_not_adopted(tmp_path):
    path = tmp_path / "raw.parquet"
    frame = pd.DataFrame({"value": [1.]})
    frame.to_parquet(path)
    before = path.read_bytes()
    with pytest.raises(FileNotFoundError):
        publish_table(frame, path)
    assert path.read_bytes() == before


def test_manifest_schema_or_row_mismatch_blocks_read(tmp_path):
    path = tmp_path / "raw.parquet"
    publish_table(pd.DataFrame({"value": [1.]}), path)
    manifest = path.with_suffix(".parquet.manifest.json")
    value = json.loads(manifest.read_text())
    value["rows"] = 99
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contract mismatch"):
        verified_table(path)
