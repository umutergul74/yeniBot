from types import SimpleNamespace

import pytest

from scripts import environment_snapshot as snapshot


def test_snapshot_follows_transitive_and_extra_dependencies_without_unrelated_packages(monkeypatch):
    packages = {
        "root": SimpleNamespace(version="1.0", requires=['child[plot]>=2', 'unused; python_version < "3"']),
        "child": SimpleNamespace(version="2.0", requires=['plot; extra == "plot"', 'root>=1']),
        "plot": SimpleNamespace(version="3.0", requires=[]),
    }
    monkeypatch.setattr(snapshot.metadata, "distribution", packages.__getitem__)
    assert snapshot.collect(["root>=1", 'unused; python_version < "3"']) == {
        "child": "2.0", "plot": "3.0", "root": "1.0"
    }


def test_snapshot_does_not_hide_a_conflicting_revisited_constraint(monkeypatch):
    packages = {
        "root": SimpleNamespace(version="1.0", requires=["child>=3", "child>=2"]),
        "child": SimpleNamespace(version="2.0", requires=[]),
    }
    monkeypatch.setattr(snapshot.metadata, "distribution", packages.__getitem__)
    with pytest.raises(ValueError, match="does not satisfy"):
        snapshot.collect(["root"])
