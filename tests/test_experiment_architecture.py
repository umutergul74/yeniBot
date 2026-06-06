from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import yenibot.experiments as legacy_experiments
from yenibot.experiment import (
    experiment_settings,
    prepare_training_holdout_split,
    run_experiment_matrix,
    write_experiment_diagnostics,
)
from yenibot.experiment.execution import WorkflowJournal, traced_workflow, workflow_checkpoint


EXPERIMENT_PACKAGE = Path(__file__).parents[1] / "yenibot" / "experiment"


def test_legacy_experiments_module_is_a_small_compatibility_facade() -> None:
    facade = Path(legacy_experiments.__file__)

    assert len(facade.read_text(encoding="utf-8").splitlines()) <= 40
    assert legacy_experiments.experiment_settings is experiment_settings
    assert legacy_experiments.prepare_training_holdout_split is prepare_training_holdout_split
    assert legacy_experiments.run_experiment_matrix is run_experiment_matrix
    assert legacy_experiments.write_experiment_diagnostics is write_experiment_diagnostics


def test_experiment_modules_stay_below_the_monolith_guardrail() -> None:
    oversized = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in EXPERIMENT_PACKAGE.glob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 1800
    }

    assert not oversized, f"Split experiment modules before they exceed 1800 lines: {oversized}"


def test_experiment_module_dependency_graph_is_acyclic() -> None:
    module_names = {path.stem for path in EXPERIMENT_PACKAGE.glob("*.py")}
    graph: dict[str, set[str]] = {name: set() for name in module_names}
    for path in EXPERIMENT_PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            prefix = "yenibot.experiment."
            if node.module and node.module.startswith(prefix):
                dependency = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
                if dependency in module_names and dependency != path.stem:
                    graph[path.stem].add(dependency)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visited:
            return
        assert module not in visiting, f"Cyclic experiment dependency detected at {module}"
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_workflow_journal_records_completed_stage(tmp_path: Path) -> None:
    status_path = tmp_path / "workflow_status.json"
    journal = WorkflowJournal("unit_test", status_path)

    journal.checkpoint("load_inputs", rows=123)
    journal.complete()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["current_stage"] == "completed"
    assert payload["history"][-1]["stage"] == "load_inputs"
    assert payload["history"][-1]["rows"] == 123


def test_traced_workflow_records_failure_and_last_checkpoint(tmp_path: Path) -> None:
    status_path = tmp_path / "workflow_status.json"

    @traced_workflow("unit_test", lambda: status_path)
    def fail_after_checkpoint() -> None:
        workflow_checkpoint("compute_metrics", profile="control")
        raise ValueError("bad metric input")

    with pytest.raises(ValueError, match="bad metric input"):
        fail_after_checkpoint()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["current_stage"] == "compute_metrics"
    assert payload["error"]["type"] == "ValueError"
    assert "bad metric input" in payload["error"]["message"]
