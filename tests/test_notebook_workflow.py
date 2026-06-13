from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_NAMES = [
    "00_phase1_auto_run.ipynb",
    "01_data_preparation.ipynb",
    "02_feature_engineering.ipynb",
    "03_labeling.ipynb",
    "04_training_walk_forward.ipynb",
    "05_diagnostics_validation.ipynb",
]
RESEARCH_BRANCH = "research/next-candidate-v1"


def _load_notebook(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_all_phase1_notebooks_pin_and_verify_the_research_branch() -> None:
    for name in NOTEBOOK_NAMES:
        source = _source(_load_notebook(name))
        assert RESEARCH_BRANCH in source, name
        assert "git', '-C', REPO_DIR, 'fetch', 'origin', REPO_BRANCH" in source, name
        assert "checkout', '-B', REPO_BRANCH" in source, name
        assert "assert repo_branch == REPO_BRANCH" in source, name
        assert "Repository commit:" in source, name
        assert "pull --ff-only" not in source, name


def test_research_notebooks_follow_the_post_oos_contract() -> None:
    combined = _source(_load_notebook("00_phase1_auto_run.ipynb"))
    training = _source(_load_notebook("04_training_walk_forward.ipynb"))
    diagnostics = _source(_load_notebook("05_diagnostics_validation.ipynb"))

    for source in (combined, training):
        assert "run_recency_ensemble_research" in source
        assert "notebook04_run.json" in source
        assert "Failed future OOS used for policy selection: False" in source or (
            "same_window_selection_allowed" in source
        )

    assert "Notebook 04 remains forbidden" not in diagnostics
    assert "notebook04_run.json" in diagnostics
    assert "notebook04_handoff" in diagnostics
    assert "Same-window policy selection allowed:" in diagnostics
    assert "future_oos_failure_summary.json" in diagnostics
    assert "future_oos_temporal_blocks.csv" in diagnostics
    assert "future_oos_regime_metrics.csv" in diagnostics
    assert "future_oos_ensemble_disagreement.csv" in diagnostics
    assert "future_oos_model_metrics.csv" in diagnostics
    assert "recency_ensemble_summary.csv" in diagnostics
    assert "recency_ensemble_by_fold.csv" in diagnostics
    assert "recency_ensemble_schedule.csv" in diagnostics
    assert "recency_ensemble_eligibility_audit.csv" in diagnostics
    assert "next_research_protocol.json" in diagnostics


def test_notebook_code_cells_are_valid_python_after_colab_magics_are_removed() -> None:
    for name in NOTEBOOK_NAMES:
        notebook = _load_notebook(name)
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            python_lines = [
                line
                for line in source.splitlines()
                if not line.lstrip().startswith(("!", "%"))
            ]
            compile(
                "\n".join(python_lines),
                f"{name}:cell-{index}",
                "exec",
            )
