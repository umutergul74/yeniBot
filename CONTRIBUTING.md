# Contributing

## Research First

yeniBot accepts changes that improve the validity, reproducibility, or
diagnostic power of the Phase 1 model pipeline. It does not accept backtesting,
execution, live-trading, XGBoost, or three-class label code while Phase 1 is
blocked.

Read `SKILLS.md`, `config.yaml`, and `docs/architecture.md` before editing the
pipeline.

## Before Writing Code

For a research change, document:

1. The failure mode or hypothesis.
2. Why it is distinct from rejected experiments in `config.yaml`.
3. The causal availability time of every new input.
4. The exact notebooks that must be rerun.
5. Whether training signatures or frozen artifacts change.
6. The metrics that would falsify the hypothesis.

Do not use an already-seen holdout to choose profiles, thresholds, score bands,
or blend weights.

## Development

```bash
python -m pip install -r requirements.txt
pytest -q
```

Keep changes within existing ownership boundaries:

- data acquisition and validation: `yenibot/data`
- causal features: `yenibot/features`
- labels: `yenibot/labeling`
- model training: `yenibot/training`
- evidence and experiment policy: `yenibot/experiment`
- automated readiness review: `yenibot/automation`

`yenibot/experiments.py` is a compatibility facade. New functionality belongs
in the focused modules under `yenibot/experiment`.

## Pull Requests

Pull requests should:

- isolate one research mechanism where possible
- keep all tunable values in `config.yaml`
- add focused tests
- preserve deterministic schemas or document migrations
- update experiment memory when an approach is rejected
- state whether notebooks `02`, `03`, `04`, or `05` must be rerun
- include no data, checkpoints, credentials, runtime state, or report bundles

Commit prefixes:

```text
feat:  new capability
fix:   correctness or leakage fix
data:  acquisition or validation pipeline
model: architecture or training behavior
docs:  documentation only
test:  test coverage
```

## Definition Of Done

A change is complete when:

- tests pass
- temporal assumptions are explicit
- no validation/test/holdout rows enter fit operations
- generated artifacts are complete or fail closed
- documentation matches the current validation charter
- Phase 2 boundaries remain intact
