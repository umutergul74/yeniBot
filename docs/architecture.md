# yeniBot Architecture

## Design Goal

The codebase is organized so data leakage, experiment-policy mistakes, training
failures, and diagnostics failures can be located without reading one giant
module. Phase 1 remains the only implemented product boundary.

## Pipeline Layers

```text
notebooks
  -> yenibot.data
  -> yenibot.features
  -> yenibot.labeling
  -> yenibot.training
  -> yenibot.experiment
  -> yenibot.automation
```

- `data`: download and validate full Binance kline/futures context data.
- `features`: causal feature construction and deterministic profile selection.
- `labeling`: long-only triple-barrier labels and label-quality checks.
- `training`: model, losses, walk-forward splits, fold training, and HMM logic.
- `experiment`: experiment policy, orchestration, diagnostics, and artifacts.
- `automation`: report completeness review and Phase 1 readiness decisions.

No lower layer may import notebooks. Phase 1 code must not import backtest,
execution, trade-management, or live-service modules.

## Experiment Package

`yenibot/experiments.py` is a compatibility facade for old notebooks and tests.
New code imports public APIs from `yenibot.experiment`.

| Module | Responsibility |
|---|---|
| `common.py` | Deterministic serialization and shared metric primitives |
| `configuration.py` | Profile policy, experiment memory, preflight, signatures, manifests |
| `training.py` | Per-profile runs, summaries, promotion gates |
| `holdout.py` | Frozen holdout and future-OOS policy |
| `folds.py` | Fold stability forensics |
| `thresholds.py` | Threshold transfer and regime-threshold diagnostics |
| `rank_ic.py` | Rank-IC uncertainty and stability evidence |
| `classification.py` | Causal classification skill and validation-charter evidence |
| `separation.py` | Score-separation and bad-fold signatures |
| `drift.py` | Feature, score, probability, and reliability drift |
| `payoff.py` | Score-band payoff and frozen-policy robustness |
| `root_cause.py` | Phase 1 blocker diagnosis |
| `ensembles.py` | Seed audits, profile blends, deltas |
| `artifacts.py` | Slim/full bundle packaging |
| `execution.py` | Atomic workflow status and failure traces |
| `orchestration.py` | Top-level notebook-facing workflows |

The architecture test rejects circular imports and experiment modules over
1,800 lines. When a module approaches the limit, split by responsibility
instead of increasing the limit.

## Failure Localization

Notebook 04 writes:

```text
checkpoints/experiments/<run_id>/workflow_status.json
```

Notebook 05 writes:

```text
checkpoints/experiments/<run_id>/diagnostics_workflow_status.json
```

Each file is atomically replaced and records:

- workflow status: `running`, `completed`, or `failed`
- current stage
- timestamped checkpoint history
- stage context such as profile count or run id
- exception type, message, and compact traceback on failure

If Colab disconnects abruptly, status can remain `running`; `current_stage`
still identifies the interrupted phase.

## Change Rules

1. Keep notebook-facing orchestration thin; domain calculations belong in the
   relevant experiment module.
2. Preserve deterministic output schemas unless a migration is documented and
   tested.
3. Add a unit test beside a new diagnostic or policy rule.
4. Never silently skip a selected profile, unavailable fold, missing artifact,
   or failed report.
5. Keep configuration and experiment memory in `config.yaml`; do not hide
   research policy in source constants.
6. Run `ruff check` and the full pytest suite before pushing.
