# Frozen Future-OOS Runbook

## Purpose

Evaluate the pre-registered candidate on fresh mature labels without changing
the candidate, fitting any object, or using notebook `04`.

## Frozen Identity

Before doing anything, confirm:

```text
candidate_id: control_fold_ensemble_v1
source_run_id: 20260605_211102
manifest_hash: 5c2acd94b3ea46a509b8a8c8b7327f6cc5ed34045a4e97e21a0b34b3313ae302
threshold: 0.4236384365293715
anchor: 2026-05-13 08:00:00+00:00
```

If any value differs, stop. Do not regenerate a replacement candidate.

## Colab Sequence

1. `git pull`
2. `Runtime -> Restart session`
3. Run `01_data_preparation.ipynb`
4. Run `02_feature_engineering.ipynb`
5. Run `03_labeling.ipynb`
6. Run `05_diagnostics_validation.ipynb`

Do **not** run `04_training_walk_forward.ipynb`.

The data refresh extends raw, processed, and labeled rows. It must not replace
the frozen source-run artifacts.

## Preflight

Notebook 05 prints the read-only preflight before diagnostics. It can also be
run explicitly in Colab:

```bash
python -m yenibot.automation.future_oos_preflight \
  --config /content/yenibot_repo/config.yaml \
  --checkpoint-dir /content/drive/MyDrive/yeniBot/checkpoints
```

The command performs no writes and no fitting.

### Acceptable Waiting State

```text
state: waiting_for_mature_labeled_rows
invariants_passed: true
ready_for_evaluation: false
fit_operations_performed: 0
```

The date is not the gate. Label maturity and the configured minimum of 720
fresh labeled rows are the gate.

### Ready State

```text
state: ready_prediction_only
invariants_passed: true
ready_for_evaluation: true
fit_operations_performed: 0
```

Notebook 05 may then load the frozen artifacts for transform/predict only.

### Blocked State

```text
state: blocked_integrity_or_data_contract
invariants_passed: false
```

Stop immediately. Do not run training to repair it. Inspect:

- `failed_checks`
- `artifact_integrity_errors`
- `missing_frozen_feature_columns`
- source run, manifest hash, threshold, and fit cutoff

The evaluator is fail-closed and must not load models after a failed preflight.

## Required Output

The slim bundle must contain:

- `future_oos_preflight.json`
- `future_oos_preflight.md`
- `frozen_candidate_manifest.json`
- `frozen_candidate_index.csv`
- `future_oos_readiness.json`
- `future_oos_evaluation.csv`
- `future_oos_evaluation.json`
- `phase2_readiness.json`
- `auto_review.json`

When evaluation is complete, verify:

- `fit_operations_performed == 0`
- `no_refit_verified == true`
- manifest hash equals the frozen expected hash
- primary candidate is not substituted by an optional benchmark
- prediction timestamps are strictly after the anchor

## Decision

- `evaluated_passed`: the frozen evidence can unlock Phase 2 eligibility,
  subject to all other active charter checks.
- `evaluated_failed`: do not tune against this same future window. Record the
  failure and design a new pre-registered research cycle.
- `blocked_required_candidate`: repair the integrity or data-contract defect
  without changing the frozen candidate.

