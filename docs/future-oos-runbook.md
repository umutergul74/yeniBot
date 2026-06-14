# Frozen Future-OOS Runbook

## Purpose

Evaluate the pre-registered candidate on fresh mature labels without changing
the candidate, fitting any object, or using notebook `04`.

## Historical Frozen Identity

Before doing anything, confirm:

```text
candidate_id: control_fold_ensemble_v1
source_run_id: 20260605_211102
manifest_hash: 5c2acd94b3ea46a509b8a8c8b7327f6cc5ed34045a4e97e21a0b34b3313ae302
threshold: 0.4236384365293715
anchor: 2026-05-13 08:00:00+00:00
```

If any value differs, stop. Do not regenerate a replacement candidate.

This candidate was evaluated and retired on June 13, 2026. The values above
are historical verification data, not the active candidate configuration.
The active frozen slot now reports
`awaiting_replacement_preregistration`. Do not run future-OOS evaluation until
a new primary candidate, source run, manifest hash, threshold, and anchor have
all been committed.

## Colab Sequence For A New Frozen Candidate

1. `git pull`
2. `Runtime -> Restart session`
3. Run `01_data_preparation.ipynb`
4. Run `02_feature_engineering.ipynb`
5. Run `03_labeling.ipynb`
6. Run `05_diagnostics_validation.ipynb`

Do **not** run `04_training_walk_forward.ipynb` after the new candidate is
frozen. Before a replacement has been selected, notebook `04` remains a
historical walk-forward research tool only.

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

### Awaiting Replacement State

```text
state: awaiting_replacement_preregistration
ready_for_evaluation: false
fit_operations_performed: 0
```

This is the expected state after a failed candidate is retired and before a
replacement is frozen. It is not an artifact-integrity error. Select the
replacement on historical walk-forward evidence, commit its manifest and new
anchor, then begin counting genuinely unseen rows.

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

## June 13, 2026 Outcome

`control_fold_ensemble_v1` completed the required evaluation on 737 mature
labeled rows. Artifact integrity and no-refit checks passed, but the candidate
failed the pre-registered evidence gates. Its status is
`retired_after_failed_future_oos`.

The next cycle must:

1. Preserve this manifest and evaluation unchanged.
2. Use the failed window for root-cause diagnosis only.
3. Select rolling/recency policies on historical rolling-origin data.
4. Freeze a replacement candidate before a new future-OOS anchor.
5. Never tune the replacement threshold or weights against this failed window.
