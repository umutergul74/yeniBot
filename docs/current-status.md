# Current Phase 1 Status

Last reviewed: **June 11, 2026**

## Decision

The current model has passed the active `v4_evidence` research charter, but
Phase 2 remains blocked until the frozen primary candidate completes its
pre-registered future unseen out-of-sample evaluation.

- **Model evidence passed** means the walk-forward research evidence is
  credible enough to justify a frozen confirmation test.
- **Phase 2 ready** requires the untouched frozen candidate to pass that
  confirmation test with no refitting or policy changes.

## Frozen Contract

| Field | Frozen value |
|---|---|
| Candidate | `control_fold_ensemble_v1` |
| Source run | `20260605_211102` |
| Anchor | `2026-05-13 08:00:00+00:00` |
| Manifest SHA-256 | `5c2acd94b3ea46a509b8a8c8b7327f6cc5ed34045a4e97e21a0b34b3313ae302` |
| Threshold | `0.4236384365293715` |
| Threshold source | `validation_constrained_threshold` |
| Frozen models | `36` |
| Future fitting allowed | `false` |

The manifest hash has been independently recomputed from the latest reviewed
bundle and matches the configured expected hash.

## Latest Evidence

| Metric | Result |
|---|---:|
| Mean walk-forward Rank IC | `0.0734` |
| Positive-IC folds | `86.1%` |
| PRAUC lift vs prevalence | `1.124` |
| Precision lift vs prevalence | `1.101` |
| F1 skill vs rate-matched random | `+0.027` |
| Positive-return folds | `69.4%` |
| Top-decile OOS forward return | `0.00317` |
| Legacy Rank IC std monitor | `0.0708` |
| Legacy raw Long F1 monitor | `0.4313` |

Raw sigmoid scores are ranking scores, not deployment-ready probabilities.
The reliability evidence does not support probability-sized positions.

## What Is Allowed While Waiting

- Improve diagnostics, tests, documentation, CI, and operator safety.
- Refresh raw data, features, and labels with notebooks `01 -> 02 -> 03`.
- Run notebook `05` to inspect readiness and, once ready, perform frozen
  prediction-only evaluation.
- Fix code defects only when the frozen numerical contract remains unchanged
  and the change is covered by regression tests.

## What Is Frozen

- Candidate identity and source run
- Ordered feature list
- Model, scaler, and HMM artifacts
- Threshold and threshold source
- Fit cutoff and anchor
- Future-OOS gates

Do not run notebook `04` for the frozen evaluation. Do not change the profile,
threshold, weights, model artifacts, or manifest hash.

Readiness is determined by mature labeled rows, not by the calendar alone.
June 13 is an estimate; `future_oos_preflight.json` is authoritative.

