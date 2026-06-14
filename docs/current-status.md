# Current Phase 1 Status

Last reviewed: **June 14, 2026**

## Decision

The current model passed the active `v4_evidence` walk-forward research
charter, but the frozen primary candidate failed its pre-registered future
unseen out-of-sample evaluation. Phase 2 remains blocked.

- **Model evidence passed** means the walk-forward research evidence is
  credible enough to justify a frozen confirmation test.
- **Phase 2 ready** requires the untouched frozen candidate to pass that
  confirmation test with no refitting or policy changes.

## Future-OOS Result

The 737-row window from `2026-05-13 09:00 UTC` through
`2026-06-13 01:00 UTC` was scored with zero fit operations and a verified
manifest hash.

| Metric | Future-OOS result |
|---|---:|
| Rank IC | `-0.0075` |
| PRAUC lift vs prevalence | `1.0267` |
| Precision lift vs prevalence | `1.0609` |
| F1 skill vs rate-matched random | `+0.0219` |
| Prediction-long rate | `79.38%` |
| Top-decile label lift | `0.9901` |
| Top-decile forward return | `-0.00166` |
| Selected forward return | `-0.00308` |

The candidate failed ranking, payoff, PRAUC-lift, and prediction-rate gates.
This is not a threshold-only miss. `control_fold_ensemble_v1` is retired.

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

## What Is Allowed Now

- Improve diagnostics, tests, documentation, CI, and operator safety.
- Use the failed OOS window for diagnosis only.
- Repair the retained control on historical purged walk-forward folds.
- Design a distinct pre-registered mechanism that addresses ranking reversal
  without repeating global deletion, hard reliability masking, or simple
  large-trade clipping.
- Add the failed window to future training data only after preserving its
  immutable evaluation record.
- Pre-register the replacement before collecting a new future-OOS window.

## Latest Mechanism Cycle

Bundle `20260614_054446` completed the train-only preprocessing experiment:

- Simple `4h_large_trade_ratio` clipping was controlled but not promotable:
  mean IC improved slightly while official F1 stayed flat and top-10 lift fell.
- Hard 4H-flow reliability masking changed 10 of 12 triage folds, increased
  Rank IC dispersion, reduced positive-fold coverage, and weakened economic
  concentration.
- Adding clipping did not repair the masking instability.

There is currently no active candidate profile and no active future-OOS
primary candidate. Notebook `04` should not be rerun until a distinct
pre-registered hypothesis is committed. Notebook `05` may be run for
diagnostic/report verification without GPU.

## What Is Frozen

- Candidate identity and source run
- Ordered feature list
- Model, scaler, and HMM artifacts
- Threshold and threshold source
- Fit cutoff and anchor
- Future-OOS gates

Do not modify the retired candidate's profile, threshold, weights, model
artifacts, or manifest hash. Do not choose replacement ensemble weights or
thresholds from the failed OOS window.

The retired manifest remains historical evidence in prior bundles and
`frozen_candidate_outcomes`. It is intentionally absent from the active
`frozen_candidates` slot. A new anchor and counter begin only after a
replacement is selected and pre-registered.
