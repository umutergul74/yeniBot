# Research evidence inventory — 2026-10-10

Read-only inspection of main config, protocol notes and selected local metadata.
No new data, model deserialization, training or final evaluation was performed.
Reported historical metrics are not new verification of the underlying artifacts.

| Protocol/evidence | Observed state | Consequence |
|---|---|---|
| Main legacy `blend_prob_mean_953a4ee825`, top_10 | Config records failed clean holdout, run `20260522_135424` | Historical benchmark only; no retuning on that holdout |
| Main primary `control_fold_ensemble_v1` | Config pins source `20260605_211102` and manifest `5c2acd94b3ea46a509b8a8c8b7327f6cc5ed34045a4e97e21a0b34b3313ae302` | Frozen candidate, not the failed legacy blend; eligibility still requires actual artifact verification |
| Main historical evidence | README documents June 2026 evidence under `v4_evidence` | Historical summary, not a fresh unseen-OOS pass |
| Advisor September 2026 | Local `output/advisor_experiments/final_review/review_summary.json` records one evaluation, 720 samples per seed (42/43/44), zero test fit operations | Treat September as seen; never call it untouched again for a new candidate |
| Advisor reservation docs/config | Still contain `reserved_not_evaluated` / historical not-run text | Stale relative to the local final-review record; do not infer readiness from that text |
| Future periods | Calendar time has advanced, but no exposure ledger or verified new-data manifest was provided | A date after an anchor is not sufficient proof of unseen data |

The advisor review JSON file was read as text only. Its observed file SHA-256 is
`f8e7c8b5129283fb4792a3c11b2cc9b54216aa5592547d0420123935e0ef5b82`.
It references freeze `291288dc7f66af79c937b93bd302e69fe0a0469e826db89e1cb0fda76ae19ad8`
and test frame `f290b1b69c9fbb6e50c3cc0cd58ec6f5c915866d5c615d9459c5a64bd5a76cec`.
Those referenced artifacts were not rehashed in this inventory. The advisory
branch remains separate, with pre-existing local edits preserved.

No `checkpoints/experiments` directory was found in the inspected original local
checkout. This is a local availability finding, not proof that the main frozen
artifacts are missing from Drive or another archive. Never reconstruct them by
retraining or silently copy advisor results into main's evidence.

## Ordered next actions

1. Validate the new locked Linux environment in CI, then run the dedicated
   synthetic Colab smoke (CPU first, an authorized GPU separately). Integrate
   the research notebooks with its isolated interpreter only after validation.
2. Locate the main frozen artifact directory for `20260605_211102`; perform the
   existing read-only preflight, checking actual hashes, fit cutoff and source
   identity. Do not run the one-shot evaluator during this inventory step.
3. Reconcile the advisor's stale reservation status in its own branch against
   reviewed evidence; do not conflate it with the main control candidate.
4. Write an experiment charter: eligible development intervals, seen-data ledger,
   baseline identity, primary validation metric, trial budget and fixed seed list.
   Pre-register the untouched evaluation boundary before opening its results.
5. Only then run a bounded single-factor development experiment. Historical
   failures already documented in SKILLS.md are not a new search space; leave
   Phase 1 gates and all holdout reservations intact.

There is no approved new parameter campaign or newly certified unseen interval
in this inventory. This document deliberately does not turn the observed
September results into a rule for selecting the next model.
