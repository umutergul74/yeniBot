# Phase 1 training temporal audit — 2026-10-13

## Findings and repairs

1. Train/validation purge was 24 bars, but validation/test embargo was 6 while
   labels and forward returns use 10 future bars. Enforce both gaps >= horizon
   in fold planning and check actual fold indices before fitting. Default embargo
   is now 10. Sequence length 64 delays first scored predictions, so the old
   configuration alone does not prove every scored test prediction leaked; short
   sequences expose the unsafe boundary directly.
2. Selection rows previously ended immediately before holdout. Their targets
   could consume holdout prices. Remove the last horizon rows from selection,
   keeping the reserved holdout window and frozen anchor unchanged. Recovery in
   a fresh diagnostics session respects the purged selection endpoint.
3. Guarded threshold selection and raw/calibrated candidate eligibility used test
   metrics. They now use validation metrics only; test performance remains an
   evaluation result and cannot switch the policy retrospectively.
4. Test-only prediction diagnostics could fall back to same-split oracle thresholds
   and later be called validation-selected. Official selection now requires
   validation provenance. Oracle diagnostics cannot pass that eligibility check.

Scaler and HMM fitting are train-only, HMM inference is forward filtering, early
stopping uses validation, and calibration fits validation. These inspected paths
did not show a fit-on-test defect. This is a bounded code audit, not proof of
absence of all leakage or of profitability.

Training/profile/diagnostic signatures are versioned forward so previous cached
results cannot silently satisfy repaired semantics. Keep old runs as historical
evidence, not clean independent performance claims. No holdout is reset or unlocked.

## Next experiment design

Do not start a large architecture/parameter search yet. TCN+GRU is the current
candidate, not an established best model. First establish a corrected baseline
with fixed data, feature list, horizon and evaluation budget. Compare a simple
linear classifier and a tree baseline against TCN+GRU under the same temporal
folds before increasing architecture complexity.

Choose hyperparameters on inner chronological validation only. Treat repeatedly
inspected walk-forward test results as development evidence, not untouched final
confirmation. Reserve genuinely unseen future data prospectively for final
confirmation; never reuse the failed historical holdout to optimize a policy.
Pre-register the search space, number of trials, primary metric and cost assumptions.
Use multiple seeds only after the design is fixed. Report fold dispersion and
label prevalence alongside PR-AUC/lift and rank IC; raw F1 is insufficient.

## Rerun boundary

The code/config identity changes. Start a new workspace with 01, then 02 and 03,
then run corrected 04 with the same full commit and exact cutoff. Shared raw
partitions can be reused; old workspace manifests and frozen model artifacts
must not be edited. Do not automatically run 05 or a one-shot holdout evaluation.
