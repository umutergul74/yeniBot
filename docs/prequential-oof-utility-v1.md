# Prequential OOF utility v1

Status: protocol written before candidate fitting. August 30, 2026.
The goal remains unachieved; this is a retrospective probe, not clean confirmation.

## Mechanism and boundaries

The closed validation-payoff model expected +20.01 adverse-net bps per selected
test opportunity but observed -0.38 bps. Its validation source also selected the
base model checkpoint. That is verified reuse, not proof of the sole cause.
This distinct probe fits the second stage on previously completed **OOF test**
predictions, never the current fold's validation payoff or current/future test
outcomes. Expanding training history also changes regime mixture, so an improved
result would not isolate checkpoint-selection bias alone.

Only the already hash-pinned 2022-2025 full OOF cache is allowed. No rejected 2026
future-OOS/forward rows, new TCN+GRU training, exit grid, feature grid, seed search,
regularization search or utility-threshold selection. Previous failures stay closed.

## Locked temporal contract

- OOF folds 0 and 1 are fixed calibration warm-up (1,314 raw rows), not evaluation.
  One 657-row fold is too small for the chosen 1,000 mature-opportunity minimum,
  approximately the prior probe's 1,007 eligible validation opportunities.
- Evaluate **every fold 2-37**, including zero-trade folds. The common window is
  January 14, 2023 02:00 UTC-December 26, 2025 10:00 UTC: 23,652 decisions.
- Before each fold, fit once on all eligible strictly earlier OOF folds. Require
  the conservative outcome-close timestamp to precede that fold's first decision.
  No current-fold online refits, recency weights or window-length tuning.
- Keep source split `test` as provenance and separately mark its role as historical
  calibration training. Explicitly disclose use of earlier-fold test outcomes;
  only current/future-fold outcomes are forbidden in the fit.
- Outcomes use unchanged adverse-cost TP2/SL5/10-bar execution at next open.
  Censored/gap paths are excluded from training. Overlapping opportunity targets
  are not treated as independent trades or summed as portfolio returns.
- Candidate ridge: alpha 10, intercept, training-only standardization of score
  percentile, decision-bar ATR/close and their product. The percentile uses only
  the base model's prior validation score distribution, not validation payoffs.
- ATR-only control: identical training observations, target, alpha and action
  rule, with score and interaction columns forced to zero. It is a context
  benchmark, never an alternative automatically chosen for deployment.
- Enter only for strictly positive predicted adverse net utility. Zero, missing
  estimates or inadequate training imply no trade; inadequate history after
  the fixed warm-up is a failed data contract, not a reason to drop a fold.

## Same-cohort comparison and decision

Compare the candidate, ATR-only control, q80 and the archived validation-payoff
policy on exactly folds 2-37. Reuse the archived policy's hash-verified signals;
do not refit it or compare against its old 38-fold total. No-trade return is zero.

Candidate economic gates remain positive base AND adverse returns, >=100 completed
trades, >=2/3 positive folds, positive ranking/payoff diagnostics and complete
execution data. Require both ordinary score permutations and serial-preserving
controls (500 each). Those measure whole-policy timing; incremental TCN value
additionally requires beating the ATR-only context control.

For both q80 and ATR-only controls, require positive lower 95% paired mean-fold
return-difference bounds under base AND adverse costs, using both 3- and 6-fold
moving blocks with 5,000 replicates. All conditions are conjunctive; no favorable
interval or control is selected. The archived validation policy is descriptive
context only. Uncertainty is approximate with 36 historical folds.

If any requirement fails, close this one probe without retuning. Passing could
justify only a separately preregistered post-lock unseen confirmation with actual
funding, execution and risk checks. Historical profile selection still prevents
any claim that this is independent evidence or live readiness.

Machine contract: `configs/prequential_oof_utility_v1.json`. Fits, training-row
membership hashes, maturity bounds, actions, ledgers and source hashes must be
checkpointed. Tests must prove current/future changes cannot alter earlier fits
or decisions, while genuinely earlier OOF outcomes can influence subsequent fits.

Method references: [scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
uses cross-validated base predictions for the second stage; our chronology must
be stricter than ordinary random cross-validation. [Bailey et al.](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
explain why repeated historical selection still requires separate confirmation.

## Implementation checkpoint

Implementation: `yenibot.phase2.prequential_utility`; runner:
`yenibot.automation.phase2_prequential_utility`. Canonical protocol hash:
`91c673a964ecd2771b349bc8e522c6d87a3cd03634fd1fb914eb472a6291b568`.
The runner rejects protocol edits, changed source/reference hashes, changed
baseline execution/cost contracts, cohort mismatches and any nonempty output
directory. Windows/Unix JSON line endings do not alter the canonical protocol.
Source artifact byte hashes remain strict.

Real-source read-only preflight passed before fitting: 23,652 evaluation rows,
36 folds, same cohort for every reference. No training occurred in that check.
13 new tests cover temporal perturbations, exact maturity boundary exclusion,
fold-local exit indices against the execution engine, score-independent ATR
control, malformed clocks/splits/cohorts and refusal to overwrite results.
The 61 existing targeted tests also passed; one new assertion initially expected
an obsolete exception message and was corrected without changing the safeguard.

Run once after the implementation commit is pushed:

```powershell
.venv/Scripts/python -m yenibot.automation.phase2_prequential_utility `
  --scope-dir checkpoints/economic_attribution/20260628_155057/baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility/full `
  --report-dir reports/phase2_local_lab/final_third_wave_local/local_gate `
  --q80-dir reports/phase2_economic_attribution/20260830_full_oof_cdf_v1/validation_cdf_q80_fixed_atr_v1 `
  --validation-probe-dir reports/phase2_economic_attribution/20260830_net_utility_v1 `
  --output-dir reports/phase2_economic_attribution/20260830_prequential_oof_v1
```

Add `--preflight-only` for a read-only source/cohort check. Completed probes are
append-only: do not rerun them or create a new directory just to retune v1.
