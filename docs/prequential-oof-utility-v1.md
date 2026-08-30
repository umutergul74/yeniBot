# Prequential OOF utility v1

Status: completed August 30, 2026; historical returns improved, but the locked
conjunctive gate FAILED. Probe CLOSED to retuning. The original pre-fit protocol
below is retained. The goal remains unachieved; this is not clean confirmation.

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

Original command (now completed; do not rerun):

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

## Completed evidence and active checkpoint

Implementation/protocol commit `1ea6768` was pushed BEFORE the actual fit.
Exec session `54897`, worker PID `23972`, completed with exit code 0. There is no
active training/evaluation wait. The separate read-only reconstruction audit
(session `47184`) also completed with exit code 0 and performed no extra fits.

The output directory is
`reports/phase2_economic_attribution/20260830_prequential_oof_v1`.
There were 36 candidate and 36 ATR-only payoff fits, no TCN+GRU or archived-policy
refits. All 36 training membership hashes and strict maturity constraints were
reconstructed successfully. Saved training scalers and intercepts matched; ridge
normal-equation maximum residual was 1.14e-13; saved prediction error was zero.
Independent itemized price/cost calculations matched ledger returns to 1.15e-16.
Every evaluated policy used the same 23,652 decisions, folds 2-37.

| Policy, same 36-fold cohort | Base return | Adverse return | Trades | Base trade-close DD |
|---|---:|---:|---:|---:|
| Past-OOF utility candidate | +149.01% | +80.65% | 315 | -19.00% |
| ATR-only matched control | +56.39% | +35.56% | 140 | -10.30% |
| Fixed q80 | +20.68% | -60.13% | 1,107 | -43.31% |
| Archived validation-payoff policy | +22.02% | -45.59% | 803 | -40.57% |

These are completed-trade compounded returns across independent historical
folds spanning January 2023-December 2025, NOT annualized returns or a continuous
live portfolio simulation. Censored positions are excluded and drawdown is
trade-close only. There are no leverage or real execution guarantees.

Candidate base/adverse profit factors were 1.517/1.320; mean net trade returns
30.68/20.47 bps. Adverse trade-close drawdown was -22.58%. Three excluded terminal
marks summed to -0.812% base/-0.977% adverse (not compounded portfolio returns;
no hypothetical exit costs charged). ATR-only adverse PF was 1.327 and mean net
24.28 bps, with -12.20% DD. More candidate total return does NOT establish better
per-trade quality or equal-risk efficiency than the context-only policy.

The candidate had 23 positive, seven negative and six no-trade base folds.
23/36 = 63.89% is below the unchanged two-thirds requirement (24/36).
Adverse folds: 21 positive, nine negative, six no-trade. Neither no-trade folds
nor losses are dropped. Both conditional nulls had p = 0.001996 (500 each,
Monte Carlo floor), but this concerns whole-policy timing, not isolated TCN value.
Serial-null mean turnover was 327.53 versus 315 actual trades.

Paired mean-fold deltas and 95% intervals below are **percentage points**, not
whole-period compounded-return differences. All pre-specified tests were required.

| Candidate minus control | Mean delta | 3-fold-block interval | 6-fold-block interval |
|---|---:|---:|---:|
| q80, base | +1.917 | [-0.272, +4.627] | [-0.278, +4.692] |
| q80, adverse | +4.022 | [+1.869, +6.652] | [+1.815, +6.755] |
| ATR-only, base | +1.444 | [+0.023, +2.954] | [+0.408, +3.249] |
| ATR-only, adverse | +0.927 | [-0.485, +2.390] | [+0.050, +2.569] |

Three original criteria failed: fold breadth, all paired q80 conditions, and all
paired ATR-only conditions. Do not choose only the favorable cost or block length.
The family is CLOSED despite the improved aggregate returns; no threshold,
alpha, history window, exit or gate changes are permitted to rescue this result.

Read-only concentration checks: base/adverse completed returns were +7.22/+3.84%
in the 2023 slice (31 trades), +60.21/+39.48% in 2024 (136) and +44.96/+24.73%
in 2025 (148). Removing the five largest winners still left +103.60/+48.37%;
removing the best fold left +108.81/+54.73%. These are descriptive robustness
checks after seeing the result, not additional independent successful trials.

Payoff calibration did transfer descriptively in this historical probe:
1,712 selected eligible overlapping opportunities had predicted/observed
adverse payoff +11.81/+19.70 bps; completed trades +8.02/+20.47 bps. No large-N
significance is inferred from overlapping opportunities. The change also expanded
and changed the training history; validation reuse is not proved the sole cause
of the earlier failure. Historical profile-selection bias remains intact.

### Resume here

1. Keep all earlier failures and this failed-gate result immutable. Do not rerun
   the completed trial, turn it into an official winner, or tune its parameters.
2. The next bounded work is an **execution/risk realism audit of the frozen
   ledgers**, not a new entry-policy search: intratrade marked equity and terminal
   positions, actual historical funding availability, and candidate versus
   ATR-only exposure/risk concentration. First specify the accounting assumptions;
   retain original returns alongside any accounting-reconciled diagnostic.
3. Evidence of promising historical utility now exists, but robust incremental
   TCN benefit and clean confirmation remain unproved. Never equate +149% with
   live readiness or the project's final goal. Any future promotion still requires
   a separately locked, genuinely unseen confirmation and execution/risk checks.
4. No notebook run or live trading is required for the next local audit.

Result SHA-256:
`41237d1a3298276b60add605c90cc3d4cef5f63bf2716c16e50f95516d560524`.
Fits SHA-256:
`f0b66218d53b32daab7807d97ab4cee56f7d110799be64e34260f0bc66515ae4`.
Candidate signals SHA-256:
`cbef47bf8150d61bb10fa1761b6a14f2dd8a987c0da3bbe876d4362ee162dcf1`.
The root result pins all original artifacts. An appended
`post_run_integrity_audit.json` records the later no-refit checks separately.
