# Conditional score attribution v1

Status: protocol locked before any conditional-control performance calculation,
August 30, 2026. No new candidate, threshold selection, training or promotion.

## Question and immutable source

Does the frozen past-OOF utility policy obtain incremental historical economic
information from the TCN+GRU score beyond its ATR context? Earlier whole-utility
shuffles also destroyed the ATR effect. The context-only control is profitable,
and the candidate has 2.229 times its exposure, so that distinction matters.

Pin the completed probe result to
`41237d1a3298276b60add605c90cc3d4cef5f63bf2716c16e50f95516d560524` and verify
every referenced artifact. Use original folds 2-37, 23,652 decisions and the
same historical bars. Reconstruct all saved utility outputs from the frozen fits
before evaluating. No 2026 failed OOS/forward rows or altered old outcomes.

## Fixed control panel

- Two conditioning resolutions, both mandatory: ratio 1.10 and ratio 1.05.
  Stratum = `(fold, UTC calendar month, floor(log(ATR/close)/log(ratio)))`.
  The dimensionless grid is fixed, not estimated from future data or returns.
- Within each stratum, move ONLY `frozen_score_percentile`. Keep the recipient
  row's ATR/close fixed, recompute the score*ATR interaction, and apply its original
  fold's saved ridge coefficients/scalers. Never shuffle the already combined
  utility score. No nuisance/model fit is needed.
- Ordinary within-stratum permutations and within-stratum circular shifts are
  both required, 500 draws each for both grids. Circular shifts include zero.
  There is no seed search. Seed = 20260830 + 1000*grid_index + 100*method_index,
  grid order `[1.10, 1.05]`, method order `[ordinary, circular]`.
- Score marginal distributions and score/ATR-bin dependence are preserved.
  Donor/recipient ATR ratios cannot exceed the specified 1.10/1.05 bound.
  Continuous score/ATR dependence is only approximated, not perfectly preserved.
- Circular shifts preserve the cyclic chronological score order **inside each
  irregularly sampled stratum**, not the entire hourly serial process. Quantify
  raw-score and action lag-one correlations within folds, changed-row fractions,
  selected-row counts and executed turnover. Do not claim exact temporal nulls.
- Singleton/constant-score groups stay uninformative; retain all such rows and
  report their counts. No data-driven merging or dropping groups. Require >=90%
  of originally selected rows to have at least two distinct scores in their group.
  Covariate-only preflight found 98.33% (10% grid) and 96.09% (5% grid), before any
  new control returns were calculated. No outcome-based bin choice was made.

## Execution and interpretation

Reuse the reference-validated independent-fold execution cache, next-open fills,
TP2/SL5/10-bar exits, stop-first and non-overlapping positions. Use BOTH unchanged
original base and adverse cost schedules; funding here is the original fixed
rate, not the separate historical-funding sensitivity. Censored trades remain
excluded from completed-return statistics. Validate actual and the first draw
of every grid/method/cost against the full engine. Cache mean-net-return is an
additive statistic only; old return/trade-selection semantics cannot change.

Two co-primary statistics: completed-trade compounded return AND mean net return
per completed trade. The latter limits interpretation driven solely by turnover;
it does not make unequal exposures risk-equivalent. Compute conservative
add-one upper-tail Monte Carlo probabilities, including numerical ties. All
16 combinations (2 grids*2 methods*2 costs*2 statistics) must be <=0.05 for this
one conditional diagnostic to pass. Use the maximum p, never select a favorable
grid, cost or statistic. This conjunction does not remove global historical
profile/policy-selection bias, approximate exchangeability or adaptive research.

Even a pass is retrospective conditional sensitivity evidence, NOT causal proof,
high future model accuracy, official candidate acceptance or live readiness.
The prior failed breadth/paired-comparison gates remain failed. A failure closes
this diagnostic without retuning. Record every trial and resume only the same
pinned computation; do not launch repeats to obtain a better outcome.

References: [Strobl et al., conditional variable importance](https://link.springer.com/article/10.1186/1471-2105-9-307)
motivates preserving correlated covariates when assessing incremental information;
our fixed stratification is not their forest-derived scheme or an exact time-series
test. [Scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
is model-specific inspection, not proof of intrinsic feature value.

## Run/recovery contract

Implementation: `yenibot.phase2.conditional_attribution`; runner:
`yenibot.automation.phase2_conditional_attribution`. The cache gains only a mean
completed-net-return statistic; its selection and return calculations stay intact.
An OS-backed output lease prevents concurrent writers and releases on process
exit, even if the lock file remains. A lock filename is not evidence of a live job.

Every 100 draws, save an atomic variant checkpoint containing BOTH cost rows for
each completed draw. `--resume` accepts only identical source/protocol/code
identity and complete, nonduplicated trial pairs. It advances the same random
stream past saved draws; no new seed or outcome selection. CSVs are regenerated
from the validated trial records. A completed audit can never be overwritten.
Before resuming, inspect the actual previous process/session; do not restart a
still-running job after a mere observation timeout.

```powershell
.venv/Scripts/python -m yenibot.automation.phase2_conditional_attribution `
  --scope-dir checkpoints/economic_attribution/20260628_155057/baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility/full `
  --probe-dir reports/phase2_economic_attribution/20260830_prequential_oof_v1 `
  --report-dir reports/phase2_local_lab/final_third_wave_local/local_gate `
  --output-dir reports/phase2_economic_attribution/20260830_conditional_score_v1
```
