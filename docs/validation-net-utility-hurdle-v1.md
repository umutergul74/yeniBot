# Validation net-utility hurdle v1 — completed, failed

Status: **completed; failed; family closed to retuning**.
Date: August 30, 2026. This is a bounded historical research contract, not
permission to deploy and not independent confirmation.

## Reason for this hypothesis

The retained score's full-history density audit shows that statistical ranking
and executable utility are different. The q80/q90 policies have only about
1.9/3.3 basis points mean net return under the base cost model, while adverse
costs make both strongly negative. Raising one fixed percentile does not fix
that fragility. More exit/threshold combinations are not the next experiment.

Hypothesis: a very small, validation-only model of **net trade payoff** can
identify when the frozen ranking component's expected gross advantage is too
small relative to costs, and abstain. This is a testable hypothesis, not an
expectation that it will pass.

## Fixed scope

Use only the two SHA-256-pinned files in `full_oof_attribution_v1.json`.
All 2026 future-OOS and forward-failure windows remain excluded. The original
TCN+GRU weights, features, labels and failures are unchanged. Notebook training
remains paused; this probe would fit only a separate low-capacity payoff model.

One candidate, one fixed model class, no hyperparameter or exit search:

- Inputs: past-validation score percentile, preceding-bar ATR/price fraction,
  and their product. Include an intercept. No test-derived features/statistics.
  Price here means the decision bar's known close, NOT the unseen next open.
- Estimator: ridge regression, fixed `alpha=10`, validation-only standardization
  of the three inputs. A constant input receives scale 1, never fabricated data.
- Target: the adverse-cost net return of a single potential trade under the
  unchanged TP2/SL5/10-bar, next-open, stop-first execution contract.
- Label construction: evaluate each eligible validation opportunity separately;
  exclude incomplete/censored paths and every path whose outcome is not mature
  before the test boundary. These overlapping opportunity labels are training
  examples, **not independent realized portfolio trades**.
- Fit exactly once per fold, using that fold's prior validation opportunities;
  minimum 200 eligible observations. No test labels may enter a fit or scaler.
- Action: trade only when predicted adverse-cost net return is strictly positive;
  missing/invalid estimates imply no trade. There is no tuned utility threshold.
- Actual performance uses non-overlapping positions and the reference engine,
  never the sum of overlapping opportunity labels.

The entire workflow must report its payoff-model fitting truthfully. It must not
inherit an unqualified `model_or_strategy_refit_performed=False` label from the
prediction-only attribution runner. It is a new research policy, not the old
frozen candidate under a renamed manifest.

## Required controls and decision

Compare against the immutable q80 reference and no-trade under identical folds,
costs and terminal censoring. The q80 reference is a fixed context benchmark,
not a chosen deployable winner. Include score-order destruction and serial
shift controls; report realized turnover, not only selected-row density.

All existing full-cache economic criteria stay unchanged: positive base AND
adverse returns, at least 100 completed trades, at least two-thirds positive
folds, positive ranking/payoff diagnostics, and complete execution inputs.
Require positive paired fold-level mean improvement over the q80 reference,
with a conservative temporal-block uncertainty assessment. Do not treat hourly
overlapping labels as independent samples for confidence intervals.

Before any candidate fit, the implementation pinned this assessment to paired
arithmetic fold-return deltas under BOTH base and adverse costs, moving blocks
of 3 and 6 adjacent folds, 5,000 resamples, and 95% percentile intervals. Require
the lower bound to be positive for both block lengths in both cost scenarios.
This is approximate uncertainty under temporal dependence, not an independent
test. The fixed machine contract is `configs/validation_net_utility_hurdle_v1.json`.
The engine-compatible action score is a monotone arctangent encoding of predicted
utility, NOT probability; exact zero and invalid estimates are forced to abstain.

If it fails, archive the one result. Do not retune alpha, utility cutoff, input
terms, validation history, exits or costs on these same test results. Any next
family requires a distinct written mechanism, not a hidden continuation sweep.
If it survives, it can justify a separately locked, post-lock unseen candidate
confirmation only. Historical profile/strategy selection bias still remains.

## Implementation acceptance checks

1. Altering test labels, later test scores or later folds cannot change fitted
   coefficients or earlier actions.
2. Targets exactly match reference single-opportunity execution and adverse
   fees/slippage/funding assumptions; boundary/gap/censor cases are tested.
3. No training opportunity crosses the validation/test boundary.
4. All fits/scalers, input hashes, coefficients, eligible/excluded counts and
   fold actions are persisted for reproducible audit.
5. A fresh full-family evaluation is append-only, non-promotable and remains
   separate from the retired candidate's immutable outcome.

No Colab work is required. The completed full OOF audits remain unchanged. Run
`yenibot.automation.phase2_net_utility` only with the pinned full-scope cache and
the verified original q80 report; it records per-fold targets/fits and candidate
actions before evaluation, and never overwrites an existing output directory.

## Completed result and restart checkpoint

Implementation and all settings were committed/pushed as `5c060de` before the
first real fit. Run: `reports/phase2_economic_attribution/20260830_net_utility_v1`.
There were 38 payoff-model fits and zero TCN+GRU refits. All fits/scalers,
coefficients, training opportunities, decisions, ledgers and both null controls
were retained. Exec session `90751` / PID `28268` completed with exit code 0;
the read-only calibration diagnostic (`47930`) also completed. Neither is live.

Main result SHA-256:
`76e76d35470f3877cb7d919b34f11a2f7d9b183fbebe62c01a7c77fbfe303d6c`.
Do not rerun or tune this completed family. The 61 relevant tests passed.

| Metric | Candidate | Fixed q80 reference |
|---|---:|---:|
| Whole-period base return | +20.75% | +13.82% |
| Whole-period adverse return | -46.27% | -65.26% |
| Completed trades | 805 | 1,186 |
| Base trade-close drawdown | -40.57% | -43.47% |
| Base profit factor | 1.063 | 1.041 |
| Positive folds | 13/38 | 21/38 |

These are completed-trade compounded returns across independent historical folds,
not annualized or live portfolio returns. Eight candidate folds had zero trades;
17 lost money. Thus insufficient breadth is not merely an abstention penalty.
Twelve terminal censored positions are not completed trades.

The candidate beat both conditional nulls (500 each, p about 0.001996, at the
Monte Carlo resolution floor); serial-null mean turnover was 812.8 versus 805
actual trades. This is evidence of historical policy timing, not proof that
the TCN score, rather than the ATR context, caused all incremental value.

The positive paired base fold-return difference was only +0.118 percentage points.
Its 6-fold-block 95% interval was [-1.142, +1.009] percentage points. Under adverse
costs, the difference was +1.092 points with interval [-0.998, +2.422]. The 3-fold
intervals also included zero. Improvement over q80 is not established. The
candidate failed adverse return, fold breadth and paired uncertainty gates.

## Calibration mechanism evidence (no new fit or selection)

Saved-fit reconstruction and the reference opportunity engine showed:

| Population | Mean predicted adverse net payoff | Mean realized adverse net payoff |
|---|---:|---:|
| Selected validation training opportunities | +18.27 bps | +19.10 bps |
| Selected test opportunities | +20.01 bps | -0.38 bps |
| Completed test portfolio trades | +15.39 bps | -6.51 bps |

The first two populations contain overlapping opportunities and are NOT
independent samples or realizable portfolio ledgers. The final row uses the
actual non-overlapping completed trades. No significance claim is made from
the large overlapping observation count. Payoff calibration did not transfer.

The source training manifest identifies a 34,399-byte historical trainer with
SHA-256 `c19640c29e6e3dc1d7a3b67e257ccc8fd06290c150c95411c61097faa070eb01`.
This matches the git blob at `973906b:yenibot/training/trainer.py` exactly.
Lines 556/559/562 select the checkpoint using validation performance; lines
585/588 restore that best state and predict the same validation set. Therefore
the payoff model's validation source was NOT independent of base-model selection.

This is a verified optimism risk, NOT proof it alone caused the loss. Regime
change, estimator misspecification and opportunity-to-portfolio selection remain
possible contributors. The test data did not enter v1's fit; its rejection stays
valid. No gate is relaxed and no failure is relabeled pending.

For context, scikit-learn's [stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
uses cross-validated base predictions for the second-stage model and warns
about reuse of fitted-data predictions. Our checkpoint-selection case is not
identical to direct in-sample stacking, but the independence concern motivates
the next audit. The analytic ridge implementation was tested against the
[reference Ridge estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Next work — not a hidden v1 retune

Update August 30: the separate [past-OOF utility v1](prequential-oof-utility-v1.md)
protocol and probe below have now been completed. That document is the active
checkpoint. It improved historical returns but failed its conjunctive gate;
neither this validation family nor the new probe may be retuned to rescue it.

Audit the feasibility of a time-ordered second stage trained ONLY on already
mature predictions/outcomes from earlier OOF test folds. Keep the current and
future fold completely excluded. Earlier-fold test labels, if used, must be
explicitly disclosed as training data; never claim no test labels anywhere were
used. Do not change v1's alpha, utility cutoff, exits or validation window.

Before any new candidate fit, write ONE separate protocol specifying the source
independence, warm-up rule, common evaluation cohort and matched benchmarks.
Keep all benchmarks on that same cohort, not the original 38-fold total if early
folds supply calibration warm-up. Include an ATR-only context control to isolate
the TCN score's incremental contribution. No automatic model/threshold search.
Such a probe would still be retrospective and could not remove historical profile
selection bias or the need for post-lock unseen confirmation. No Colab run or
live trading is needed at this checkpoint.
