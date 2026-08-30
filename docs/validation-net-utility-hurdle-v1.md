# One next historical probe: validation net-utility hurdle v1

Status: implemented; **candidate not fitted/evaluated yet at this code checkpoint**.
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
