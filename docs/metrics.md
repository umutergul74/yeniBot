# Phase 1 Metric Guide

## Discrimination

### Rank IC

Spearman correlation between `P(Long)` ranking and forward return. It measures
ordering, not calibration or profitability after costs.

- Positive mean IC: higher scores tend to precede higher returns.
- Positive-fold fraction: how broadly that relationship survives through
  time.
- Raw fold standard deviation: a visible stability monitor, interpreted
  beside dependent-data uncertainty rather than in isolation.

### PRAUC Lift

`PRAUC / long-label prevalence`. A value above `1.0` beats no-skill prevalence
for the imbalanced long-label task. Report fold-macro and pooled estimates
separately.

### Precision Lift

`precision / long-label prevalence` at the frozen policy threshold. It shows
whether selected observations contain more long labels than the unconditional
base rate.

### F1 Skill

Observed F1 minus the F1 of random selection at the same prediction rate.
This is more informative than raw F1 when class prevalence and prediction rate
vary.

## Temporal Robustness

- **Positive-IC fold fraction:** fraction of walk-forward test folds with
  positive Rank IC.
- **Positive-return fold fraction:** fraction of folds where selected rows
  have positive average forward return.
- **Sign test:** tests whether positive folds occur more often than chance.
- **Random-effects lower bound:** estimates whether the average effect remains
  positive after accounting for fold heterogeneity.

Market observations are dependent. Confidence intervals must resample folds
as clusters and preserve local dependence with moving blocks.

## Economic Ordering

### Top-Decile Lift

Long-label rate in the highest score decile divided by overall prevalence.
It is classification concentration, not PnL.

### Top-Decile Forward Return

Mean label-horizon forward return among the highest score decile. This is
useful economic ordering evidence but still excludes fees, slippage,
overlapping positions, execution latency, and risk sizing. Those belong to
Phase 2.

## Probability Quality

The reliability diagram compares mean predicted score with observed long-label
rate.

- Points on the diagonal: calibrated probabilities.
- Points below the diagonal: scores overstate event frequency.
- Negative Brier or log-loss skill versus climatology: do not interpret scores
  as probabilities, even if ranking metrics are positive.

Calibration must be fit on each fold's validation split and assessed on that
fold's test split. Pooled post-hoc calibration is not a valid deployment
claim.

## Estimand Labels

- **Fold-macro:** every temporal fold has equal weight. This is the active
  charter estimand.
- **Pooled rows:** every observation has equal weight. This is diagnostic and
  must not be compared directly with fold-macro gates.
- **Seen holdout:** diagnostic only after review; never tune against it.
- **Future unseen OOS:** frozen, no-refit confirmation evidence.

No single metric is sufficient. Promotion requires discrimination, temporal
robustness, controlled selection rate, economic ordering, integrity checks,
and the frozen future-OOS result.

