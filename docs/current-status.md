# Current Phase 1 Status

Last reviewed: **June 28, 2026**

## Decision

The retained TCN+GRU control passes the active historical walk-forward evidence
charter, but Phase 2 remains blocked. Historical research and frozen future-OOS
confirmation are separate tracks:

- Historical CV research may continue without changing a frozen candidate.
- Frozen future-OOS candidates must remain prediction-only and immutable.
- A historical candidate cannot be promoted from the rolling holdout.

## Retained Control

Profile:
`baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility`

Latest full-CV run: `20260627_205102`

| Metric | Result |
|---|---:|
| Mean Rank IC | `0.0739` |
| Rank IC std, monitor only | `0.0690` |
| Positive-IC folds | `86.8%` |
| Selected-threshold F1 | `0.4681` |
| PRAUC lift vs prevalence | `1.1203` |
| Precision lift vs prevalence | `1.0951` |
| Top-decile label lift | `1.1219` |
| Top-decile forward return | `0.00299` |
| Worst-five mean Rank IC | `-0.0350` |

The model has useful ranking and payoff evidence. Its raw sigmoid output is not
a calibrated probability: raw and validation-only calibrated Brier skill remain
below climatology. Phase 2 must treat the output as a ranking score unless a
future frozen evaluation proves probability quality.

## Latest Mechanism Result

Bundle `20260627_205102` tested:

1. static label-uniqueness loss weighting, and
2. uniqueness plus broad order-flow event weighting.

Neither is promotable. More importantly, the audit showed that neither
mechanism was materially active:

- Static normalized uniqueness weights had mean Kish effective-sample fraction
  near `0.9983`; p10, p50, and p90 were effectively identical.
- The earlier overlap value near `0.094` is an information-dependence proxy,
  not the effective sample fraction of normalized loss weights.
- Averaging percentile ranks across 20 event columns almost never crossed the
  configured `0.80` threshold, leaving event weights effectively at one.
- The two candidates therefore produced almost identical predictions.

These v1 profiles are archived and cannot re-enter automatic training.

## Active Historical Experiment

The only active candidate is
`baseline_stable_orderflow_event_weighted_loss_v2`.

It keeps the control features, labels, architecture, and base loss unchanged.
Inside each train fold it:

- uses only the diagnosed order-flow family,
- computes mean absolute family strength and then its train-fold percentile,
- softly emphasizes the top 20 percent,
- disables static uniqueness weighting, and
- fails before training if active fraction, weight spread, or dominant-weight
  concentration indicates another no-op.

Seed audit remains enabled for seeds `42, 43, 44`, but every seed now uses the
same eight temporally spaced folds instead of running all folds for seeds 43 and
44.

## Frozen Future-OOS Track

`control_recent3_equal_v2` remains pinned as historical frozen-candidate
evidence. The latest diagnostics counted `313 / 720` mature labeled rows after
its anchor. Research changes must not refit, replace, or tune that frozen
candidate against its accumulating window.

The failed June 13 candidate remains immutable historical evidence and cannot
be retested on the same window.

## Next Operator Run

No data, feature, or label formula changed. Run:

1. `git pull`
2. Colab `Runtime -> Restart session`
3. `04_training_walk_forward.ipynb`
4. `05_diagnostics_validation.ipynb`

Notebook 04 should train the full control, one 12-fold v2 triage candidate, and
three eight-fold seed-audit scopes. At most one candidate may advance to full
CV. Notebook 05 must include the sample-weight effectiveness audit in the slim
bundle.
