# Current Phase 1 Status

Last reviewed: **August 24, 2026**

## Decision

The retained TCN+GRU control passed the historical walk-forward evidence
charter, but its preregistered Future-OOS candidate failed the confirmation
charter. Official/promotable Phase 2 therefore remains blocked.

- `control_recent3_equal_v2` is retired after its completed Future-OOS result.
- Its manifest, predictions, and failure result remain immutable audit evidence.
- The failed window cannot be reused for threshold, ensemble, feature, or policy selection.
- A new candidate requires historical-only research and a new preregistered OOS anchor.

## Retained Control

Profile:
`baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility`

Latest full-CV run: `20260628_155057`

| Metric | Result |
|---|---:|
| Mean Rank IC | `0.0748` |
| Rank IC std, monitor only | `0.0832` |
| Positive-IC folds | `76.3%` |
| Official-threshold F1 | `0.4377` |
| PRAUC lift vs prevalence | `1.1448` |
| Precision lift vs prevalence | `1.0847` |
| Top-decile label lift | `1.1321` |
| Top-decile forward return | `0.00286` |
| Worst-five mean Rank IC | `-0.0549` |

The model has useful ranking and payoff evidence. Its raw sigmoid output is not
a calibrated probability: raw and validation-only calibrated Brier skill remain
below climatology. Phase 2 must treat the output as a ranking score unless a
future frozen evaluation proves probability quality.

## Final Mechanism Result

Bundle `20260628_155057` completed primary-preserving multitask projection:

- only `1.8%` of 6,400 batches had conflicting shared gradients
- mean primary/auxiliary gradient cosine was `+0.61`
- candidate mean IC improved from `0.0542` to `0.0606`

It still failed the pre-registered promotion gates:

- Rank IC std worsened from `0.1078` to `0.1086`
- official F1 fell from `0.4423` to `0.4214`
- positive top-decile-lift fold coverage fell from `58.3%` to `50%`
- bad-fold top-decile lift remained only `0.7885`

Gradient interference is not the remaining instability mechanism. The
auxiliary multitask family is closed.

## Completed Future-OOS Result

The August diagnostics evaluated 1,524 mature hourly rows from June 13 through
August 15 without refitting or reselection. The candidate passed the observed
F1, PR-AUC lift, precision lift, and point-estimate Rank IC gates, but failed
the preregistered `rank_ic_lower_ci` gate:

| Metric | Future-OOS result |
|---|---:|
| Rows | `1,524` |
| F1 | `0.4896` |
| PR-AUC lift | `1.1645` |
| Precision lift | `1.1002` |
| Rank IC | `0.0380` |
| Rank IC lower confidence bound | `-0.0780` |
| Top-decile forward return | `0.00220` |

The point estimate is mildly positive, but the confidence interval includes a
materially negative ranking outcome. Under the frozen charter this is a real
failure, not a pending state and not a threshold-only issue.

## Frozen Future-OOS Track

The old preflight state `ready_prediction_only` and its action
`run_notebook_05_prediction_only` describe the moment before evaluation. Once
`evaluation_completed=True`, they are superseded by the evaluation outcome.
Generated reports now record this scope explicitly and expose the current
lifecycle action separately. The evaluator also treats the recorded outcome as
one-shot: it reuses the original verified artifact and refuses to rescore an
expanded version of the same window.

## Phase 2 Sandbox Track

The Phase 2 sandbox and clean-forward tooling remain useful engineering
artifacts, but neither is promotable evidence after the Phase 1 candidate
failure. The clean-forward run also did not establish a positive deployable
strategy. Do not use Notebook 06 or 07 to repair the failed model result.

## Next Operator Run

`recent6_validation_lcb_top3_v1` completed on the immutable historical cache
and failed its confirmatory gates:

| Metric | Recent-3 control | Adaptive candidate | Delta |
|---|---:|---:|---:|
| Mean Rank IC | `0.05157` | `0.04754` | `-0.00403` |
| Rank IC std | `0.08178` | `0.09080` | `+0.00901` |
| Worst-five Rank IC | `-0.08311` | `-0.09326` | `-0.01015` |
| Mean F1 | `0.39678` | `0.38672` | `-0.01006` |
| Positive selected-return folds | `65.8%` | `55.3%` | `-10.5 pp` |

The failure is broad, not a borderline single-gate miss. The pool size, top-k,
confidence level, validation split, and purge remain closed to tuning. The
result is preserved in `phase1_latest_policy_research_bundle.zip` with zero fit
operations and no failed-OOS selection rows.

The materially distinct `trajectory_swa_v1` mechanism is now preregistered.
It preserves the control features, labels, TCN+GRU widths, primary loss, seed,
fold boundaries, and thresholds. Inside each train fold it averages one
post-burn-in optimization trajectory and emits one checkpoint. This is not a
seed ensemble, recency ensemble, validation expert selector, or inference-time
blend.

Run Notebook 04 once. The candidate first runs only on the committed triage
folds and reaches full CV only if every existing triage gate passes. Then run
Notebook 05 once to create the review bundle. Do not run Notebook 04a, 06, or
07 in this cycle. The two failed Future-OOS windows remain excluded from
selection, and even a historical pass cannot auto-freeze a replacement.

The authoritative machine-readable instruction is
`operator_next_step.json`. A completed evaluation must always supersede the
older preflight action in that report.
