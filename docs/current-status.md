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

Latest full-CV run: `20260627_232543`

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

Bundle `20260627_232543` completed corrected event weighting v2. Unlike the v1
implementation, the audit proved that this mechanism was active:

- active-row fraction: `0.20`
- p90-p10 weight spread: `0.1691`
- Kish effective-sample fraction: `0.9936`

It still failed. Against the same 12 triage folds:

- mean Rank IC improved only `+0.0012`
- Rank IC improved in only 4 of 12 folds
- Rank IC std and worst-five IC worsened
- PRAUC fell
- global top-decile lift fell from `1.2276` to `1.1536`

The v1 and v2 sample/event-weighting profiles are archived and cannot re-enter
automatic training.

## Active Historical Experiment

The only active candidate is
`baseline_stable_multitask_return_head_light`.

It keeps the control features, labels, encoder widths, P(Long) head, primary
loss, and validation Rank-IC early stopping unchanged. It adds:

- one separate forward-return regression head on the shared representation,
- clipped and scaled targets,
- Huber loss with fixed weight `0.10`, and
- per-fold auxiliary Rank IC, MAE, and head-agreement audits.

Inference remains the original binary P(Long) output. This is not a retry of
the rejected pairwise-return loss, which directly constrained the P(Long)
scalar.

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

Notebook 04 should train the full control, one 12-fold multitask triage
candidate, and three eight-fold seed-audit scopes. At most one candidate may
advance to full CV. Notebook 05 must include the auxiliary-task audit in the
slim bundle.
