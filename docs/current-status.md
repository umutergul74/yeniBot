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

Latest full-CV run: `20260628_093830`

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

Bundle `20260628_093830` completed the fixed-sum auxiliary-return experiment.
Against the paired 12-fold triage control it:

- improved mean Rank IC from `0.0518` to `0.0602`
- improved PRAUC from `0.3591` to `0.3652`
- improved global top-decile lift from `1.2276` to `1.2705`
- improved Rank IC in 7 of 12 folds

It still failed the pre-registered promotion gates:

- Rank IC std worsened from `0.0985` to `0.1032`
- official F1 fell from `0.4557` to `0.4497`
- bad-fold top-decile lift was only `0.6406`
- the auxiliary and P(Long) rankings were `0.921` correlated

The target carries useful information, but fixed gradient summation does not
protect the primary task in unstable periods. The fixed-sum profile is archived.

## Active Historical Experiment

The only active candidate is
`baseline_stable_multitask_return_head_conflict_projected`.

It keeps the same auxiliary target and fixed weight `0.10`. The only change is
shared-gradient handling:

- the primary P(Long) gradient remains unchanged
- aligned auxiliary gradients are retained
- only a negatively aligned auxiliary component is projected away
- conflict frequency, cosine before/after, and relative gradient norms are
  persisted per fold

Inference remains the original binary P(Long) output. No auxiliary-weight
search or additional target is active.

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

Notebook 04 should train the full control, one 12-fold conflict-projected
multitask triage candidate, and three eight-fold seed-audit scopes. At most one
candidate may advance to full CV. Notebook 05 must include both auxiliary-task
and multitask-gradient audits in the slim bundle.
