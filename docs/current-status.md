# Current Phase 1 Status

Last reviewed: **June 28, 2026**

## Decision

The retained TCN+GRU control passes the active historical walk-forward evidence
charter, but Phase 2 remains blocked. Historical research and frozen future-OOS
confirmation are separate tracks:

- Historical CV model research is frozen; no active candidate remains.
- Frozen future-OOS candidates must remain prediction-only and immutable.
- A historical candidate cannot be promoted from the rolling holdout.

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

## Model Research State

Historical model research is frozen. `candidate_profiles` is empty. The
retained control passes every active historical model-evidence gate, including
same-seed reproducibility, leakage, stationarity, aggregate Rank IC,
classification skill, and top-score payoff. It is suitable for Phase 2
backtest research only after frozen future-OOS confirmation passes.

## Frozen Future-OOS Track

`control_recent3_equal_v2` remains pinned as historical frozen-candidate
evidence. The latest diagnostics counted `313 / 720` mature labeled rows after
its anchor. Model changes must not refit, replace, or tune that frozen
candidate against its accumulating window.

The failed June 13 candidate remains immutable historical evidence and cannot
be retested on the same window.

## Next Operator Run

Do not run notebook 04. When at least 720 mature post-anchor rows are available:

1. `git pull`
2. Colab `Runtime -> Restart session`
3. `01_data_preparation.ipynb`
4. `02_feature_engineering.ipynb`
5. `03_labeling.ipynb`
6. `05_diagnostics_validation.ipynb`

Notebook 05 performs prediction-only frozen evaluation. If it passes, Phase 2
implementation may open from the reviewed design and entry checklist. If it
fails, the failure becomes the only valid source for reopening model research.
