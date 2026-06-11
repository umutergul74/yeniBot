# Experiment Memory

This summary complements the machine-readable experiment memory in
`config.yaml`. It exists to prevent repeated dead ends.

## Retained Baseline

`baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility`

This remains the strongest stability-oriented control. The frozen confirmation
candidate is built from its pre-anchor fold ensemble.

## Rejected Directions

| Direction | Lesson |
|---|---|
| Classical TA signals | Lagging price transforms did not provide durable edge |
| XGBoost meta-learner | Added no measurable value over the sequence model |
| Unshifted 4H merge | Produced severe future leakage and fake IC |
| Full stable-structure replacement | Reduced mean IC and increased instability |
| Broad raw volatility re-adds | Did not solve fold instability |
| Broad futures-context overlays | Improved isolated tail metrics but harmed CV stability |
| CVD pressure stable overlays | Failed promotion gates |
| Short-horizon large-trade pressure | Less stable than the long-horizon benchmark |
| Rank-only or z-score-only pressure | Did not beat the retained branch |
| Deleting all 4H bounded flow | Removed useful context |
| Label-margin pairwise loss | Improved one tail metric while worsening core CV evidence |
| Forward-return pairwise loss | Worsened mean IC, stability, F1, and worst folds |
| Stable-tanh taker-flow gating | Produced weaker IC and stronger reversal risk |
| Regime-specific thresholds | Did not improve official F1 causally |

## Governance Rule

Do not rerun a rejected profile because its name looks promising. A retest
requires:

1. a distinct causal mechanism,
2. a written reason,
3. pre-registration in experiment memory, and
4. a comparison that does not use the seen holdout for selection.

The current priority is frozen future-OOS confirmation, not profile search.

