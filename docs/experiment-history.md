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
| Global deletion of 4H taker mean-12 | Removed useful good-period signal and worsened CV |
| Global deletion of 4H large-trade ratio | Improved isolated F1/lift but damaged IC stability |
| Direct taker-flow x large-trade interactions | Failed to preserve the retained control |
| Hard train-fold 4H-flow reliability masking | Altered 10/12 triage folds, increased dispersion, and reduced positive-fold coverage |
| Simple train-only 4H large-trade clipping | Controlled but non-promotable; small IC gain came with lower top-10 lift |
| Clipping plus hard reliability masking | Did not repair masking instability |
| Uniform and component-only model shrinkage | Improved isolated stability measures but failed to preserve F1 and top-score payoff |
| Equal-weight seed-rank ensemble | Raised mean IC while worsening dispersion, worst folds, F1, and top-decile payoff |
| Static label-uniqueness loss weighting | Normalized weights were nearly constant; low overlap information was incorrectly assumed to imply useful row-level weights |
| Broad v1 event weighting | Mean rank across 20 columns almost never crossed the event threshold, making the component a no-op |
| Corrected order-flow event weighting v2 | Weights were active, but paired triage IC wins occurred in only 4/12 folds while dispersion, PRAUC, worst folds, and top-decile lift worsened |
| Fixed-sum auxiliary return head | Improved mean IC, PRAUC, and top-decile lift, but worsened dispersion and official F1; auxiliary and primary rankings were highly correlated |
| Primary-preserving multitask projection | Only 1.8% of batches conflicted; projection did not improve dispersion, F1, or bad-fold payoff |

## Completed Research Freeze

Bundle `20260628_155057` closed the final preregistered multitask follow-up.
Gradient conflict was rare and projection remained non-promotable.

No active historical profile candidate remained at that point. The retained
control passed the active historical evidence charter, but the subsequently
completed `control_recent3_equal_v2` Future-OOS evaluation failed the frozen
Rank-IC lower-confidence-bound gate. That exact candidate is retired.

## Active Preregistered Mechanism

`recent6_validation_lcb_top3_v1` is the only open confirmatory hypothesis. It
does not add features, change the TCN+GRU, tune a seed ensemble, or repeat a
static recency policy. At each historical target fold it compares the fixed
six-model recent pool on one common, already-mature validation selector
window; it selects three models by block-bootstrap Rank-IC lower bound and
uses a disjoint post-purge validation segment for threshold calibration.

The experiment reuses immutable cached predictions ending in December 2025.
Neither failed Future-OOS window is eligible for model, threshold, or policy
selection. The candidate must clear every committed ranking, dispersion,
classification, and payoff gate against the recent-three benchmark. Passing
historical gates only permits a separately reviewed fit at a new anchor.

## Governance Rule

Do not rerun a rejected profile because its name looks promising. A retest
requires:

1. a distinct causal mechanism,
2. a written reason,
3. pre-registration in experiment memory, and
4. a comparison that does not use the seen holdout for selection.

## Completed Mechanism Experiment

The retained control's five negative folds are not explained by label balance.
The dominant mechanism is score-ranking reversal:

- `4h_taker_imbalance_mean_12` changes sign between good and bad periods.
- `4h_large_trade_ratio` shows material distribution drift.

Bundle `20260614_054446` compared:

1. train-only clipping of `4h_large_trade_ratio`,
2. train-only reliability masking of `4h_taker_imbalance_mean_*`, and
3. their combination.

Every fold decision was written to `preprocessing_audit.csv`. No candidate
passed the pre-registered promotion gates. The hard masking rule is rejected;
clipping is archived as inconclusive/non-promotable. The next experiment must
use a distinct mechanism and improve both ranking stability and top-score
payoff rather than optimizing mean IC alone.

Before another mechanism experiment, the retained control must complete its
missing multi-seed audit. The audit is appended to run `20260614_054446` as
isolated seed scopes; it is not a repeat of full CV and does not reopen the
failed preprocessing candidates.

## Seed Audit Completion

The June 15 seed extension completed all configured scopes:

- seed `42`: mean IC `0.0316`, positive folds `50.0%`
- seed `43`: mean IC `0.0188`, positive folds `62.5%`
- seed `44`: mean IC `0.0602`, positive folds `75.0%`
- equal seed ensemble: mean IC `0.0407`, positive folds `75.0%`

The result confirms meaningful seed sensitivity, but it does not yet isolate
initialization as the only cause. The original full seed-42 run has mean IC
`0.0494` on the same eight folds, versus `0.0316` in the appended seed-42
scope. The next diagnostic is therefore an exact same-seed reproducibility
audit. No feature or loss experiment may be selected until manifest and
runtime compatibility are reviewed.
