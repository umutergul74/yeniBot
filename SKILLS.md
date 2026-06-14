# yeniBot - Phase 1 Operational Manual (v4.0)

This file is the project source of truth for the Phase 1 ML foundation. It replaces the early Scalp2-era guidance and must be read before changing data, features, labels, training, diagnostics, or experiment policy.

The project goal is not to find a short-term attractive metric. The goal is to build a bias-free, durable BTC/USDT perpetual futures long-signal model that can honestly graduate to Phase 2 and, later, live deployment.

## Mission And Boundary

Build a professional, bias-free ML pipeline that trains a binary TCN+GRU sequence model to identify BTC/USDT perpetual futures long opportunities from market microstructure features.

Phase 1 ends at model validation. Do not build backtesting, execution, trade management, order routing, live bot services, alerting, position sizing, or Phase 2 notebooks until Phase 1 readiness gates pass. Preparing a Phase 2 design document is allowed; writing Phase 2 trading code is not.

## Current Safe Baseline

The current safe control profile is configured in `config.yaml`:

- `experiments.control_profile`: `baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility`
- `features.active_profile` may remain a feature-generation default and is not by itself a promotion decision.
- `experiments.policy_review.status`: `failed_clean_holdout_review`
- `future_oos_monitor.allow_holdout_roll_forward`: `false`
- Frozen candidate `control_fold_ensemble_v1` completed its 737-row future-OOS
  evaluation on June 13, 2026 and failed. It is retired and must not be tuned
  or retested on that same window.
- `experiments.research_focus.mode`: `walk_forward_cv_repair`
- Future-OOS and recency replacement work is paused, not erased. Phase 2 remains
  blocked while the historical walk-forward failure mechanism is repaired.
- The active mechanism experiment keeps the strong control feature set and
  tests train-fold-only preprocessing. It must never fit clipping bounds or
  reliability decisions on validation, test, holdout, or failed future-OOS rows.
- Historical rolling-origin evidence from bundle `20260613_134953` showed a
  real trade-off: `recent_3_equal` improved mean Rank IC, positive-fold
  coverage, worst-fold IC, and F1 versus `all_eligible_equal`, while
  `all_eligible_equal` retained stronger top-decile lift. Do not repeat the
  broad recency sweep.
- Bundle 70 rejected `dual_horizon_all_recent3_50_50`: its IC improvement was
  real, but mean F1 was `0.4383` and positive top-decile return occurred in
  only `69.4%` of folds. `recent_3_equal` alone cleared every committed
  balanced non-inferiority gate. `control_recent3_equal_v2` was fitted and is
  retained as historical research, but its manifest-pinning workflow is paused
  until the underlying control profile is repaired on historical CV.

Treat these as operational facts unless a newer committed config deliberately changes them. Do not promote any profile, blend, score band, or threshold from the already-seen holdout window.

## Non-Negotiable Data Rules

- Use Binance USDT-M full kline data, not CCXT OHLCV-only data.
- Require the full 12-column kline schema, especially `num_trades`, `taker_buy_base_vol`, and `taker_buy_quote_vol`.
- Use Binance Vision or configured fallback sources when REST access is blocked by geography or HTTP 451.
- Validate duplicates, malformed rows, gaps, non-positive `num_trades`, and taker columns before feature generation.
- Rare zero-volume or no-trade archive rows may only be handled by the configured `zero_volume_policy`; do not silently keep malformed rows.
- Do not commit parquet data, raw downloads, checkpoints, runtime JSON state, `.env`, model weights, or diagnostic zip bundles.

## Feature Engineering Rules

Market microstructure is the signal source. Do not add RSI, MACD, EMA crossovers, Bollinger Bands, stochastic oscillators, or XGBoost meta-learners.

Use these families only when implemented causally and selected through `config.yaml` profiles:

- True order flow from taker buy/sell volume.
- Whale and ticket-size features from `num_trades`, volume per trade, and large-trade context.
- Volatility and structure as context, not directional TA.
- Stable rank, z-score, tanh, or bounded transforms for scale-sensitive features.
- Causal intrahour aggregates, such as 15m order-flow shape summarized into completed 1H bars.
- Futures context, such as open-interest, positioning, and funding features, only as pre-registered candidates.

Forbidden feature behavior:

- Do not feed raw price, raw volume, raw CVD, raw pressure, or raw scale-level columns directly into the model unless the active profile explicitly allows a proven stationary transform.
- Do not apply wavelet denoising to the full time series. Use rolling causal windows only.
- Do not compute rolling, rank, z-score, or normalization values with future rows.
- Do not match incomplete 4H bars to 1H rows.
- Do not let `merge_asof(backward)` see a 4H bar until that 4H bar is complete and shifted forward by exactly 4 hours.
- Do not use feature names alone as proof of safety. Verify stationarity, MTF alignment, and source timestamps in diagnostics.

## Labeling Rules

- Use long-only binary labels.
- `1` means the long TP was hit before long SL inside the configured horizon.
- `0` means not-long, including timeout or SL-first cases.
- Do not create 3-class long/flat/short labels until short-label quality has been explicitly validated in a separate future plan.
- Do not change label semantics to make F1 look better.
- Label-quality assertions must pass before training.

## Model And Training Rules

- The model is binary TCN+GRU with sigmoid `P(Long)`.
- Keep hyperparameters in `config.yaml`; do not hardcode research settings in source.
- Fit scalers on train folds only. Never fit on validation, test, or holdout rows.
- Train-fold quantile clipping and feature reliability masks are allowed only
  when their fitted bounds, block statistics, mask decision, and reason are
  persisted in `preprocessing_audit.csv`.
- Use purged walk-forward CV with configured purge and embargo.
- HMM regimes are diagnostics/filter metadata only, not a second-stage predictor.
- HMM validation, test, holdout, and live-style inference must be forward-only.
- Do not add an XGBoost, random forest, or other meta-learner on top of TCN+GRU outputs.
- Do not blindly retry old training-stability experiments. The May 21 branch already tested val-loss/rolling-IC style training changes, stronger regularization, and longer validation windows, then restored baseline defaults. New training experiments must isolate one change at a time and be pre-registered.
- The latest score-separation loss candidates are retired. `baseline_stable_score_margin_loss` and `baseline_stable_return_pairwise_loss_light` both failed versus the control on core CV gates. Do not retry loss-only score-separation tweaks unless a new diagnostic report identifies a distinct mechanism.

## Experiment Memory Discipline

Before adding or rerunning any profile, read:

- `config.yaml` -> `experiments.experiment_memory.reference_notes`
- `config.yaml` -> `experiments.experiment_memory.rejected_profiles`
- The latest `profile_comparison.csv`, `profile_blend.csv`, `performance_gap_analysis.csv`, and `fold_stability_summary.csv`

Do not repeat an already rejected experiment unless it has a new, explicit reason and is listed in `allow_retest_profiles`.

Known lessons:

- Full stable-structure replacement hurt the stronger raw/stable control balance.
- Broad futures-context overlays improved some holdout-looking tail metrics but damaged CV stability; split OI, positioning, and funding into narrower tracks.
- Large broad interaction families and raw volatility re-adds have generally not solved fold stability.
- The clean holdout invalidated the frozen top-10 `control + long_pressure` policy for active promotion; keep it as historical benchmark only.
- Run `20260528_193824` showed that regime-specific validation thresholds did not improve official F1; use `regime_threshold_policy_*` as diagnostics, not as a promotion mechanism.
- Run `20260528_193824` also identified bad-fold score separation compression/reversal as the main actionable failure mode; prioritize narrow score-separation training or feature hypotheses over broad profile search.
- Run `20260528_214759` showed that pairwise label-margin loss weight `0.05` improved top-10 lift but worsened mean IC, Rank IC std, positive-fold coverage, worst folds, and official F1. Do not repeat that exact label-margin objective; the next score-separation experiment should target forward-return ordering instead of class-only separation.
- Run `20260604_141520` showed that the forward-return pairwise loss candidate also failed: it lowered mean IC, worsened std, reduced positive-fold coverage, lowered official F1, and weakened top-10 lift versus the control. The next step is root-cause diagnostics, not another loss-only candidate.
- Run `20260605_141414` showed that stable-tanh taker-flow gating by stable large-trade context also failed: mean IC fell to `0.0393`, std rose to `0.0962`, positive folds fell to `66.7%`, and the low-pass guard became the strongest reversal suspect. Do not retry direct deletion, raw interaction, or this guarded-transform family without a distinct causal mechanism.
- Bundle `20260613_134953` root-cause evidence localized the remaining control
  weakness to score-ranking reversal rather than label balance. The strongest
  suspects are `4h_taker_imbalance_mean_12` sign reversal and
  `4h_large_trade_ratio` distribution drift. Global deletion and direct
  interaction variants already failed; the active new hypothesis is
  fold-fitted reliability masking plus train-only clipping.

## Holdout And Future-OOS Policy

Holdout is a one-shot validation gate, not a development playground.

- The reserved holdout window must not be used to tune profiles, blend weights, thresholds, score bands, or feature families.
- If a policy fails clean holdout, mark it as failed/retired and do not tune it against the same holdout.
- Do not roll the holdout forward while `future_oos_monitor.allow_holdout_roll_forward` is `false`.
- Use `future_oos_monitor` to count fresh unseen bars after the anchor.
- Promotion is blocked until `future_unseen_oos_ready` is true and the candidate was pre-registered before that future-OOS window.
- Observed-best holdout rows are diagnostic only. They can inform future hypotheses, not immediate promotion.
- Frozen future-OOS candidates must be recorded in `frozen_candidate_manifest.json`.
- Each required frozen candidate must pin an explicit `source_run_id`, frozen
  threshold payload, and expected manifest hash. Diagnostics from a newer run
  must never silently replace those source artifacts.
- Every referenced model, scaler, HMM, feature order, training signature, threshold source, and fit cutoff must be content-hashed before future-OOS scoring.
- Training and diagnostics signatures are separate. Report, chart, calibration
  audit, charter, or future-OOS display changes must not invalidate fitted
  model artifacts or trigger training by themselves.
- Future-OOS evaluation is prediction-only. It must perform zero fit operations and fail closed on missing or modified artifacts.
- Enough new bars is only readiness to evaluate. Phase 2 remains blocked until the frozen primary candidate is actually evaluated and passes its pre-registered future-OOS gates.
- Future-OOS readiness, evaluation, and pass are sequential states. Before the minimum row count is reached, evaluation/pass checks must be reported as pending rather than as additional failures.
- Optional historical benchmark candidates may remain unavailable without invalidating the required primary candidate. They must be reported as warnings and must never be substituted for the primary candidate.
- Keep an append-only `experiment_registry.jsonl`; do not rewrite historical decision records.
- A failed frozen candidate stays immutable as historical evidence. Record its
  outcome outside the frozen manifest so the original manifest hash remains
  reproducible.
- A failed future-OOS window may be used for root-cause diagnosis and may enter
  a later training set only after the candidate is retired. It may never be
  reused to choose the replacement threshold, ensemble weights, or promotion
  decision.
- Replacement candidates require a new pre-registration and a new future-OOS
  anchor. Their deployment policy must be selected on historical rolling-origin
  windows only.
- Recency-policy comparisons must use paired target-fold deltas, temporal
  moving-block uncertainty, positive-fold coverage, worst-fold IC, and
  economic payoff. A higher average IC alone cannot select a replacement.

## Phase 1 Readiness Gates

Phase 2 is blocked until `auto_review.py` reports all readiness checks passing:

1. `report_complete`: all required reports exist.
2. `mean_rank_ic`: walk-forward mean Rank IC is at least `0.03`.
3. `positive_fold_fraction`: positive Rank IC occurs in at least `75%` of folds.
4. `positive_fold_sign_test_pvalue`: fold positivity has one-sided sign-test p-value at most `0.01`.
5. `random_effects_positive_all_blocks`: the random-effects lower confidence bound stays positive across configured block lengths.
6. `prauc_lift_vs_prevalence`: PRAUC is at least `1.05x` label prevalence.
7. `precision_lift_vs_prevalence`: official-policy precision is at least `1.05x` label prevalence.
8. `f1_skill_vs_rate_matched_random`: F1 skill is positive versus random selection at the same prediction rate.
9. `positive_f1_skill_fold_fraction`: rate-normalized F1 skill is positive in at least `75%` of folds.
10. `positive_forward_return_fold_fraction`: selected rows have positive forward return in at least `60%` of folds.
11. `prediction_long_rate`: the official policy predicts long on at most `70%` of rows.
12. `calibration_separation`: backward-compatible gate name for positive
    long-vs-not-long **score separation**. It is not evidence that raw sigmoid
    values are calibrated probabilities. Probability quality must be reported
    separately with Brier skill versus climatology, log-loss skill versus
    climatology, ECE, and calibration slope/intercept.
13. `mtf_leakage`: MTF leakage audit passes.
14. `stationarity_policy`: stationarity policy audit passes.
15. `seed_audit_coverage`: configured seed/fold coverage is complete.
16. `future_unseen_oos_ready`: enough fresh unseen bars exist after the anchor.
17. `frozen_candidate_manifest`: the pre-anchor candidate artifacts are complete and hash-verified.
18. `future_unseen_oos_evaluated`: the frozen candidate has been scored on the fresh window without refitting.
19. `future_unseen_oos_passed`: the pre-registered future-OOS evidence gates pass.

The active validation charter is `v4_evidence`. It was activated by an explicit reviewed config and documentation commit after run `20260605_211102` showed:

- The observed fold Rank IC std was `0.0708`, while dependent-data bootstrap noise floors ranged from about `0.039` to `0.095`.
- The legacy `<0.03` std target was below the estimated measurement-noise floor for every configured block length.
- Random-effects lower confidence bounds remained positive across all configured block assumptions, and the positive-fold sign test was strongly significant.
- Raw Long F1 was below `0.45`, but the old target was also below the always-long no-skill F1 baseline. Rate-matched F1 skill, PRAUC lift, precision lift, prediction-rate control, and realized forward-return consistency provide the discriminative evidence.

Legacy `rank_ic_std < 0.03` and raw `long_f1 > 0.45` remain mandatory visible monitors. They are not blocking gates under `v4_evidence`; they must never be hidden or rewritten. This charter change does not declare Phase 1 complete and does not weaken leakage, stationarity, calibration, seed coverage, frozen-candidate, or future unseen OOS requirements. No draft charter may activate itself automatically.

If Rank IC is near `0.01`, features are inadequate. Do not respond by tuning model hyperparameters first.

If Rank IC exceeds `0.10`, assume leakage until proven otherwise. Audit MTF alignment, rolling windows, scaler fitting, and source timestamps.

## Diagnostics Workflow

Notebook order:

1. `01_data_preparation.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_labeling.ipynb`
4. `04_training_walk_forward.ipynb`
5. `05_diagnostics_validation.ipynb`

Operational rules:

- Run `02` again when new raw feature columns or feature-generation formulas are added.
- Run `03` again when labels, ATR source, data range, or processed features change.
- Run `04` only when training inputs, profiles, folds, labels, model/training config, or checkpoints need new predictions.
- Run `05` for diagnostics/report changes; it should not require GPU.
- Before frozen scoring, run the read-only future-OOS preflight. A failed
  preflight must block model loading; it must never be repaired by silently
  regenerating or retraining the candidate.
- Prefer slim diagnostics bundles for review unless full per-profile prediction bundles are explicitly needed.
- Keep review artifacts inside the slim/full zip bundle. Do not litter Drive report roots with separate `latest_*` files.

Required diagnostic artifacts include:

- `profile_comparison.csv`
- `profile_blend.csv`
- `preprocessing_audit.csv`
- `performance_gap_analysis.csv`
- `fold_stability_forensics.csv`
- `fold_stability_summary.csv`
- `fold_reliability_gate.csv`
- `fold_reliability_gate_summary.csv`
- `regime_threshold_policy_summary.csv`
- `regime_stability_summary.csv`
- `threshold_forensics.csv`
- `threshold_score_quantile_review.csv`
- `rank_ic_variance_decomposition.csv`
- `rank_ic_sampling_uncertainty.csv`
- `rank_ic_aggregate_evidence.csv`
- `rank_ic_block_sensitivity.csv`
- `causal_threshold_policy_summary.csv`
- `causal_threshold_policy_by_fold.csv`
- `classification_skill_summary.csv`
- `classification_skill_by_fold.csv`
- `seed_audit_coverage.csv`
- `validation_charter_review.csv`
- `validation_charter_proposal.csv`
- `validation_charter_status.json`
- `frozen_candidate_manifest.json`
- `frozen_candidate_index.csv`
- `future_oos_preflight.json`
- `future_oos_preflight.md`
- `future_oos_readiness.json`
- `future_oos_evaluation.csv`
- `future_oos_predictions.parquet`
- `future_oos_temporal_blocks.csv`
- `future_oos_score_bands.csv`
- `future_oos_regime_metrics.csv`
- `future_oos_ensemble_disagreement.csv`
- `future_oos_model_metrics.csv`
- `future_oos_failure_summary.json`
- `next_research_protocol.json`
- `recency_ensemble_summary.csv`
- `recency_ensemble_by_fold.csv`
- `recency_ensemble_schedule.csv`
- `recency_ensemble_eligibility_audit.csv`
- `recency_ensemble_paired_comparison.csv`
- `recency_ensemble_decision.json`
- `recency_ensemble_manifest.json`
- `experiment_registry_snapshot.jsonl`
- `holdout_evaluation.csv`
- `holdout_policy_decision.csv`
- `future_oos_candidate_plan.csv`
- `phase2_readiness.json`
- `phase1_transition_plan.json`
- `auto_review.json`
- `phase1_blocker_root_cause.csv`
- `threshold_oracle_gap.csv`
- `bad_fold_mechanism_summary.csv`
- `prediction_error_audit.csv`
- `historical_experiment_memory_audit.csv`
- `score_reversal_context_audit.csv`
- `phase1_decision_ladder.json`
- `model_performance_dashboard.md`
- `model_performance_summary.json`
- `model_performance_scorecard.csv`
- `model_metric_definitions.csv`
- `model_calibration_reliability.csv`
- `model_precision_recall_curve.csv`
- `model_evidence_uncertainty.csv`
- `probability_calibration_comparison.csv`
- `probability_calibration_comparison_by_fold.csv`
- `model_scorecard.png`
- `rank_ic_stability.png`
- `classification_quality.png`
- `score_band_payoff.png`

The auto-review command for a report directory is:

```bash
python -m yenibot.automation.auto_review --report-dir <report_dir>
```

## Interpreting Current Blockers

When mean IC and positive-fold rate are strong but Phase 2 still fails, focus in this order:

1. Fold stability: identify the folds contributing most to Rank IC std using `fold_stability_forensics.csv`.
2. Fold reliability: use `fold_reliability_gate_summary.csv` to test validation-only gates that may reduce bad-fold exposure; treat them as future-OOS hypotheses, not immediate promotions.
3. Regime stability: use `regime_stability_summary.csv` to determine whether HMM regimes explain bad-fold concentration before adding new feature families.
4. Threshold quality: separate selected-threshold F1, constrained-threshold F1, regime-threshold F1, score-quantile diagnostic F1, and pred-long-rate guardrails using `threshold_forensics.csv`, `threshold_score_quantile_review.csv`, and `regime_threshold_policy_summary.csv`.
5. IC uncertainty: use `rank_ic_variance_decomposition.csv`, `rank_ic_aggregate_evidence.csv`, and `rank_ic_block_sensitivity.csv` to separate finite-fold measurement noise from estimated between-fold instability. Under `v4_evidence`, raw std is a visible monitor; blocking evidence comes from positive-fold coverage, the fold sign test, and random-effects lower confidence bounds across multiple block assumptions.
6. Causal threshold transfer: use `causal_threshold_policy_summary.csv` to test thresholds formed from validation and past scores only. Full-test score quantiles remain diagnostic-only because they see the complete test score distribution.
7. Classification skill: interpret F1 only beside the always-long F1, a rate-matched random F1, PRAUC divided by label prevalence, precision lift, prediction-rate guardrails, and selected forward return. Raw F1 alone is not evidence of predictive skill and must not justify promotion.
   - Keep estimands explicit. Active charter classification gates are equal-weight fold-macro quantities. A pooled-row metric may be shown as a diagnostic, but it must never be compared directly with a fold-macro gate.
   - Classification uncertainty must resample temporal folds as clusters and then use moving blocks within each sampled fold. A single moving-block bootstrap across concatenated fold boundaries is not acceptable.
   - Report point estimates, confidence intervals, and probability-above-gate for both `macro_fold` and `pooled_rows`; only `macro_fold` is gate-comparable.
   - Probability calibration must be fit on each fold's validation split and evaluated on that fold's test split. Report macro-fold and pooled Brier/log-loss skill separately, plus the fraction of folds with positive skill.
   - Negative Brier or log-loss skill versus climatology means the score may still rank observations, but it must not be described or deployed as a calibrated probability.
8. Seed robustness: `seed_audit_coverage.csv` must prove that every configured seed fold exists, completed, and spans the available walk-forward history. Never silently ignore unavailable fold ids.
9. Validation charter: use `validation_charter_status.json` to identify the active committed charter and `validation_charter_proposal.csv` as its criterion-level evidence table. Reports may evaluate the active charter, but they may never change `active_version` automatically.
10. Score-band payoff: verify that high-score bands produce positive forward return, not only label lift.
11. Future-OOS readiness: do not promote until enough fresh unseen bars have accumulated.

## Executive Model Dashboard

The Phase 1 report must make the model's state understandable without hiding
technical risk behind a single score.

- `model_scorecard.png` is the first-look summary. It must distinguish
  `MODEL EVIDENCE` from `PHASE 2 READY`; these are not the same decision.
- `rank_ic_stability.png` shows every walk-forward OOS fold, negative folds,
  the mean IC, and the active mean-IC gate.
- `classification_quality.png` contains the precision-recall curve and
  reliability diagram. Always compare precision with label prevalence.
- `score_band_payoff.png` shows both label lift and forward return by score
  concentration. A high label lift with non-positive return is not economic
  evidence.
- `model_performance_scorecard.csv` is the machine-readable source for the
  dashboard. It must include discrimination, temporal robustness,
  statistical confidence, imbalanced classification, calibration,
  score-band payoff, seed robustness, integrity, and future-OOS readiness.
- `model_metric_definitions.csv` must explain every executive metric and its
  healthy interpretation.
- Legacy raw Rank IC std and raw Long F1 remain visible monitors. The dashboard
  must place them beside dependent-data uncertainty, rate-matched F1 skill,
  PRAUC lift, precision lift, and prediction-rate guardrails.
- Do not add Sharpe, drawdown, turnover, fee-adjusted return, or PnL to Phase 1
  model evidence. Those require the Phase 2 backtest and execution assumptions.
- Seen-holdout charts must be labeled diagnostic-only. Never present them as
  final promotion evidence.

Do not chase every holdout-best row. A holdout-best row seen after the fact is a hypothesis generator only.

## Git Discipline

Use focused commits with one of these prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation
- `data:` data pipeline
- `model:` model or training change

Always check `git status` before committing. Never revert user changes unless explicitly requested.

## Code Architecture Discipline

- New experiment code belongs under `yenibot/experiment/`; do not grow
  `yenibot/experiments.py` beyond its compatibility-facade role.
- Import notebook-facing experiment APIs from `yenibot.experiment`.
- Keep experiment modules responsibility-focused and below the architecture
  guardrail enforced by `tests/test_experiment_architecture.py`.
- Do not create circular dependencies between experiment modules.
- Training and diagnostics orchestration must update their atomic workflow
  status files so interrupted or failed stages can be located from the run
  directory.
- Do not silently swallow selected profiles, invalid fold ids, missing
  predictions, incomplete reports, or artifact-write failures.
- Preserve existing report schemas and compatibility imports during internal
  refactors unless a tested migration is deliberately committed.

## Absolute Prohibitions

- Do not build Phase 2 backtest, execution, trade management, live bot, or order-routing code before all Phase 1 gates pass.
- Do not add XGBoost or any meta-learner.
- Do not add 3-class labels.
- Do not use classical TA as a predictive signal family.
- Do not fit scalers, calibrators, feature selectors, or policy selectors on validation/test/holdout rows.
- Do not tune against the frozen holdout.
- Do not promote current-holdout winners without future unseen OOS confirmation.
- Do not repeat historically rejected experiments without an explicit retest reason.
- Do not silently relax criteria to force Phase 2.
- Do not count a partial seed audit as complete. Validate configured fold ids against the current purged walk-forward fold universe before training starts.
