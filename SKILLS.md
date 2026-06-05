# yeniBot - Phase 1 Operational Manual (v3.7)

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

## Holdout And Future-OOS Policy

Holdout is a one-shot validation gate, not a development playground.

- The reserved holdout window must not be used to tune profiles, blend weights, thresholds, score bands, or feature families.
- If a policy fails clean holdout, mark it as failed/retired and do not tune it against the same holdout.
- Do not roll the holdout forward while `future_oos_monitor.allow_holdout_roll_forward` is `false`.
- Use `future_oos_monitor` to count fresh unseen bars after the anchor.
- Promotion is blocked until `future_unseen_oos_ready` is true and the candidate was pre-registered before that future-OOS window.
- Observed-best holdout rows are diagnostic only. They can inform future hypotheses, not immediate promotion.

## Phase 1 Readiness Gates

Phase 2 is blocked until `auto_review.py` reports all readiness checks passing:

1. `report_complete`: all required reports exist.
2. `mean_rank_ic`: walk-forward mean Rank IC is above `0.03`.
3. `rank_ic_std`: fold-to-fold Rank IC std is below `0.03`.
4. `positive_ic_fraction`: positive Rank IC in more than `75%` of folds.
5. `long_f1`: Long F1 exceeds `0.45` using the documented threshold source.
6. `calibration_separation`: actual long labels separate from non-long labels.
7. `mtf_leakage`: MTF leakage audit passes.
8. `stationarity_policy`: stationarity policy audit passes.
9. `future_unseen_oos_ready`: enough fresh unseen bars exist after the anchor.

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
- Prefer slim diagnostics bundles for review unless full per-profile prediction bundles are explicitly needed.
- Keep review artifacts inside the slim/full zip bundle. Do not litter Drive report roots with separate `latest_*` files.

Required diagnostic artifacts include:

- `profile_comparison.csv`
- `profile_blend.csv`
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
5. IC uncertainty: use `rank_ic_variance_decomposition.csv`, `rank_ic_aggregate_evidence.csv`, and `rank_ic_block_sensitivity.csv` to separate finite-fold measurement noise from estimated between-fold instability. Require multi-block sensitivity and random-effects evidence before claiming that fold std reflects structural regime failure. Do not silently replace or relax the official std gate, but do not invent new profiles merely to chase an std target that is below the measured noise floor.
6. Causal threshold transfer: use `causal_threshold_policy_summary.csv` to test thresholds formed from validation and past scores only. Full-test score quantiles remain diagnostic-only because they see the complete test score distribution.
7. Classification skill: interpret F1 only beside the always-long F1, a rate-matched random F1, PRAUC divided by label prevalence, precision lift, prediction-rate guardrails, and selected forward return. Raw F1 alone is not evidence of predictive skill and must not justify promotion.
8. Seed robustness: `seed_audit_coverage.csv` must prove that every configured seed fold exists, completed, and spans the available walk-forward history. Never silently ignore unavailable fold ids.
9. Validation charter: use `validation_charter_review.csv` to identify statistically weak legacy targets and `validation_charter_proposal.csv` to organize an inactive replacement draft. Neither report may change readiness gates automatically.
10. Score-band payoff: verify that high-score bands produce positive forward return, not only label lift.
11. Future-OOS readiness: do not promote until enough fresh unseen bars have accumulated.

Do not chase every holdout-best row. A holdout-best row seen after the fact is a hypothesis generator only.

## Git Discipline

Use focused commits with one of these prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation
- `data:` data pipeline
- `model:` model or training change

Always check `git status` before committing. Never revert user changes unless explicitly requested.

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
