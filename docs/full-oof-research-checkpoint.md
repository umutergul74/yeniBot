# Full OOF research checkpoint

Updated August 30, 2026. This is a continuation checkpoint, not a success report.

## Objective and current evidence

The goal remains demonstrated, robust model contribution to executable net
returns and a new clean confirmation. Historical ranking, a passing unit test,
or a profitable selected backtest does not meet that objective. The prior
future-OOS failure and all rejected model families remain immutable.

The previous goal turn made progress: acquired and hash-verified the full OOF
source, audited its schema, and established a feasible historical-only test.

## Authoritative source

- Branch: `codex/phase1-research-v2`.
- Source run: `20260628_155057`, retained control profile, `full` scope.
- Local cache: `checkpoints/economic_attribution/20260628_155057/` followed by
  the profile in the JSON specification and `/full`.
- Source is the same subtree of Drive `yeniBot/checkpoints/experiments/`.
- Predictions SHA-256:
  `a09a0179068b14a9379bebb6a602a5627c8364b88cf3e5d5ea901cf1682468f6`.
- Training-manifest SHA-256:
  `24e488533ec54579a80256f26ea8af520514145d6538717e736be7560dd29b8e`.
- 63,612 prediction rows: 38,646 validation and 24,966 test; 38 folds,
  each with 1,017 validation and 657 test rows.
- Test open timestamps: November 15, 2022 01:00 UTC through December 26,
  2025 09:00 UTC. No failed 2026 future-OOS rows enter this audit.
- Duplicate `source_row_position` values are fold-local offsets, NOT timestamp
  duplicates. The builder validates actual timestamps, not these offsets.

## Audit contract

`configs/full_oof_attribution_v1.json` pins the source, cutoff, fold sizes,
timestamp convention, score transform and all four density thresholds.

Each test score becomes its empirical percentile against only its own preceding
validation scores. Test labels and future test-score distributions never enter
the transform. Decisions occur at the close of the source Binance hourly bar;
entry is the next bar open with the preceding bar's ATR. The baseline TP2/SL5,
10-hour maximum holding period, non-overlap and stop-first ambiguity rules stay
unchanged. No later selected raw threshold is projected back into 2022-2025.

Thresholds q50/q70/q80/q90 are a bounded **retrospective exploratory panel**,
chosen after the historical marginal score diagnostics were inspected. This is
not a preregistered untouched experiment. The historical profile itself was
selected using historical evidence. Holm correction covers this panel only,
not the project's entire research history.

The score-only execution cache reuses reference-engine exit resolution and
costs. Each run checks equivalence against the actual score, first random
control, always-on and inverted-score paths. Synthetic tests additionally cover
no signals, random selections, score/ATR filters, dynamic stops, same-bar
ambiguity and data gaps. It is not a live portfolio engine.

Zero-trade folds remain in the positive-fold-share denominator. Returns are
completed-trade compounded returns over independent fold paths, not a continuous
live equity curve; censored terminal positions are excluded and remain reported.
Drawdown here is completed-trade-close drawdown, not intrabar worst-case risk.
Base funding is a fixed-rate estimate, not complete historical funding; no result
can authorize deployment.

## Current run / resume

Run directory: `reports/phase2_economic_attribution/20260830_full_oof_cdf_v1`.

The command is:

```powershell
.venv/Scripts/python -m yenibot.automation.phase2_oof_attribution `
  --scope-dir checkpoints/economic_attribution/20260628_155057/baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility/full `
  --report-dir reports/phase2_local_lab/final_third_wave_local/local_gate `
  --output-dir reports/phase2_economic_attribution/20260830_full_oof_cdf_v1
```

The initial local handle was exec session `1234`, OS worker PID `9428`.
These are historical identifiers, not proof it remains alive. Recheck the handle
or process before restarting. Existing output directories are never overwritten.
Each completed strategy writes its result; `full_oof_attribution_suite.json`
appears only after the complete four-strategy family finishes.

## Next decision

1. Inspect all four results together, including adverse costs, all-fold breadth,
   reference controls and family-wise significance. Do not choose on raw return.
2. Ordinary within-month score permutations do not preserve signal serial
   dependence. Any apparent model contribution still needs a dependence-aware
   control (e.g. circular shifts/block methods) before a stronger claim.
3. If economic evidence fails, close this fixed-exit score-density panel without
   searching exits. Use the observed failure mechanism to specify ONE model
   utility/abstention hypothesis; do not reopen already rejected families.
4. If robust historical evidence survives, preregister one post-lock unseen
   confirmation, with realistic actual funding and execution/risk checks.
5. No Notebook 04/04a/06/07 or live trading is requested at this checkpoint.

## Statistical references

Randomized p-values use the conservative add-one convention described in the
[SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
The chosen null still needs an appropriate exchangeability assumption; the
calculation alone does not establish that hourly observations are independent.
Historical strategy selection creates optimism that cannot be undone merely by
renaming a test split; see Bailey et al.,
[The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
