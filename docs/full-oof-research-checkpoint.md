# Full OOF research checkpoint

Updated August 30, 2026. Both audits completed; the goal is NOT achieved.

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

## Completed runs / resume

Run directory: `reports/phase2_economic_attribution/20260830_full_oof_cdf_v1`.

The command is:

```powershell
.venv/Scripts/python -m yenibot.automation.phase2_oof_attribution `
  --scope-dir checkpoints/economic_attribution/20260628_155057/baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility/full `
  --report-dir reports/phase2_local_lab/final_third_wave_local/local_gate `
  --output-dir reports/phase2_economic_attribution/20260830_full_oof_cdf_v1
```

Exec session `1234` / worker PID `9428` completed with exit code 0. The additional
serial-control run (session `88174`) also completed with exit code 0. Neither is
an active wait. Do not restart either completed run; outputs are append-only.

Serial output: `reports/phase2_economic_attribution/20260830_full_oof_serial_v1`.
Runner: `yenibot.automation.phase2_oof_serial_control`, with the same scope and
gate paths, `--attribution-dir` pointing at the original run and `--output-dir`
pointing at the serial output. Both complete suite JSONs exist.

Suite JSON SHA-256:

- Original: `3e84792e679932459e12ae7a38b9976196381ede2b51decf260a0c85fb15c6dc`.
- Serial: `0109b42ea00023c94145d10711cf6944f18dde36f7ece936955c245e00a3fb69`.

## Completed result

These are whole-period, completed-trade returns over independent November
2022-December 2025 folds at unit notional, NOT annualized/live portfolio returns.
Terminal censored positions are excluded, and drawdown is trade-close based.

| Validation cutoff | Base return | Adverse return | Trades | Positive folds | Base drawdown |
|---|---:|---:|---:|---:|---:|
| q50 | -76.30% | -97.18% | 2,126 | 12/38 | -77.76% |
| q70 | -34.17% | -86.12% | 1,552 | 18/38 | -49.96% |
| q80 | +13.82% | -65.26% | 1,186 | 21/38 | -43.47% |
| q90 | +19.54% | -42.84% | 737 | 21/38 | -31.59% |

Every policy failed adverse costs and fold breadth. q80/q90 profit factors were
only 1.041/1.070 and mean base net returns 1.90/3.27 bps per trade. This fixed
four-density panel is closed to further tuning; no policy is deployable.

Ordinary shuffling preserved selected rows but increased realized turnover.
For q80, 1,186 actual trades became 2,010.7 mean null trades. The follow-up
circular-shift control preserved cyclic score order inside each fold/month:

| Cutoff | Actual trades | Serial-null mean trades | Serial-null median return | Serial p |
|---|---:|---:|---:|---:|
| q50 | 2,126 | 2,126.6 | -89.31% | 0.00399 |
| q70 | 1,552 | 1,563.8 | -82.17% | 0.00200 |
| q80 | 1,186 | 1,198.9 | -75.23% | 0.00200 |
| q90 | 737 | 741.7 | -60.30% | 0.00200 |

There were 500 controls of EACH kind per policy. Using each policy's larger
p-value from the two controls and Holm correction across the four policies
gives approximately 0.00798 for each. Three serial p-values hit the Monte Carlo
minimum, not a precisely estimated smaller value. Circular shifts preserve
cyclic order with a wrap seam and assume local shift-invariance; they do not
force identical turnover or eliminate historical model-selection bias.

Historical model contribution survives the turnover-aware comparison. This is
stronger evidence than the narrow three-fold audit, but is NOT high, future-stable
model success or a deployable system. Adverse-cost failure, substantial drawdown,
historical selection bias and the original failed future-OOS remain.

## Next decision

1. Do not rerun the completed audits or select q90 on its highest base return.
2. The [validation net-utility hurdle v1](validation-net-utility-hurdle-v1.md)
   follow-up is now completed and failed: +20.75% base, -46.27% adverse and only
   13/38 positive folds. Do not rerun or retune it. The calibration-source audit
   led to the separate [past-OOF utility v1 protocol](prequential-oof-utility-v1.md).
   That probe is now complete: +149.01% base/+80.65% adverse, but the locked gate
   still failed breadth and paired comparisons. It is closed to retuning.
   Resume from that document's execution/risk realism audit checkpoint. Keep
   its fixed 36-fold cohort separate from these older 38-fold totals.
3. Preserve every failure. Any historical success still needs a separately
   locked post-lock unseen confirmation with actual funding/execution/risk checks.
4. No Notebook 04/04a/06/07 or live trading is requested at this checkpoint.

## Statistical references

Randomized p-values use the conservative add-one convention described in the
[SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
The chosen null still needs an appropriate exchangeability assumption; the
calculation alone does not establish that hourly observations are independent.
Historical strategy selection creates optimism that cannot be undone merely by
renaming a test split; see Bailey et al.,
[The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
