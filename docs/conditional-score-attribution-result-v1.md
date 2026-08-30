# Conditional score attribution v1 result

Status: completed and independently reconstructed on August 30, 2026. This is
retrospective sensitivity evidence, not clean confirmation or live permission.

The protocol was committed and pushed as `7f30d6a` before the first conditional
return was calculated. The immutable result is anchored by SHA-256
`409c46ebe172007e58708e534885544b73e534fce705f9040c1d7992cf481ee6`.
The local output is
`reports/phase2_economic_attribution/20260830_conditional_score_v1`.

## Result

The frozen past-OOF policy was compared with controls that moved only the
TCN+GRU score percentile inside fixed fold/month/ATR strata. The recipient ATR,
the previously fitted three-input ridge parameters, exit contract and costs
were unchanged. No model, payoff fit, threshold or candidate selection ran.

| Policy | Base return | Adverse return | Trades | Mean base/adverse net return |
|---|---:|---:|---:|---:|
| Actual frozen policy | +149.01% | +80.65% | 315 | 30.68 / 20.47 bps |
| 1.10 ATR ordinary-null median | +43.86% | -2.41% | 378.5 mean | 11.31 / 1.07 bps |
| 1.10 ATR circular-null median | +46.35% | +1.58% | 356.1 mean | 12.43 / 2.19 bps |
| 1.05 ATR ordinary-null median | +43.60% | -2.58% | 379.1 mean | 11.24 / 0.97 bps |
| 1.05 ATR circular-null median | +47.67% | +1.15% | 368.2 mean | 12.31 / 2.08 bps |

All 16 locked combinations (two ATR grids, two perturbations, two costs and two
statistics) had zero of 500 controls reach the actual value. Their conservative
add-one Monte Carlo value is therefore `1/501 = 0.001996007984`, the resolution
floor rather than evidence for a still smaller probability. The narrowest
actual-versus-null-maximum return margin was +3.82 percentage points for the
1.05 ordinary/base comparison.

Exchangeable selected-row coverage was 98.332% for the 1.10 grid and 96.090%
for the 1.05 grid. Maximum observed donor/recipient ATR ratios were 1.0998888
and 1.0499534, within the locked bounds.

## Independent reconstruction

A separate read-only audit verified all 12 result artifact hashes, every source,
protocol and implementation hash, and all four `500 trial x 2 cost` CSV files.
Checkpoint JSON rows and CSV values had zero difference. Trials 0 and 499 were
regenerated from the locked seeds for every variant; mapping and execution-cache
metrics had zero error. Trial 499 was also rerun through the full engine for all
eight variant/cost combinations with zero error. All 36 historical payoff fits
kept their training outcomes strictly before the test block (minimum gap 64h),
and manual frozen-prediction reconstruction error was zero. No 2026 row entered
this audit.

## Exact interpretation boundary

This materially strengthens the historical claim: the combined policy is not
explained by ATR alone, and the TCN+GRU score contains incremental historical
timing information after coarse volatility conditioning. It does **not** yet
prove an independently replicating or causal market signal.

The conditional circular controls do not preserve the full score time series.
Actual raw-score/action lag-one correlations were 0.8717/0.8245; circular-null
means were only 0.4905/0.7237 (1.10) and 0.3800/0.6901 (1.05). Actual turnover
was 315 trades versus null means of 356-379. Consequently the reported Monte
Carlo values are diagnostic sensitivity measures, not exact inferential
p-values. Mean net completed return reduces, but does not remove, turnover and
exposure differences. Global profile/policy adaptivity across the same 2022-25
history remains uncorrected.

## Active checkpoint

Do not run another historical parameter search to improve these numbers. The
evidence belongs to a **block-prequential process**: one causally trained model
for each walk-forward block, its own prior-validation empirical CDF, and a payoff
layer fitted only from outcomes mature before that block. It does not identify
the retired fixed `control_recent3_equal_v2` candidate or the legacy Phase 2
forward lock as a replacement.

The next candidate must therefore lock the complete rolling fit/inference
schedule, produce decisions before labels/forward returns exist, keep an
append-only hash-chained ledger, and compare against the ATR-only control on a
new post-registration window. The failed 2026 Future-OOS window can be training
history after the recipe is locked, but can never be counted as new confirmation
or used to choose parameters. Automatic promotion and live orders remain off.

