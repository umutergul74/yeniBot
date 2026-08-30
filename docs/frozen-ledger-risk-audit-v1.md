# Frozen-ledger risk/accounting audit v1

Status: completed August 30, 2026. The original pre-computation protocol follows.
This is an accounting/identification audit, not another candidate or a gate retune.

## Immutable inputs and comparisons

- Source probe result SHA-256:
  `41237d1a3298276b60add605c90cc3d4cef5f63bf2716c16e50f95516d560524`.
- Recheck all 68 artifacts pinned by that result, including both policy ledgers.
- Use the same pinned full OOF hourly prices and exactly folds 2-37. Never use
  later 2026 outcome rows. Include all zero-trade folds.
- Compare only the frozen candidate and ATR-only context control; no entry/exit,
  threshold, position-size, training, signal or model selection changes.
- Keep original completed-trade summaries next to the accounting diagnostics;
  no historical source report may be overwritten or relabeled a success.

## Accounting contract

1. Reconstruct independent-fold reference-notional cashflows from the saved
   trades. No trade engine rerun, new signals or fit operations. Assert restored
   original completed returns against the root result for each policy/cost.
2. Report hourly **contract-price close** marked equity, not just trade-close
   drawdown. This is not exchange mark-price liquidation accounting, intrabar
   worst-case loss, or a continuous live portfolio across the fold gaps.
3. Report terminal positions both at their saved mark without exit costs and
   with a separately labeled hypothetical final-close liquidation charging exit
   fees and slippage. This is an accounting sensitivity, not a new trading policy.
4. Concatenate independent-fold curves by chaining their end values; preserve
   fold identity and gaps. Do not annualize returns/Sharpe or claim gap exposure
   was traded. Retain original unit-notional exposure, without hindsight sizing.
5. Compare occupied hours / observed fold hours, initial-stop risk fractions,
   R-multiples, per-hour simple payoff and actual funding charges. These are
   descriptive exposure/risk comparisons, not equal-risk causal identification.

## Historical funding and missing marks

Funding snapshot: `data/raw/snapshots/20260830_integrity_v2/btc_funding_rates.parquet`,
SHA-256 `25119a0e1b66f09709091ad7dbe685810f2271476507c015a94c3b41881658dc`.
The original REST events cover the historical range, with small timestamp jitter.
Require the complete eight-hour event grid over this BTCUSDT audit interval;
preserve raw event timestamps for charges. Do not round settlement timestamps
or silently replace a missing rate with zero.

Read-only discovery found 34 missing held-event mark prices for the candidate and
11 for ATR-only. A fresh official funding endpoint sample still returns blank
mark prices in January 2023. The official one-minute mark-price kline endpoint
does have data for a sampled missing event.

Retrieve ONLY the union of missing held-event one-minute mark bars, with raw
responses and hashes retained. Do not call the minute open/close the exact
settlement mark. Use low/high as an explicitly approximate price-interval
sensitivity under the assumption that the settlement mark lies in its reported
minute. Respect negative funding signs when forming favorable/adverse charge
bounds. Known settlement marks remain exact reported values, not replaced by
minute proxies. Missing intervals or rate coverage fail closed.

Funding charge is rate * settlement mark / entry reference price; held interval
is entry-inclusive, exit-exclusive, using the original intrabar exit close proxy.
That exit-time ambiguity remains and must be disclosed. Show both interval ends
under both original fee/slippage schedules. These are **historical-rate funding
sensitivities**, not complete exact exchange funding. Keep the original fixed
adverse-rate stress case; do not select whichever funding scenario is favorable.

## Decision and limitations

The previous failed acceptance gate remains failed regardless of this audit.
Report accounting reconciliation, missing-data coverage and whether apparent
advantage depends on exposure, a few trades, or costs. No new p-value family,
parameter grid or promotion is authorized by this audit. Genuine unseen
confirmation and execution validation remain necessary for the overall goal.

Primary source: [Binance USD-M market-data documentation](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data),
funding-rate history and mark-price klines. The former associates `markPrice`
with the settlement charge; the latter is bar data, not a replacement exact fill.

## Completed audit

Implementation commit `a6fdcba` was pushed before the run. Exec session `29757`
completed with exit code 0; no process is being waited on. Output:
`reports/phase2_economic_attribution/20260830_frozen_ledger_risk_v1`.
No training, signal selection, exit change or risk sizing was performed.

All original 68 probe artifact hashes were verified before computation. Coverage
contained 3,233 eight-hour funding events; 360 distinct events intersected at
least one policy's holding interval. The union had 35 missing settlement marks.
All 35 required official mark-price minutes were retrieved and raw responses
retained. Prices remain interval sensitivities, not fabricated exact settlements.

### Accounting-reconciled results

All rows below charge hypothetical terminal exits and chain the same independent
36 folds. Drawdown includes hourly contract-close marks. Percentages are whole
historical-period returns, not annualized or continuous-live returns.

| Policy | Funding assumption | Base return | High-fee/slippage return | Base hourly DD | High-fee/slippage hourly DD |
|---|---|---:|---:|---:|---:|
| Candidate | Original fixed funding schedules | +146.49% | +78.31% | -20.84% | -24.16% |
| Candidate | Historical rates, adverse mark-interval end | +138.75% | +85.13% | -21.00% | -23.62% |
| ATR-only | Original fixed funding schedules | +55.09% | +34.19% | -12.44% | -13.50% |
| ATR-only | Historical rates, adverse mark-interval end | +52.39% | +36.03% | -12.47% | -13.28% |

Historical funding can lower the original adverse cost because that original
scenario assumed 2 bps paid every eight hours. It must not replace or erase that
stress scenario. Under the stated minute-range assumption, the favorable/adverse
candidate return endpoints differed by less than 0.0006 percentage points;
this narrow sensitivity does not remove settlement-time or data-source ambiguity.

Original candidate completed-only returns remain +149.01%/+80.65%, with three
terminal positions excluded and trade-close DD -19.00%/-22.58%. The corresponding
terminal-marked (no exit charge) returns are +146.93%/+78.85%, with hourly DD
-20.84%/-24.16%. Thus the bookkeeping corrections reduce the reported base
gain and reveal more drawdown, but do not explain away historical profitability.
No original report was modified.

### Exposure and model identification

| Descriptive measure | Candidate | ATR-only |
|---|---:|---:|
| Occupied hours, including terminal positions | 2,777 | 1,246 |
| Share of the 23,652 observed fold hours | 11.74% | 5.27% |
| Mean saved initial-stop loss / entry notional | 6.49% | 8.06% |
| 95th percentile saved initial-stop fraction | 10.10% | 11.27% |
| High fees + historical-rate adverse-price-end net bps / occupied hour | 2.411 | 2.757 |
| Same scenario mean net R-multiple | 0.02579 | 0.02657 |

The candidate is exposed for **2.229 times** as long. It earns more total money
in this historical chained simulation, but not more simple payoff per occupied
hour than the context-only control. R-multiples use each saved initial stop, not
the current configuration's ATR multiplier, and include the explicitly handled
terminal positions. These are descriptive reference-notional comparisons, not
equal-risk portfolio estimates or new significance tests. The initial-stop risk
fractions are much too large to mistake this unit-notional research ledger for
a finished capital-risk policy.

This supports a narrower conclusion than "TCN is highly successful": the combined
historical policy has positive net utility, while some of the higher total return
comes with more opportunity coverage/exposure. Robust incremental score value
conditional on volatility remains unproved. The earlier failed gates remain
failed; neither this audit nor its positive return promotes a candidate.

### Verification and reproducibility

- 44 targeted tests passed initially; the later portable-path guard adds one
  further test. The 14 new audit tests passed after that addition.
- 24 accounting curves were independently checked against their ledgers; exact
  entry/exit times and prices matched the frozen source in every variant.
- End equity, compounded trade cashflows and hourly drawdown reconstructed.
  Cash-equivalent plus exposure error was at most 4.44e-16.
- All 86 original audit artifact hashes were verified. The subsequent checks are
  appended separately as `post_run_reconciliation.json`.
- Audit result SHA-256:
  `9702f455f5dbec79390856278e856d457e0d1f7ff85744720ecaceb660c47376`.

Runner: `yenibot.automation.phase2_ledger_risk_audit`, using the same scope/probe
paths as the earlier probe and `--output-dir` pointing at the audit directory.
Completed output is immutable. Incomplete data acquisition can resume only with
matching source/implementation identity; cached minute responses are reused.
A post-run portability guard normalizes historical Windows artifact separators
for Colab and rejects bundle path escapes. This changes neither the completed
audit nor its pinned implementation identity; new manifests use POSIX separators.

## Resume here

Previous goal turn: concrete progress, not blocked. All jobs above are terminal.

Update: the fixed [conditional score attribution](conditional-score-attribution-result-v1.md)
has now completed and passed its retrospective diagnostic. Its result document
supersedes the steps below as the active checkpoint; the original plan remains
for audit chronology.

1. Do not rerun/refit the closed candidate or optimize to clear its failed gate.
2. The next bounded research question is **incremental TCN score contribution
   after controlling for ATR/volatility context**, using the same frozen fits and
   historical decisions. Write the exact conditional-control protocol BEFORE
   computing its performance. Do not simply shuffle the fitted utility score:
   that also destroys the volatility context that independently earns returns.
3. Specify how the control preserves score/ATR dependence, local score serial
   structure and turnover; quantify frozen/sparse groups and approximation limits.
   Do not choose a favorable control/binning/cost result after seeing outcomes.
   Any positive result remains retrospective with global profile-selection bias.
4. True exchange mark-price/intrabar drawdown, execution latency/liquidity and an
   appropriately capital-sized portfolio remain outside this audit. They and a
   separately locked unseen confirmation are still required before live readiness.
5. No Colab notebook or live order is required for the next local attribution step.
