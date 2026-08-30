# Frozen-ledger risk/accounting audit v1

Status: protocol recorded before computing reconciled results, August 30, 2026.
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
