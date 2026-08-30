# Phase 2 economic attribution

Reviewed: August 30, 2026.

## Purpose

Phase 1 metrics answer whether a frozen score ranks labels or forward returns.
They do not establish that the score adds executable value after fills, costs,
position overlap and market-regime filters. This audit measures that incremental
contribution before any further TP/SL or threshold research.

The implementation is fail-closed:

- accepts one explicit `test` split only;
- verifies candidate identity, frozen threshold, unique UTC decisions, OHLC/ATR
  validity and source hashes;
- evaluates disjoint walk-forward folds independently, never carrying a position
  through the 64-hour fold embargoes;
- excludes censored end-of-fold positions from completed-trade returns;
- runs the same causal execution contract and base costs after shuffling scores
  within each fold and calendar month;
- preserves the score distribution and above-threshold count in every shuffle
  group, so a control cannot win or lose merely by changing selection frequency;
- reports an always-on long context and inverted model ranking;
- applies Holm family-wise correction when several registered strategies are
  audited together;
- never promotes, selects or permits live trading from this seen window.

Code:

- `yenibot.phase2.economic_attribution`
- `yenibot.automation.phase2_attribution`

## First bounded audit

Source: the immutable three-fold replacement-control test cache bundled with the
Phase 2 sandbox (`folds 40-42`, 1,971 hourly decisions, February 27-May 25,
2026). This is already-seen retrospective evidence, not an independent test.
There were 100 rank-destroyed controls per strategy; the minimum attainable raw
p-value was approximately 0.0099.

| Strategy | Base return | Adverse return | Trades | Positive folds | Null median | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline fixed ATR | -10.08% | -25.47% | 187 | 1/3 | -9.03% | 0.594 | 1.000 |
| score margin 0.07, 7 bars, TP2/SL4 | +2.72% | -4.64% | 78 | 1/3 | -9.79% | 0.079 | 0.238 |
| score margin 0.04, ATR 0.7%-1.0%, 6 bars, TP1.5/SL4 | +5.31% | -0.03% | 56 | 2/3 | +6.95% | 0.614 | 1.000 |

No strategy passed the diagnostic gate.

The narrow ATR-band strategy is the most important negative finding. Its
positive historical return is not evidence of model contribution: rank-destroyed
selection had a higher median return (+6.95%), inverted model ranking returned
+6.99%, and the same filters with always-on long scores returned +17.43% under
base costs. The regime/exit contract, not the model ranking, produced the seen
window result. Its adverse return was slightly negative and only 56 trades were
completed.

The score-margin 0.07 strategy is the only current lead that showed directional
model contribution: +2.72% actual versus a -9.79% null median, and -4.49% under
inverted ranking. That is not sufficient evidence. It failed adverse costs,
minimum trade count, fold breadth, raw significance and family-wise significance.
It was also identified on the same seen window, so even a stronger retrospective
result could only justify a new confirmation contract.

The common score diagnostics were mildly positive (score/forward-return RankIC
0.0405 and top-minus-bottom forward-return spread 0.00137). Those statistical
effects did not translate into a robust executable strategy.

## Interpretation

The current TCN+GRU is useful as a weak ranking component, but it has not proved
that it can support a deployable trading bot. Adding more TP/SL variants now
would increase multiple-testing risk and can make the backtest look better
without strengthening model contribution. Dynamic exits remain closed until the
ranking contribution gate is passed.

The repaired June 13-August 15 replay reached the same practical conclusion for
the two previously locked ATR-band forward policies: -0.2043% and -0.1890% with
actual 2026 funding. Those results remain immutable and cannot be used to tune a
replacement.

## Single next research decision

The full-cache follow-up is now complete. The 119 MB OOF artifact was verified
locally; same-fold past-validation percentiles avoided transferring a later raw
threshold backwards in time. Four densities were evaluated at unchanged exits.
None passed the economic gate. q80/q90 were weakly positive under base costs but
strongly negative under adverse costs and insufficiently broad across folds.

Ordinary shuffling increased realized turnover, despite preserving selected-row
counts. The separate `yenibot.automation.phase2_oof_serial_control` corrects this
confound using within-fold/month cyclic shifts, reference-engine verification,
explicit turnover reporting and conservative two-null/family-wise comparison.
Historical model contribution survived, not deployable utility. Shift-invariance
assumptions and historical selection bias remain. All figures, source hashes and
limitations are in the [full OOF checkpoint](full-oof-research-checkpoint.md).

The full-cache gate remains exactly the code defaults:

- positive base and adverse-cost compounded return;
- one-sided rank-destroyed permutation p <= 0.05 after family-wise adjustment;
- at least 100 completed trades;
- at least two-thirds positive walk-forward folds;
- positive score/forward-return RankIC and top-minus-bottom spread;
- complete execution data contract.

The fixed-density panel is closed as a deployable trade-entry engine, but the
score is not wholly uninformative. The subsequent single
[validation net-utility hurdle v1](validation-net-utility-hurdle-v1.md) probe also
completed and failed adverse costs, breadth and paired uncertainty. It exposed
poor payoff-calibration transfer and verified reuse of checkpoint-selection
validation data. Next audit genuinely earlier-OOF calibration sources before
any separate model protocol. Do not optimize more exits or reopen either family.
Only robust historical evidence could justify a post-lock unseen confirmation.

## Local command

```powershell
.venv/Scripts/python -m yenibot.automation.phase2_attribution `
  --report-dir reports/phase2_local_lab/final_third_wave_local/local_gate `
  --bars reports/phase2_forward/current_locked/bundle_extract/phase2_sandbox/phase2_bars.csv `
  --signals reports/phase2_forward/current_locked/bundle_extract/phase2_sandbox/phase2_signals.csv `
  --input-manifest reports/phase2_forward/current_locked/bundle_extract/phase2_sandbox/phase2_input_manifest.json `
  --output-dir reports/phase2_economic_attribution/<run-id> `
  --incumbent-suite `
  --permutations 100
```

Use 100 permutations for bounded triage. Only a locked candidate close to or
through every non-statistical gate warrants a larger confirmatory permutation
count. Reports are ignored local artifacts and record all source hashes.
