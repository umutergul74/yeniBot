# Phase 2 Sandbox Runbook

Status: **implementation branch active; official Phase 2 still gated**

This runbook explains how Phase 2 work can progress while the pinned
Future-OOS candidate is still waiting for enough mature unseen rows.

## Principle

Sandbox Phase 2 is engineering preparation, not model promotion.

Allowed work:

- trade-ledger schema
- next-bar fill mechanics
- conservative same-bar TP/SL ambiguity handling
- fee, slippage, and funding accounting
- portfolio/equity curve plumbing
- report writer
- readiness gate integration
- tests using synthetic fixtures

Blocked work:

- live trading
- order routing
- leverage optimization
- changing the frozen model
- changing the frozen threshold
- using Future-OOS data for tuning
- treating sandbox backtest output as Phase 2 approval

## Modes

`yenibot.phase2` has two intended modes:

```text
sandbox  -> may run before Future-OOS pass; output is non-promotable
official -> fails closed unless Phase 2 readiness and Future-OOS pass
```

The current expected status while waiting is:

```text
evidence_status: sandbox_not_promotable_until_future_oos_passes
official_allowed: false
```

## First Strategy Contract

The initial contract is deliberately simple and pre-registered:

- candidate: `control_recent3_equal_v2`
- side: long-only
- signal: `prob_long` ranking score
- threshold: `0.42674046854178105`
- entry: next-bar open only
- take-profit: `2 x ATR`
- stop-loss: `5 x ATR`
- maximum hold: `10` bars
- same-bar TP/SL ambiguity: conservative stop-first
- overlapping positions: disabled

Any alternative exit, cooldown, overlap, sizing, or threshold policy must be
added as a separate pre-registered strategy variant. It cannot overwrite the
baseline contract after results are known.

## Required Outputs

The sandbox writer emits:

```text
phase2_trade_ledger.csv
phase2_equity_curve.csv
phase2_sandbox_report.json
phase2_sandbox_report.md
```

Every report must include the gate state and whether the result is promotable.

## Command-Line Entry Point

The first implementation can be called with CSV inputs:

```bash
python -m yenibot.automation.phase2_sandbox \
  --report-dir /path/to/reports/experiments/<run_id> \
  --bars /path/to/causal_bars.csv \
  --signals /path/to/frozen_predictions.csv \
  --output-dir /path/to/phase2_sandbox \
  --mode sandbox
```

Official mode is intentionally fail-closed:

```bash
python -m yenibot.automation.phase2_sandbox ... --mode official
```

This command should raise until Phase 2 readiness and Future-OOS promotion
gates pass.

## Promotion Rule

Promotion is impossible unless all are true:

- `phase2_readiness.ready_for_phase2 == true`
- report consistency audit passed
- Future-OOS evaluation completed
- Future-OOS primary candidate passed
- `promotion_allowed == true`
- no Phase2 blockers remain

Until then, Phase 2 work is useful engineering, not deployment evidence.
