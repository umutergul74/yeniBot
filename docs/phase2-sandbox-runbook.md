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
- signal queueing: disabled; signals whose immediate next bar occurs while a
  prior position is still open are ignored, not delayed into a future entry
- data-gap guard: entries delayed by more than `1.5` hours are rejected; open
  positions are closed at the last observed close before an internal gap
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
phase2_bars.csv
phase2_signals.csv
phase2_input_manifest.json
phase2_trade_ledger.csv
phase2_equity_curve.csv
phase2_sandbox_report.json
phase2_sandbox_report.md
phase2_cost_scenario_summary.csv
phase2_trade_ledger_all_costs.csv
phase2_equity_curve_all_costs.csv
phase2_strategy_registry.json
phase2_strategy_variant_summary.csv
phase2_strategy_variant_by_fold.csv
phase2_trade_ledger_all_variants.csv
phase2_forensics_summary.json
phase2_bootstrap_summary.json
phase2_exit_reason_forensics.csv
phase2_fold_forensics.csv
phase2_month_forensics.csv
phase2_score_band_forensics.csv
phase2_holding_forensics.csv
phase2_signal_funnel.csv
```

Every report must include the gate state and whether the result is promotable.
The optimistic, base, and adverse cost scenarios are written separately; the
root report remains the base scenario for compatibility.

The strategy registry contains four bounded contracts. The baseline remains the
root report. Dynamic-exit outputs are stored under
`strategy_variants/<strategy_id>/<cost_scenario>/`. No command ranks, selects,
or promotes a winner from the current test window.

## Colab Entry Point

Use `notebooks/06_phase2_sandbox_backtest.ipynb` for Phase 2 work. It does not
call the Phase 1 diagnostics orchestrator, does not import the training stack,
and performs zero fit operations. Notebook 05 remains the Future-OOS and Phase
1 diagnostics notebook.

## Command-Line Entry Point

Preferred Colab path: let the sandbox build inputs directly from the frozen
candidate manifest and its pinned `predictions_all` artifact:

```bash
python -m yenibot.automation.phase2_sandbox \
  --report-dir /content/drive/MyDrive/yeniBot/reports/experiments/<run_id> \
  --checkpoint-dir /content/drive/MyDrive/yeniBot/checkpoints \
  --output-dir /content/drive/MyDrive/yeniBot/reports/experiments/<run_id>/phase2_sandbox \
  --mode sandbox \
  --all-cost-scenarios \
  --strategy-suite
```

By default this uses `split=test` from the frozen `predictions_all` file. Use
`--split all` only when intentionally auditing the complete prediction timeline.

The adapter writes:

```text
phase2_bars.csv
phase2_signals.csv
phase2_input_manifest.json
```

The manifest records the source prediction path, frozen candidate hash,
threshold, row counts, and duplicate timestamp handling. This makes the
sandbox reproducible without changing the Phase 1 or Future-OOS gate.

Manual CSV inputs are still supported:


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
