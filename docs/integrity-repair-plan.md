# Integrity repair and next research plan

Reviewed: August 30, 2026. This is engineering work, not a new strategy selection.

## Objective and preserved boundaries

Establish repeatable BTCUSDT long-only economic value after executable fills,
fees, slippage, funding and portfolio risk. Do not substitute IC/F1 improvements
for this objective. Neither profitability nor a future validation pass is promised.

Both frozen OOS failures remain unchanged. SWA run `20260824_154330` is archived;
no thresholds, model weights, strategy parameters or acceptance gates were tuned.
Training is paused until a distinct research contract is explicitly approved.

## Implemented repair contract

- `phase2_mtm_v2`: explicit bar open/close and decision times, gap-through stop
  fills, hourly marked equity/exposure/costs, finite input checks, and no overlapping
  positions. v1 bundles explicitly normalize their mislabeled open timestamps.
- Entry/exit fees scale with actual slipped fill notionals. Historical funding
  events can be provided via `--funding-events`; missing inputs are labeled
  duration-based estimates, never silently described as actual funding.
- Intrabar order is conservative stop-first. Intrabar exit time remains unknown
  with hourly OHLC; a bar-close time proxy is disclosed. Hourly marked drawdown
  is not the worst possible intrabar drawdown. MFE/MAE are bar bounds.
- Right-edge positions are marked, not declared completed timeouts. A position
  crossing a data gap is censored, not retroactively sold before the gap; the
  simulation stops there because subsequent balance is unknown. Such runs are
  not complete evidence. Forensics exclude censored positions.
- Realized daily-loss controls retain their original meaning. A marked-equity
  drawdown breach also blocks subsequent entries; it is not a live liquidation
  engine or a guarantee that losses cannot exceed the configured threshold.
- Raw OHLC/finite values are validated; incomplete candles are removed. Labels
  and sequences exclude windows crossing missing hours. Stale 4H context is not
  indefinitely forward-filled; raw prices/timestamps are not imputed.
- Immutable OOS reuse verifies candidate, committed rows/window/metrics, prediction
  consistency and file hashes. A legacy family is admitted only after these
  checks. The result JSON is no longer rewritten with current lifecycle metadata.
- The separate forward ledger verifies identity, continuity, overlapping rows
  and a hash chain. It does not extend the one-shot model evaluation. Old forward
  locks are audit-only under corrected accounting and remain unchanged.
- Current-state instructions close rejected research. Application dependency
  versions are pinned; runtime/hardware manifests still matter for reproduction.
  CI now covers research branches. Raw data/weights/reports are not committed.

## New local market snapshot

`data/raw/snapshots/20260830_integrity_v2/snapshot_manifest.json`

| Dataset | Valid rows | Latest completed boundary (UTC) | Quality |
|---|---:|---|---|
| BTCUSDT 1H | 40,857 | 2026-08-30 09:00 | No gaps |
| BTCUSDT 4H | 10,214 | 2026-08-30 08:00 | No gaps |
| BTCUSDT 15m | 163,427 | 2026-08-30 09:30 | Two May 2022 gaps after 3 zero-activity rows removed |
| Funding | 5,108 | Last event 2026-08-30 08:00 | 2,005 missing mark prices in 2022–2023; none in 2026 |

History begins January 1, 2022. `complete` in the snapshot manifest means the
download completed, not that all historical funding costs can be computed.
Missing funding marks run through October 31, 2023. Exact full-history funding
accounting remains unavailable for those events; do not fabricate prices or
silently use zero funding. Positioning metrics are not refreshed because they
are not used by the retained control. No training, feature rebuild, OOS scoring
or live orders were performed with the new snapshot. Drive was not mounted.

## Diagnostic replay of the already-seen forward window

Frozen scores and policies unchanged; original evidence files untouched.
The corrected local replay uses actual 2026 funding events, June 13–August 15:

| Policy | Trades | Net portfolio return | Hourly marked max drawdown | Profit factor |
|---|---:|---:|---:|---:|
| primary_balanced | 18 | -0.2043% | -0.5336% | 0.6962 |
| challenger_return | 21 | -0.1890% | -0.5500% | 0.7653 |

Reports: `reports/integrity_audit_20260830/`. This is an accounting audit, not a
new clean OOS test. It supplies no permission to tune these policies on this
window. The repaired engine did not reveal hidden profitability.

## Next sequence (no automatic model search)

1. Verify and stage the actual historical out-of-fold prediction cache and
   manifests locally. Current market prices alone cannot replace model outputs.
   Keep both failed OOS windows excluded from parameter/strategy selection.
2. Compare the retained score against no-trade, identical filters/exits without
   model selection, and reproducible matched/randomized entry references on
   eligible historical research data. Report exposure, net expectancy, costs,
   concentration and temporal uncertainty, not only total return.
3. Decide whether the residual problem is absent ranking value, label/payoff
   mismatch, trading frequency, or execution costs. Only then preregister one
   bounded hypothesis. Dynamic TP/SL is a hypothesis, not a default repair.
4. Reuse verified matching cached control artifacts. Any training/data change
   requires a new signature; reporting-only changes must not trigger retraining.
   Do not add a broad model/feature/threshold sweep.
5. Only if historical economic evidence warrants it, lock a new candidate,
   accounting version and unseen confirmation boundary. Append future scores
   separately; do not overwrite completed evaluations or infer readiness from
   a calendar date alone. No live deployment without independent review.

Local refresh command (choose a new directory each time):

```powershell
.venv/Scripts/python -m yenibot.automation.refresh_data --output-dir data/raw/snapshots/<new-id>
```

At this checkpoint there is no reason for the operator to rerun 04/04a/05/07.
The next step is historical economic attribution with verified cached predictions,
not another expensive training cycle.

## Verification

The default CI-equivalent suite passed locally: 308 tests, with the five long
experiment-integration tests excluded as in routine CI. Those five were not
completed during this repair; this is not a claim that a full model-training
cycle was verified. Notebook syntax/contracts and the new integrity regressions
are included in the passing suite.
