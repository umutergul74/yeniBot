# Phase 2 Clean Forward Confirmation

Status: **legacy lock preserved; historical audit only** (August 30, 2026)

The frozen Phase 1 candidate failed. This v1 lock also predates the corrected
accounting contract and admits decisions before its own registration date.
It remains readable for audit, but cannot produce a clean-confirmation pass.
Do not change its hash, cutoff, policies or failure history to repair that.
A new candidate needs a newly reviewed lock, pinned `phase2_mtm_v2` accounting
and a defensible unseen boundary after registration/selection.

Continuous future inference can append to `forward_predictions.jsonl` using
`phase2.prediction_ledger.append_forward_predictions`. It verifies overlap,
model identity, hourly continuity and a content hash chain. This ledger is
separate from the immutable `future_oos_predictions.parquet` outcome. The
forward runner prioritizes the ledger when present; it never extends a
completed model evaluation. Generating a new candidate's scores still requires
its actual verified model artifacts; market data alone is not predictions.

The following describes the historical v1 design, not current authorization.

This workflow is separate from the already-seen Phase 2 sandbox. It consumes
only frozen-model decisions strictly after `2026-06-13T01:00:00Z`, performs no
fit or strategy-selection operation, and cannot promote automatically.

## Immutable Lock

The committed lock is
`yenibot/phase2/forward_lock.json`. Its canonical hash is verified before every
run. A parameter change invalidates the hash and stops the evaluator.

Locked candidates:

| Role | Strategy |
|---|---|
| Primary balanced | `score_margin_04_atr_band_007_010_time_stop_6bar_tp15_sl4_v1` |
| Return challenger | `score_margin_07_atr_band_005_010_time_stop_6bar_tp2_sl4_v1` |

The primary is evaluated first. The challenger is reported independently and
cannot replace it on the same clean window.

## Portfolio Risk Contract

The fixed risk policy is intentionally conservative and is not optimized:

- initial research equity: `10,000`
- equity risk budget per trade: `0.25%`
- notional cap: `25%` of equity
- leverage: disabled
- daily realized-loss entry lock: `1%`
- permanent realized-drawdown entry lock: `5%`

Position notional is:

```text
min(0.25, 0.0025 / (initial_stop_distance_fraction + stressed_round_trip_cost))
```

The sizing cost uses the active cost scenario and maximum allowed holding
duration, so a stopped trade's fees, slippage, and funding are included in the
risk budget. The model score is never treated as a calibrated win probability.

## Clean Success Gates

A robustness decision is withheld until there are at least `75` completed
trades and `90` days of clean coverage. The locked primary must then satisfy:

- positive base-cost compounded return
- positive adverse-cost compounded return
- base profit factor at least `1.10`
- moving-block bootstrap probability of positive compounded return at least
  `0.65`
- maximum drawdown no worse than `-10%`
- non-negative return after removing the best month

Passing these numerical checks still does not bypass the Phase 1 Future-OOS
and report-consistency gates.

## Local One-Command Run

Notebook 05's slim bundle is the preferred clean input because
`future_oos_predictions.parquet` now carries causal OHLC and ATR columns:

```bash
python -m yenibot.automation.phase2_forward \
  --bundle "C:/Users/Umut/Downloads/phase1_latest_experiment_slim_bundle.zip" \
  --output-dir reports/phase2_forward/current
```

An extracted Phase 1 report is also accepted:

```bash
python -m yenibot.automation.phase2_forward \
  --input-dir /path/to/reports/experiments/<run_id> \
  --report-dir /path/to/reports/experiments/<run_id> \
  --output-dir reports/phase2_forward/current
```

The older `phase2_latest_sandbox_bundle.zip` remains valid as an audit input,
but all of its decisions end before the lock boundary. The correct result for
that bundle is `waiting_for_post_anchor_data`.

## Outputs

```text
phase2_forward_lock_snapshot.json
phase2_forward_boundary_report.json
phase2_forward_bars.csv
phase2_forward_signals.csv
phase2_forward_decision.json
phase2_forward_decision.md
candidates/<role>/<base|adverse>/
```

Every report records the lock hash, accepted time range, source fingerprint,
zero fit operations, zero selection operations, position size, equity before
and after each trade, and which locked gates remain unsatisfied.
