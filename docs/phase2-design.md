# Phase 2 Research Backtest Design

Status: **prepared, implementation blocked pending frozen future-OOS pass**

Phase 2 begins only when `phase2_readiness.json` reports
`ready_for_phase2: true`. This document defines the first backtest before its
results are known. It does not authorize execution or live deployment.

## Objective

Determine whether the frozen Phase 1 ranking signal remains economically useful
after causal trade timing, fees, funding, slippage, intrabar ambiguity, and
position overlap are applied.

Phase 2 is not another model-selection surface. It consumes the frozen model
identity and policy from Phase 1 without refitting or threshold tuning.

## Frozen Input Contract

The first Phase 2 run must record:

- frozen candidate id and manifest hash
- source experiment run and model component hashes
- feature ordering and scaler hashes
- signal threshold and threshold source
- prediction timestamp, bar availability timestamp, and execution timestamp
- data fingerprint and backtest configuration hash

The score is a ranking signal named `prob_long` for compatibility. It is not a
calibrated probability. Position sizing must not interpret `0.60` as a 60%
success probability.

## Causal Timing

- Generate a decision only after the complete 1H bar is available.
- Aligned 4H inputs must already satisfy the four-hour completion shift.
- Execute no earlier than the next tradable bar after the decision.
- The default research fill is next-bar open plus configured adverse slippage.
- Never fill at the close used to create the signal.
- Record decision, order, and fill timestamps separately.

## Initial Strategy Contract

- BTCUSDT perpetual futures, long-only.
- At most one open position.
- Entry uses the frozen Phase 1 policy without threshold optimization.
- Initial exit mirrors the label contract: `2 x ATR` take-profit,
  `5 x ATR` stop-loss, and 10-bar maximum holding period.
- ATR is the causal value available at the decision timestamp.
- If stop and take-profit are both touched inside one bar and ordering cannot be
  reconstructed, use the conservative stop-first outcome.
- A new signal while a position is open is ignored in the first baseline run.

Alternative exits, cooldowns, sizing rules, or overlapping positions are later
experiments and require separate preregistration.

## Cost And Carry Model

The baseline backtest must itemize rather than combine:

- entry fee
- exit fee
- entry slippage
- exit slippage
- funding paid or received while the position is open
- liquidation or margin constraints, when leverage is introduced later

Fee and funding inputs must come from versioned data or explicit configuration.
The first run must include at least base, optimistic, and adverse cost
scenarios. Zero-cost results are diagnostic only.

## Required Outputs

### Trade Ledger

One row per completed trade:

- signal, order, entry, and exit timestamps
- entry/exit prices and fill assumptions
- score, threshold, ATR, and regime probabilities at entry
- exit reason
- gross return
- each cost component
- net return
- holding bars
- maximum favorable and adverse excursion

### Portfolio Series

- hourly marked-to-market equity
- cash and exposure
- gross and net returns
- cumulative fees, slippage, and funding
- drawdown and underwater duration

### Performance Evidence

- trade count and exposure
- hit rate and payoff ratio
- expected net return per trade
- profit factor
- annualized return and volatility
- Sharpe and downside-risk measures
- maximum drawdown and recovery duration
- turnover
- bootstrap confidence intervals using temporal blocks
- results by year, market regime, score band, volatility state, and cost
  scenario

All metrics must be shown gross and net of costs. A pooled headline metric
cannot replace temporal breakdowns.

## Robustness And Failure Tests

- next-bar execution delay stress
- wider slippage and fee stress
- adverse same-bar barrier ordering
- funding inclusion and exclusion comparison
- threshold perturbation for sensitivity only, never winner selection
- year and regime concentration
- removal of the best trades and best month
- moving-block bootstrap of net trade returns
- stale or missing prediction handling

## Phase 2 Decision Boundary

The first backtest is evidence, not deployment approval. Live development
requires a separately reviewed charter covering at least:

- positive net expectancy after adverse costs
- adequate trade count and temporal coverage
- acceptable drawdown
- no single-period or single-regime dependence
- stable result under execution-delay and cost stress
- immutable data, model, and configuration manifests

Numerical live gates must be committed before the first Phase 2 result is read.

## Explicit Exclusions

The first Phase 2 implementation must not include:

- live exchange connectivity
- order placement
- leverage optimization
- Kelly sizing
- short trades
- XGBoost or another model layer
- threshold search on frozen future-OOS data
- discretionary removal of losing trades or periods

