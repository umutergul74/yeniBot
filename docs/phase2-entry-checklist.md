# Phase 2 Entry Checklist

Official/promotable Phase 2 implementation may start only when every blocking
item is checked. A gated sandbox may exist earlier for schema, accounting,
cost, and reporting preparation, but its output is non-promotable until this
checklist passes.

## Phase 1 Evidence

- [ ] `phase2_readiness.json` has `ready_for_phase2: true`
- [ ] `phase2_readiness.json` has no blockers
- [ ] report consistency audit passes
- [ ] active validation charter is `v4_evidence`
- [ ] leakage and stationarity checks pass
- [ ] seed coverage and same-seed reproducibility pass

## Frozen Future OOS

- [ ] at least 720 mature post-anchor labeled rows exist
- [ ] frozen preflight passes with zero fit operations
- [ ] candidate id is `control_recent3_equal_v2`
- [ ] manifest hash matches the committed expected hash
- [ ] future-OOS evaluation completed
- [ ] frozen primary candidate passed every preregistered gate
- [ ] no profile, threshold, model, scaler, or policy was changed after anchor

## Artifact Freeze

- [ ] source run id recorded
- [ ] model, scaler, HMM, and feature-order hashes recorded
- [ ] data fingerprint recorded
- [ ] threshold and threshold source recorded
- [ ] Phase 1 decision bundle archived

## Phase 2 Preregistration

- [ ] signal timing and next-bar fill rule committed
- [ ] entry and exit rules committed
- [ ] same-bar TP/SL ambiguity rule committed
- [ ] fee, funding, and slippage sources committed
- [ ] cost stress scenarios committed
- [ ] portfolio accounting rules committed
- [ ] required trade-ledger schema committed
- [ ] required performance and robustness outputs committed
- [ ] numerical Phase 2 success gates committed before results are read

## Authorization

When all boxes above are satisfied, change this document's status in a reviewed
commit and implement the research backtest. Do not add live execution in the
same change.
