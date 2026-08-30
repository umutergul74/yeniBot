# Block-prequential forward shadow v2

Status: protocol locked before model preparation or post-lock performance,
August 30, 2026. No candidate is sealed yet and no confirmation clock has begun.

## Why this is a new candidate

The retired `control_recent3_equal_v2` averaged three fixed raw sigmoid outputs.
The strongest later historical evidence came from a different process: each
walk-forward block's single TCN+GRU checkpoint, that checkpoint's own preceding
validation-score CDF, and a payoff ridge trained only on mature earlier OOF
outcomes. Renaming the retired ensemble would mismatch the tested mechanism.

The machine contract is `configs/forward_shadow_v2.json`. Its 2022-25 inputs are
hash-pinned. The June-August 2026 failed Future-OOS window is already seen: it may
enter a later model's fixed training window, but can never be counted as v2
confirmation, used to select parameters, or rescored as if unseen.

## Fixed block process

Each candidate block contains 720 chronological hours. The first 63 are sequence
burn-in and the next 657 are evidence decisions, matching the historical fold
semantics. Before a block starts, one model is fitted with:

- 5,040 chronological train bars;
- 24 purged bars;
- 1,080 validation bars;
- 6 embargo bars;
- 64 post-fit audit bars used only for prediction parity, never checkpoint
  selection;
- at least 72 hours between preparation and the aligned 720-hour context block,
  so the sealed manifest can be reviewed, committed and pushed before any row
  can acquire confirmation status;
- preparation must still leave at least 24 hours between the final manifest
  lock and that context block; a late or long-running preparation aborts closed;
- the retained feature profile, model, loss, optimizer and deterministic
  `project seed + block ordinal` rule.

The final selected checkpoint generates a sorted validation-score reference.
Future raw scores are transformed with the fixed right-sided empirical CDF.
No label, forward return or future OHLC may enter inference.

The payoff layer keeps alpha 10 and exactly three inputs: score percentile,
decision ATR/close and their product. An ATR-only ridge uses the same target rows
with the score terms fixed at zero. Initial fits use only the pinned 2022-25 OOF
opportunities. A later ridge/model update can occur once, before the next 720h
block, under the same algorithm and only with outcomes mature before that block.
There is no mid-block refit, history-window choice, threshold choice or exit
choice.

Entry requires predicted adverse-net utility strictly above zero. Signal
accounting retains next-open, TP 2 ATR, SL 5 ATR, ten-bar maximum and stop-first.
Because next-open assumes effectively zero computation/order latency, this is
model-signal evidence only. Live readiness separately requires latency, spread,
liquidity, mark-price and rejected-order validation.

## Seal and ledger boundary

Preparation must write model/scaler/HMM hashes, validation-CDF hash, both ridge
fits, training membership/cutoff, code/config/protocol hashes and runtime identity
to one immutable manifest. A separate v2 lock then pins that manifest and its
Git commit. Confirmation begins strictly after both exist. Pre-lock rows and
post-hoc backfills are excluded even if deterministic.

The ledger records raw model score, CDF percentile, candidate/ATR utility and
actions, model/block identity, feature snapshot hash, source-bar time, decision
time and generation time in an append-only hash chain. It must be possible to
distinguish timely ex-ante scores from later deterministic batch replay. Batch
replay can support frozen model evidence; it cannot establish execution latency.

The preparation runner is
`yenibot.automation.phase2_forward_shadow_prepare`. It publishes a block only
after exact saved-artifact/label-free parity. The exact manifest must then be
committed and pushed before
`yenibot.automation.phase2_forward_shadow_register` creates its registration
proof. `yenibot.automation.phase2_forward_shadow_score` refuses ledger writes
without that proof, strips any outcome columns, and scores only already closed
source bars. Late rows remain audit-visible as `sealed_batch_replay` but never
count as timely confirmation evidence.

## Decision horizon and gates

There is no early success. Looks after blocks 3, 6 and 9 are monitoring only.
The first confirmation decision requires all of:

- at least 12 complete blocks, 360 days and 100 candidate trades;
- positive candidate base and adverse returns;
- base/adverse profit factors at least 1.10/1.05;
- at least two-thirds positive adverse-return blocks;
- positive lower 95% paired block-return differences versus ATR-only under both
  costs, for both fixed two- and three-block bootstrap lengths;
- a positive moving-block lower 95% bound for score/payoff Rank IC;
- candidate adverse net bps per occupied hour not below ATR-only;
- hourly marked drawdown no worse than -15%;
- complete common-cohort, artifact and chronology integrity.

The paired checks use 10,000 fixed-protocol bootstrap replicates. No gate may be
dropped after seeing the result. A failure retires the exact v2 candidate and
requires a new hypothesis and boundary. A pass still permits only reviewed
paper/shadow progression: automatic promotion and live orders are false.

## Implementation order

1. Add label-free inference and prove parity against saved historical scores.
2. Add immutable prepare/seal tooling and a richer append-only ledger.
3. Add candidate-versus-ATR evaluator with corrected marked accounting.
4. Add a single Notebook 08 entrypoint: GPU prepare only when a new block needs
   a model; CPU append/evaluate otherwise.
5. Run preparation, commit the concrete manifest/lock hash, and only then begin
   collecting v2 evidence.

Notebook 04/05/07 do not implement this candidate. The old forward lock stays
unchanged as historical audit evidence.
