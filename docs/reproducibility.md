# Reproducibility Guide

## Purpose

This document defines the minimum steps required to reproduce a yeniBot Phase
1 experiment without changing its temporal meaning.

The exact model result depends on:

- the git commit
- `config.yaml`
- raw data cutoff
- processed feature and label files
- feature ordering
- fold definitions
- random seed and deterministic mode
- model, scaler, HMM, and threshold artifacts

An experiment is not reproducible if only its final metric table is retained.

## Environment

The canonical environment is Google Colab with:

- a CUDA runtime for notebook 04
- Google Drive mounted at `/content/drive`
- the repository cloned to `/content/yenibot_repo`
- dependencies installed from `requirements.txt`

During the current replacement-candidate cycle, notebooks default to
`research/next-candidate-v1`, verify the checked-out branch, and print the
exact commit. This prevents a Colab runtime left on `main` from silently
running older notebook logic.

After every pull:

1. Use `Runtime -> Restart session`.
2. Run setup cells from the beginning.
3. Confirm the printed commit and configuration.

Restarting execution without restarting the Python session is insufficient
because imported modules remain cached.

## Notebook Dependency Graph

```text
01 raw klines
  -> 02 processed features
  -> 03 labeled rows
  -> 04 fold artifacts, historical recency research, and run handoff
  -> 05 diagnostics, manifests, and bundles for that exact run
```

Do not rerun more expensive stages unless their inputs changed:

| Changed input | Minimum rerun |
|---|---|
| Diagnostics code only | `05` |
| Training/model/profile selection | `04 -> 05` |
| Labels or ATR semantics | `03 -> 04 -> 05` |
| Feature formulas or columns | `02 -> 03 -> 04 -> 05` |
| Raw source or date range | `01 -> 02 -> 03`, then required downstream stages |

Notebook 04 writes `notebook04_run.json` under the Drive checkpoint root only
after its configured recency research completes. Notebook 05 consumes this
handoff before falling back to the most recent experiment directory. This
avoids diagnosing a different run when training artifacts were reused from an
older matching run id.

## Data Integrity

Notebook 01 must verify:

- full Binance kline schema
- nonzero taker-buy data
- valid trade counts
- no duplicate timestamps
- configured handling of zero-volume/no-trade rows
- gap sizes relative to the interval-specific policy

The raw and processed parquet files belong on Drive and must not be committed.

## Temporal Integrity

Reproduction must preserve:

- causal rolling windows
- completed-bar availability timestamps
- the four-hour forward shift for 4H features
- train-only scaler fitting
- purged and embargoed fold boundaries
- forward-only HMM inference outside training
- tail removal for labels requiring future bars

Changing any of these creates a different experiment even when feature names
and model architecture remain unchanged.

## Artifact Identity

Every reusable training scope records a signature. Frozen future-OOS
candidates additionally record content hashes for:

- model weights
- scaler
- HMM
- ordered feature list
- training signature
- threshold payload
- source run
- fit cutoff

Future-OOS evaluation must verify those hashes and perform no fitting.

Run the read-only preflight before frozen scoring:

```bash
python -m yenibot.automation.future_oos_preflight \
  --config config.yaml \
  --checkpoint-dir <checkpoint_dir>
```

For the current frozen cycle, refresh with `01 -> 02 -> 03 -> 05`. Notebook
`04` is forbidden because it creates fitted artifacts. See
[`future-oos-runbook.md`](future-oos-runbook.md).

## Evidence Interpretation

Use fold-macro metrics for active charter gates. Pooled-row metrics are useful
diagnostics but are not interchangeable with equal-weight fold estimates.

For dependent market data:

- resample folds as clusters
- use moving blocks inside sampled folds
- inspect more than one block length
- report confidence intervals and probability-above-gate

Do not describe sigmoid scores as probabilities unless Brier and log-loss skill
beat fold-specific climatology and calibration diagnostics are stable.

## Sharing Results

Prefer the slim bundle produced by notebook 05. It contains the evidence
needed for routine review without model weights or large prediction payloads.

Use a full bundle only when row-level or embedding-level analysis is required.
Do not commit either bundle to git.

For every reviewed run, record:

- run id
- git commit
- data start/end
- profile and feature count
- fold scope
- training signature
- active validation charter
- report completeness
- decision and rejection reason
