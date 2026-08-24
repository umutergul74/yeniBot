# yeniBot

[![Phase](https://img.shields.io/badge/phase-1%20model%20validation-16794b)](#current-status)
[![Phase 2](https://img.shields.io/badge/phase%202-future%20OOS%20pending-c47b13)](#current-status)
[![CI](https://github.com/umutergul74/yeniBot/actions/workflows/ci.yml/badge.svg)](https://github.com/umutergul74/yeniBot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-TCN%20%2B%20GRU-EE4C2C?logo=pytorch&logoColor=white)](yenibot/models/hybrid.py)

Bias-aware market-microstructure research for a long-only BTC/USDT
perpetual-futures sequence model.

`yeniBot` is a Phase 1 quantitative ML research system. It downloads full
Binance futures klines, constructs causal microstructure features, trains a
binary TCN+GRU model under purged walk-forward validation, and produces
artifact-verified evidence for or against promotion.

> [!IMPORTANT]
> This repository does not contain a trading bot, backtester, execution
> engine, position-sizing system, or live deployment service. Model scores are
> research outputs, not trading advice or calibrated probability estimates.

## Current Status

**Model evidence passes the active `v4_evidence` research charter. Historical
model research is frozen, while Phase 2 awaits one preregistered future-OOS
confirmation.**

The first frozen candidate failed and remains immutable history. Its
replacement, `control_recent3_equal_v2`, was selected using historical
rolling-origin evidence only and pinned before collecting a new unseen window.
It has not been refit or tuned against that window.

Latest retained walk-forward evidence snapshot, generated from run
`20260628_155057` and reviewed on **June 28, 2026**:

| Evidence | Result | Interpretation |
|---|---:|---|
| Mean walk-forward Rank IC | `0.0748` | Passes the `0.03` gate |
| Positive-IC folds | `76.3%` | Passes the `75%` gate narrowly |
| PRAUC lift vs prevalence | `1.145` | Robust under hierarchical bootstrap |
| Precision lift vs prevalence | `1.085` | Positive point estimate; uncertainty remains |
| F1 skill vs rate-matched random | `+0.028` | Small but positive classification skill |
| Positive-return folds | `68.4%` | Passes the active consistency gate |
| Top-decile OOS forward return | `0.00286` | Positive walk-forward economic ordering |
| Raw probability calibration | Not deployment-ready | Use outputs as ranking scores |
| Replacement future-OOS rows | `313 / 720` | Confirmation not ready yet |

The legacy monitors remain visible:

- Fold Rank IC standard deviation: `0.0832`
- Raw official Long F1: `0.4377`

These are not hidden or rewritten. Under the active evidence charter they are
monitors rather than standalone promotion gates because the original targets
did not account for dependent time-series sampling noise or no-skill class
baselines.

The current interpretation is deliberately narrow:

- The model has credible **ranking evidence**.
- The model does **not** yet have reliable probability calibration.
- The already-seen holdout produced a negative top-decile return and is not
  used for further tuning.
- Phase 2 requires the pinned replacement to pass its new, no-refit future-OOS
  window. Historical profile search is closed while this confirmation is
  pending.

## Research Boundary

### Implemented

- Binance USDT-M full-kline ingestion with Binance Vision fallback
- 1H primary, completed-and-shifted 4H context, and causal 15m intrahour inputs
- Strict raw schema, gap, duplicate, taker-volume, and trade-count validation
- Microstructure-first causal feature engineering
- Long-only binary triple-barrier labeling
- TCN+GRU binary sequence encoder
- Purged walk-forward cross-validation with train-only scaling
- Audited train-fold-only quantile clipping and feature reliability masking
- Forward-only HMM regime metadata
- Versioned validation charters and append-only experiment memory
- Frozen candidate manifests with content-hashed artifacts
- Prediction-only future-OOS evaluation
- Fail-closed Phase 2 sandbox backtesting and trade forensics
- Locked clean-forward candidates with fixed-fractional portfolio risk controls
- Slim/full evidence bundles and executive diagnostic dashboards

### Explicitly Not Implemented

- Trade execution or exchange order routing
- Live leverage or discretionary position management
- Live services or alerting
- Short-side or three-class labels
- XGBoost or other second-stage meta-learners
- Classical TA signal families such as RSI, MACD, or EMA crossovers

## Pipeline

```mermaid
flowchart LR
    A[Binance full klines<br/>1H / 4H / 15m] --> B[Schema and temporal validation]
    B --> C[Causal microstructure features]
    C --> D[Long-only triple-barrier labels]
    D --> E[Purged walk-forward CV]
    E --> F[TCN + GRU]
    E --> G[Forward-only HMM metadata]
    F --> H[Evidence and failure diagnostics]
    G --> H
    H --> I[Frozen artifact manifest]
    I --> J[Future unseen OOS<br/>no refit]
    J -->|all gates pass| K[Phase 2 eligibility]
```

## Leakage Controls

The repository treats leakage prevention as a first-class product feature.

- A 4H bar is shifted forward by exactly four hours before backward as-of
  merging, so incomplete higher-timeframe bars cannot reach 1H rows.
- Wavelet denoising uses rolling causal windows, never a full-series transform.
- Robust scalers are fitted independently on each training fold.
- Walk-forward folds include purge and embargo gaps.
- Validation, test, holdout, and future-OOS HMM inference is forward-only.
- Future-OOS scoring verifies model, scaler, HMM, feature-order, threshold, and
  training-signature hashes before prediction.
- Future-OOS performs zero fitting operations and fails closed on modified or
  unavailable artifacts.

## Model

The model receives sequences shaped `(batch, 64, n_features)`:

```text
TCN: causal dilated residual blocks [1, 2, 4, 8, 16]
GRU: 2 layers, hidden size 128, unidirectional
Fusion: concat(TCN_last, GRU_last) -> LayerNorm -> MLP
Output: one sigmoid score for the Long class
```

Training uses AdamW, focal loss, a rank-oriented auxiliary objective,
gradient clipping, cosine warm restarts, and early stopping on validation Rank
IC. All research settings are controlled by [`config.yaml`](config.yaml).

## Repository Layout

```text
yeniBot/
|-- config.yaml                 # Hyperparameters and research policy
|-- SKILLS.md                   # Phase 1 operational source of truth
|-- notebooks/                  # Colab workflow 01 through 05
|-- yenibot/
|   |-- data/                   # Download and raw-data validation
|   |-- features/               # Causal feature construction
|   |-- labeling/               # Long-only triple barrier
|   |-- models/                 # TCN and hybrid encoder
|   |-- training/               # Dataset, trainer, walk-forward CV
|   |-- regime/                 # Forward-only HMM
|   |-- experiment/             # Evidence, policy, freezing, orchestration
|   `-- automation/             # Report completeness and readiness review
|-- tests/                      # Unit and integration tests
|-- docs/                       # Architecture and reproducibility
`-- .github/                    # CI and research-aware contribution templates
```

See [`docs/architecture.md`](docs/architecture.md) for module ownership and
failure-localization rules.

Operational references:

- [`docs/current-status.md`](docs/current-status.md): frozen identity and
  current decision
- [`docs/future-oos-runbook.md`](docs/future-oos-runbook.md): exact no-refit
  evaluation procedure
- [`docs/metrics.md`](docs/metrics.md): metric definitions and estimands
- [`docs/experiment-history.md`](docs/experiment-history.md): retained lessons
  and rejected directions
- [`docs/phase2-design.md`](docs/phase2-design.md): frozen Phase 2 research
  backtest contract
- [`docs/phase2-entry-checklist.md`](docs/phase2-entry-checklist.md): blocking
  entry conditions before implementation

## Colab Workflow

All production research notebooks run on Google Colab with source code from
GitHub and data/checkpoints stored on Google Drive.

The current Phase 1/Phase 2 sandbox workflow is isolated on
`codex/phase1-research-v2`. Every notebook fetches, checks out, and verifies
that branch by default, then prints the exact commit. Override
`YENIBOT_REPO_BRANCH` only for a deliberate reviewed run.

Run in strict order:

1. [`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb)
2. [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)
3. [`03_labeling.ipynb`](notebooks/03_labeling.ipynb)
4. [`04_training_walk_forward.ipynb`](notebooks/04_training_walk_forward.ipynb)
5. [`04a_phase1_adaptive_ensemble_research.ipynb`](notebooks/04a_phase1_adaptive_ensemble_research.ipynb), only when its exact cached-policy hypothesis is active
6. [`05_diagnostics_validation.ipynb`](notebooks/05_diagnostics_validation.ipynb)
7. [`06_phase2_sandbox_backtest.ipynb`](notebooks/06_phase2_sandbox_backtest.ipynb)
8. [`07_phase2_clean_forward_confirmation.ipynb`](notebooks/07_phase2_clean_forward_confirmation.ipynb)

After every `git pull`, use **Runtime -> Restart session** before importing the
package again. Colab otherwise retains stale Python modules in memory.

Rerun rules:

| Change | Required notebooks |
|---|---|
| Raw-data range or source | `01 -> 02 -> 03`, then the needed downstream stage |
| Feature-generation formula | `02 -> 03 -> 04 -> 05` |
| Label semantics | `03 -> 04 -> 05` |
| Model, loss, training config, or active training profile | `04 -> 05` |
| Diagnostics/reporting only | `05` |
| Frozen-prediction Phase 2 sandbox/backtest only | `06`; no training and no Phase 1 report rebuild |
| Locked clean-forward confirmation | `07` after `05` writes post-anchor Future-OOS predictions |
| Unevaluated frozen future-OOS data refresh | `01 -> 02 -> 03 -> 05`; do not refit with `04` |
| Historical walk-forward preprocessing/profile experiment | `04 -> 05`; notebooks `02/03` are unchanged |
| Preregistered cached adaptive-ensemble experiment | `04a` only; review its bundle before any other notebook |

When `candidate_profiles` is empty, do not run notebook `04` merely to recreate
the control. Use notebook `05` for reporting changes and wait until a distinct
candidate is explicitly pre-registered.

Notebook 04a is reserved for an explicitly active cached-policy
preregistration. Its current hypothesis has completed and failed, so it must
not be rerun or followed by Notebook 05. The preserved result is
`phase1_latest_policy_research_bundle.zip`.

Before any unevaluated frozen-candidate evaluation, follow
[`docs/future-oos-runbook.md`](docs/future-oos-runbook.md). Its preflight is
authoritative; a calendar date alone does not establish label readiness.

Notebook 05 writes a shareable slim archive under Google Drive:

```text
/content/drive/MyDrive/yeniBot/reports/phase1_latest_experiment_slim_bundle.zip
```

Notebook 06 is the fast, independent Phase 2 entry point. It consumes the
already-pinned prediction artifact, runs no fit operation, executes the
pre-registered optimistic/base/adverse cost scenarios, produces fold/month/
exit/score/holding forensics, and evaluates a bounded dynamic-exit family
without automatically selecting a winner. It writes:

```text
/content/drive/MyDrive/yeniBot/reports/phase2_latest_sandbox_bundle.zip
```

Notebook 07 is the immutable clean-forward evaluator. It uses only post-anchor
Future-OOS predictions from Notebook 05, applies the committed candidate and
risk lock, and refuses to select or promote automatically. See
[`docs/phase2-forward-confirmation.md`](docs/phase2-forward-confirmation.md).

See [`docs/reproducibility.md`](docs/reproducibility.md) before reproducing or
reviewing an experiment.

## Local Development

```bash
git clone https://github.com/umutergul74/yeniBot.git
cd yeniBot

python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pytest -q
```

The package is research-oriented and notebook-driven; local tests use
synthetic fixtures and do not download market data.

## Evidence Artifacts

The diagnostics layer reports multiple estimands instead of compressing model
quality into one number:

- Fold-macro and pooled PRAUC/precision/F1 skill
- Hierarchical fold-cluster and moving-block uncertainty
- Rank IC sign tests, random-effects estimates, and block sensitivity
- Score-band label lift and realized forward returns
- Raw, Platt, and isotonic probability-quality audits
- Fold stability, score reversal, feature drift, and threshold-transfer audits
- Seed coverage and selected-profile completeness
- Frozen candidate hash verification and no-refit future-OOS evidence

Raw sigmoid outputs currently rank observations better than chance but are not
calibrated probabilities. Probability-based confidence language and
probability-sized positions are therefore out of scope.

## Research Governance

Before changing data, features, labels, training, diagnostics, or experiment
policy, read [`SKILLS.md`](SKILLS.md). It records:

- non-negotiable leakage rules
- rejected experiments that must not be repeated blindly
- the active validation charter
- holdout and future-OOS governance
- required diagnostic artifacts
- the Phase 1 to Phase 2 boundary

`config.yaml` is the machine-readable source for model settings, feature
profiles, experiment memory, evidence thresholds, and frozen candidates.
Historical failures are retained so the project advances cumulatively rather
than cycling through previously rejected ideas.

## Contributing

This is a research repository with unusually strict temporal-validity
requirements. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull
request. Every research change must state its causal availability assumptions,
notebook rerun scope, and effect on frozen artifacts.

Security and responsible disclosure guidance is in
[`SECURITY.md`](SECURITY.md).

## License

No open-source license has been declared yet. Until the repository owner adds
one, the source remains available for review but no general reuse license is
granted.

## Disclaimer

This software is for research and educational purposes. It does not provide
investment advice, does not guarantee future performance, and is not ready for
live capital deployment.
