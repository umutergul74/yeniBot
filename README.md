# yeniBot

Phase 1 ML foundation for BTC/USDT perpetual futures direction modeling.

The project intentionally stops at validated model research. It does not include backtesting, trade execution, live deployment, XGBoost meta-learning, or 3-class short/hold/long labels.

## What This Builds

- Binance USDT-M full-kline downloader for `BTCUSDT` 1H and 4H data.
- Binance Vision fallback for Colab or other hosts that receive HTTP 451 from the REST API.
- Config-controlled dropping of rare zero-volume/no-trade archive bars before feature generation.
- Bias-safe microstructure feature engineering.
- Correct 4H-to-1H alignment by shifting 4H bars forward before merge.
- Long-only binary triple-barrier labels.
- Binary TCN+GRU sequence model with focal and rank-correlation losses.
- Purged walk-forward CV with train-only scaling.
- Forward-only HMM regime diagnostics.
- Colab notebooks `01` through `05`.

## Colab Workflow

Run notebooks in strict order:

1. `01_data_preparation.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_labeling.ipynb`
4. `04_training_walk_forward.ipynb`
5. `05_diagnostics_validation.ipynb`

After any `git pull` in Colab, use `Runtime -> Restart session` before re-running cells. Python keeps imported modules in memory and will otherwise use stale code.

## Local Checks

```bash
pip install -r requirements.txt
pytest
```

## Phase 1 Gate

Proceed only when diagnostics show:

- Rank IC mean above `0.03`
- Long F1 above `0.45`
- Rank IC std below `0.03`
- Positive Rank IC in more than `75%` of folds
- Calibration separation between actual long and non-long outcomes
- No leakage alerts

Notebook `05` writes a shareable diagnostics archive under `Drive/yeniBot/reports/`.
Send that `phase1_diagnostics_*.zip` when a run needs review.
