# yeniBot — Phase 1 Operational Manual

This file is the project source of truth for the v3 rebuild. It intentionally replaces older Scalp2 guidance around 3-class labels, XGBoost, execution logic, and live trading.

## Mission

Build a professional, bias-free ML pipeline that trains a binary TCN+GRU sequence model to identify BTC/USDT perpetual futures long opportunities from market microstructure features. Phase 1 ends at model validation. Do not build backtesting, execution, trade management, or live bot code until the Phase 1 criteria pass.

## Non-Negotiable Constraints

- Use Binance USDT-M full kline REST data, not CCXT OHLCV-only data.
- Require 12 kline columns, especially `num_trades`, `taker_buy_base_vol`, and `taker_buy_quote_vol`.
- Use market microstructure features as the signal source. Do not add RSI, MACD, EMA crossovers, Bollinger Bands, stochastic oscillators, or XGBoost meta-learners.
- Apply wavelet denoising causally with rolling windows only. Never transform the full time series.
- Align 4H features by shifting 4H timestamps forward exactly 4 hours before merging into 1H rows.
- Fit scalers on train folds only.
- Use long-only binary labels. Do not create 3-class labels.
- Use HMM regimes only as diagnostics/filter metadata, not as a second-stage predictor.
- Use forward-only HMM probabilities outside the training fold.
- Keep all hyperparameters in `config.yaml`.
- Do not commit data, checkpoints, `.env`, runtime state JSON, or model weights.

## Phase 1 Success Criteria

- Validation fold Rank IC, Spearman of `P(Long)` versus forward return: consistently above `0.03`.
- Long class F1 walk-forward average above `0.45`.
- Mean `P(Long)` for actual long labels separates from non-long labels.
- Fold-to-fold Rank IC standard deviation below `0.03`.
- IC positive in more than `75%` of folds.
- Zero look-ahead bias verified by temporal holdout and leakage tests.

If IC is near `0.01`, the features are inadequate. Do not tune model hyperparameters as the first response. Improve the feature set. If IC exceeds `0.10`, assume leakage until proven otherwise.

## Required Pipeline

1. Download and validate full Binance 1H and 4H klines from `2022-01-01` to present.
2. Build causal microstructure features on 1H and 4H bars.
3. Shift 4H feature timestamps forward by one complete 4H period before joining.
4. Create long-only binary triple-barrier labels.
5. Train the binary TCN+GRU model with purged walk-forward CV.
6. Save per-fold scalers, models, HMMs, and prediction parquet files.
7. Run diagnostics and print a Phase 1 PASS/FAIL report.

## Git Discipline

Use focused commits with one of these prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation
- `data:` data pipeline
- `model:` model or training change

Always check `git status` before committing.
