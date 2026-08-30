"""Score-only replay cache for retrospective, independent-fold attribution.

This is NOT a portfolio/live engine. Exit paths and fixed-duration costs are
resolved by the reference engine once; only score selection and position overlap
are replayed. There is deliberately no risk-policy or funding-events argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import CostScenario, Phase2StrategyContract
from yenibot.phase2.costs import net_long_return
from yenibot.phase2.engine import (
    _entry_filter_rejection_reason,
    _entry_index_after_decision,
    _prepare_bars,
    _resolve_exit,
)
from yenibot.phase2.market_contract import normalize_execution_inputs


@dataclass(frozen=True)
class FoldExecutionCache:
    contract: Phase2StrategyContract
    score_count: int
    folds: tuple[np.ndarray, ...]
    entry_indices: np.ndarray
    exit_indices: np.ndarray
    net_returns: np.ndarray
    censored: np.ndarray
    data_gaps: np.ndarray

    def evaluate(self, scores: np.ndarray) -> dict[str, Any]:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.score_count,) or not np.isfinite(scores).all():
            raise ValueError("Cache requires one finite score per original signal")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("Cache scores must be in [0, 1]")
        selected = scores >= self.contract.threshold
        eligible = selected & (
            scores - self.contract.threshold >= self.contract.min_score_margin
        )
        returns: list[float] = []
        selected_count = 0
        gap_count = 0
        for indices in self.folds:
            minimum_entry = 0
            for idx in indices:
                if not selected[idx]:
                    continue
                selected_count += 1
                entry_idx = self.entry_indices[idx]
                if not eligible[idx] or entry_idx < minimum_entry:
                    continue
                if not self.censored[idx]:
                    returns.append(float(self.net_returns[idx]))
                if self.data_gaps[idx]:
                    gap_count += 1
                    break
                minimum_entry = self.exit_indices[idx] + 1
        values = np.asarray(returns, dtype=float)
        if values.size:
            equity = np.cumprod(1.0 + values)
            drawdown = equity / np.maximum.accumulate(np.maximum(equity, 1.0)) - 1
            gain = values[values > 0].sum()
            loss = -values[values < 0].sum()
            compounded = float(equity[-1] - 1)
            max_drawdown = float(drawdown.min())
            profit_factor = float(gain / loss) if loss > 0 else None
            hit_rate = float((values > 0).mean())
        else:
            compounded = max_drawdown = profit_factor = hit_rate = 0.0
        return {
            "compounded_return": compounded,
            "completed_trade_compounded_return": compounded,
            "max_drawdown": max_drawdown,
            "trade_count": int(values.size),
            "selected_signal_count": selected_count,
            "profit_factor": profit_factor,
            "hit_rate": hit_rate,
            "data_contract_complete": gap_count == 0,
        }


def build_fold_execution_cache(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    contract: Phase2StrategyContract,
    scenario: CostScenario,
) -> FoldExecutionCache:
    """Inputs must be sorted/validated by economic_attribution._validate_inputs."""
    contract.validate()
    scenario.validate()
    if not signals.index.equals(pd.RangeIndex(len(signals))):
        raise ValueError("Cache signals require a positional RangeIndex")
    count = len(signals)
    entries = np.full(count, -1, dtype=int)
    exits = np.full(count, -1, dtype=int)
    returns = np.zeros(count)
    censored = np.zeros(count, dtype=bool)
    gaps = np.zeros(count, dtype=bool)
    folds = []
    for _, group in signals.groupby("fold", sort=True, dropna=False):
        indices = group.index.to_numpy(dtype=int)
        folds.append(indices)
        start = group[contract.decision_time_column].min()
        end = group[contract.decision_time_column].max()
        frame = bars.loc[bars[contract.bar_time_column].between(start, end)].copy()
        frame = _prepare_bars(frame, contract)
        frame, decisions, _ = normalize_execution_inputs(frame, group, contract)
        for idx, decision_time in zip(
            indices, decisions[contract.decision_time_column]
        ):
            entry_idx = _entry_index_after_decision(frame.bar_open_time, decision_time)
            if entry_idx is None or entry_idx == 0:
                continue
            entry_time = frame.loc[entry_idx, "bar_open_time"]
            delay = (entry_time - decision_time).total_seconds() / 3600
            if delay > contract.max_bar_gap_hours or (
                frame.loc[entry_idx - 1, contract.bar_time_column] != entry_time
            ):
                continue
            atr = float(frame.loc[entry_idx - 1, contract.atr_column])
            price = float(frame.loc[entry_idx, "open"])
            if (
                atr <= 0
                or _entry_filter_rejection_reason(
                    contract, score=1.0, entry_price=price, atr=atr
                )
                is not None
            ):
                continue
            resolution = _resolve_exit(
                frame, contract, entry_idx=entry_idx, entry_price=price, atr=atr
            )
            exit_time = frame.loc[
                resolution.exit_idx,
                "bar_open_time"
                if resolution.exit_at_open
                else contract.bar_time_column,
            ]
            is_censored = resolution.exit_reason.endswith("_censored")
            result = net_long_return(
                scenario,
                entry_price=price,
                exit_price=resolution.exit_price,
                holding_hours=(exit_time - entry_time).total_seconds() / 3600,
                charge_exit=not is_censored,
            )
            entries[idx] = entry_idx
            exits[idx] = resolution.exit_idx
            returns[idx] = result["net_return"]
            censored[idx] = is_censored
            gaps[idx] = resolution.exit_reason == "data_gap_censored"
    return FoldExecutionCache(
        contract, count, tuple(folds), entries, exits, returns, censored, gaps
    )


def assert_cache_matches_reference(
    cached: dict[str, Any], reference: dict[str, Any]
) -> None:
    for key, value in cached.items():
        expected = reference[key]
        if value is None or expected is None:
            matched = value is expected
        else:
            matched = bool(np.isclose(value, expected, rtol=1e-10, atol=1e-12))
        if not matched:
            raise RuntimeError(f"Attribution execution cache mismatch: {key}")
