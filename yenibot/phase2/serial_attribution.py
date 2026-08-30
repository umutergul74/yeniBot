"""Dependence-aware sensitivity control, not proof of an independent signal."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from yenibot.phase2.contracts import Phase2StrategyContract
from yenibot.phase2.execution_cache import FoldExecutionCache


def circular_shift_scores(scores, groups, rng):
    """Preserve each group's cyclic order and marginal distribution.

    There is one wrap seam per group. This assumes local shift-invariance, not
    independence, and does not guarantee exactly identical executed turnover.
    Zero shift is included in the randomization space; seeds are never searched.
    """
    shifted = np.asarray(scores, dtype=float).copy()
    for indices in groups:
        shifted[indices] = np.roll(scores[indices], rng.integers(len(indices)))
    return shifted


def cache_with_score_threshold(
    cache: FoldExecutionCache, contract: Phase2StrategyContract
) -> FoldExecutionCache:
    contract.validate()
    normalized = replace(
        contract,
        threshold=cache.contract.threshold,
        strategy_id=cache.contract.strategy_id,
    )
    if normalized != cache.contract:
        raise ValueError("Shared cache may change only score threshold/strategy id")
    return replace(cache, contract=contract)


def run_serial_null(cache, scores, groups, *, permutations, seed):
    if permutations < 20:
        raise ValueError("At least 20 circular shifts required")
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(permutations):
        shifted = circular_shift_scores(scores, groups, rng)
        rows.append({"trial": trial, "seed": seed, **cache.evaluate(shifted)})
    return pd.DataFrame(rows)
