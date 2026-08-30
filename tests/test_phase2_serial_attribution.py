from dataclasses import replace

import numpy as np
import pytest

from yenibot.phase2.contracts import DEFAULT_PHASE2_CONTRACT
from yenibot.phase2.execution_cache import FoldExecutionCache
from yenibot.phase2.serial_attribution import (
    cache_with_score_threshold,
    circular_shift_scores,
)


def test_shift_preserves_within_group_cyclic_order_and_density():
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.6])
    groups = [np.arange(4), np.arange(4, 8)]
    shifted = circular_shift_scores(scores, groups, np.random.default_rng(42))
    for group in groups:
        before, after = scores[group], shifted[group]
        np.testing.assert_array_equal(np.sort(before), np.sort(after))
        np.testing.assert_allclose(
            np.sort(np.diff(np.r_[before, before[0]])),
            np.sort(np.diff(np.r_[after, after[0]])),
        )
        for threshold in (0.5, 0.7, 0.8, 0.9):
            assert (before >= threshold).sum() == (after >= threshold).sum()


def test_shared_cache_rejects_exit_and_cost_changes():
    a = np.array([], dtype=int)
    cache = FoldExecutionCache(DEFAULT_PHASE2_CONTRACT, 0, (), a, a, a, a, a)
    changed = replace(DEFAULT_PHASE2_CONTRACT, threshold=0.8, strategy_id="density_q80")
    assert cache_with_score_threshold(cache, changed).contract == changed
    with pytest.raises(ValueError, match="only score threshold"):
        cache_with_score_threshold(cache, replace(changed, stop_loss_atr=3))
    different_costs = tuple(
        replace(cost, entry_fee_bps=7) for cost in changed.cost_scenarios
    )
    with pytest.raises(ValueError, match="only score threshold"):
        cache_with_score_threshold(cache, replace(changed, cost_scenarios=different_costs))
