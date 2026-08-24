from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from yenibot.training.weight_averaging import (
    TrajectoryWeightAverager,
    WeightAveragingSettings,
    clone_model_with_state,
    validate_weight_averaging_horizon,
    weight_averaging_settings,
)


def test_trajectory_weight_averager_computes_equal_weight_state_average() -> None:
    model = nn.Linear(2, 1, bias=True)
    settings = WeightAveragingSettings(
        enabled=True,
        start_epoch=1,
        update_interval_epochs=1,
        min_snapshots=2,
    )
    averager = TrajectoryWeightAverager(settings)

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(2.0)
    averager.update(model, epoch=1)
    assert averager.ready is False

    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(6.0)
    averager.update(model, epoch=2)

    averaged = clone_model_with_state(model, averager.state_dict())
    assert averager.ready is True
    torch.testing.assert_close(averaged.weight, torch.full_like(averaged.weight, 2.0))
    torch.testing.assert_close(averaged.bias, torch.full_like(averaged.bias, 4.0))
    assert averager.mean_absolute_parameter_delta(model) == pytest.approx(1.5)


def test_weight_averaging_settings_reject_inert_or_unreachable_plan() -> None:
    with pytest.raises(ValueError, match="min_snapshots must be at least 2"):
        weight_averaging_settings(
            {"training": {"weight_averaging": {"enabled": True, "min_snapshots": 1}}}
        )

    settings = WeightAveragingSettings(
        enabled=True,
        start_epoch=10,
        update_interval_epochs=2,
        min_snapshots=3,
    )
    with pytest.raises(ValueError, match="need epoch 14"):
        validate_weight_averaging_horizon(settings, epochs=13)


def test_weight_average_state_is_independent_of_later_live_updates() -> None:
    model = nn.Linear(1, 1, bias=False)
    settings = WeightAveragingSettings(
        enabled=True,
        start_epoch=1,
        min_snapshots=2,
    )
    averager = TrajectoryWeightAverager(settings)
    for epoch, value in ((1, 1.0), (2, 3.0)):
        with torch.no_grad():
            model.weight.fill_(value)
        averager.update(model, epoch=epoch)
    retained = copy.deepcopy(averager.state_dict())

    with torch.no_grad():
        model.weight.fill_(99.0)

    torch.testing.assert_close(retained["weight"], torch.tensor([[2.0]]))
