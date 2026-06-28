from __future__ import annotations

import pytest
import torch
from torch import nn

from yenibot.training.multitask import apply_primary_preserving_projection


class _ToyTwoHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor([1.0]))
        self.output = nn.Linear(1, 1, bias=False)
        self.auxiliary_return_output = nn.Linear(1, 1, bias=False)


def test_primary_preserving_projection_removes_only_conflicting_auxiliary_gradient() -> None:
    model = _ToyTwoHeadModel()
    primary_loss = model.shared.square().sum()
    auxiliary_loss = (model.shared - 2.0).square().sum()

    audit = apply_primary_preserving_projection(
        model,
        primary_loss=primary_loss,
        weighted_auxiliary_loss=auxiliary_loss,
    )

    assert audit["conflict"] is True
    assert audit["cosine_before"] == pytest.approx(-1.0)
    assert audit["cosine_after"] == pytest.approx(0.0, abs=1e-7)
    torch.testing.assert_close(model.shared.grad, torch.tensor([2.0]))


def test_primary_preserving_projection_keeps_aligned_auxiliary_gradient() -> None:
    model = _ToyTwoHeadModel()
    primary_loss = model.shared.square().sum()
    auxiliary_loss = (2.0 * model.shared).square().sum()

    audit = apply_primary_preserving_projection(
        model,
        primary_loss=primary_loss,
        weighted_auxiliary_loss=auxiliary_loss,
    )

    assert audit["conflict"] is False
    assert audit["cosine_before"] == pytest.approx(1.0)
    torch.testing.assert_close(model.shared.grad, torch.tensor([10.0]))
