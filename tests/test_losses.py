from __future__ import annotations

import torch

from yenibot.losses import PairwiseLabelMarginLoss, PairwiseReturnOrderLoss


def test_pairwise_label_margin_loss_penalizes_reversed_score_order() -> None:
    loss = PairwiseLabelMarginLoss(margin=0.25)
    good_logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
    bad_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert loss(good_logits, labels) < loss(bad_logits, labels)


def test_pairwise_label_margin_loss_is_zero_for_single_class_batch() -> None:
    loss = PairwiseLabelMarginLoss(margin=0.25)

    assert float(loss(torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]))) == 0.0


def test_pairwise_return_order_loss_penalizes_return_order_reversal() -> None:
    loss = PairwiseReturnOrderLoss(margin=0.05, min_return_diff=0.0001, return_scale=0.005)
    returns = torch.tensor([0.02, 0.01, -0.01, -0.02])
    good_logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
    bad_logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])

    assert loss(good_logits, returns) < loss(bad_logits, returns)


def test_pairwise_return_order_loss_ignores_tiny_return_differences() -> None:
    loss = PairwiseReturnOrderLoss(margin=0.05, min_return_diff=1.0, return_scale=0.005)

    assert float(loss(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.001, 0.002, 0.003]))) == 0.0
