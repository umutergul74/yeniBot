from __future__ import annotations

import torch

from yenibot.losses import (
    FocalLossWithLogits,
    PairwiseLabelMarginLoss,
    PairwiseReturnOrderLoss,
    RankICLoss,
    ScaledHuberReturnLoss,
)


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


def test_focal_and_rank_ic_losses_accept_sample_weights() -> None:
    logits = torch.tensor([0.2, -0.3, 1.0, -1.2])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    returns = torch.tensor([0.01, -0.002, 0.004, -0.003])
    weights = torch.tensor([2.0, 0.5, 1.5, 0.25])

    focal_loss = FocalLossWithLogits()(logits, labels, weights)
    rank_loss = RankICLoss()(torch.sigmoid(logits), returns, weights)

    assert torch.isfinite(focal_loss)
    assert torch.isfinite(rank_loss)


def test_scaled_huber_return_loss_scales_clips_and_weights_targets() -> None:
    loss = ScaledHuberReturnLoss(
        target_scale=0.01,
        target_clip=2.0,
        beta=1.0,
    )
    returns = torch.tensor([0.01, -0.01, 0.20])
    exact_scaled_predictions = torch.tensor([1.0, -1.0, 2.0])
    weights = torch.tensor([1.0, 2.0, 0.5])

    exact = loss(exact_scaled_predictions, returns, weights)
    shifted = loss(exact_scaled_predictions + 0.5, returns, weights)

    assert float(exact) == 0.0
    assert torch.isfinite(shifted)
    assert shifted > exact
