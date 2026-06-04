from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLossWithLogits(nn.Module):
    def __init__(self, *, gamma: float = 2.0, alpha: float = 0.6) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


class RankICLoss(nn.Module):
    """Differentiable Pearson proxy for rank-IC alignment."""

    def forward(self, probs: torch.Tensor, forward_returns: torch.Tensor) -> torch.Tensor:
        probs = probs.float()
        returns = forward_returns.float()
        valid = torch.isfinite(probs) & torch.isfinite(returns)
        if valid.sum() < 3:
            return probs.new_tensor(0.0)
        x = probs[valid] - probs[valid].mean()
        y = returns[valid] - returns[valid].mean()
        denom = torch.sqrt((x.square().sum() + 1e-8) * (y.square().sum() + 1e-8))
        corr = (x * y).sum() / denom
        return -corr


class PairwiseLabelMarginLoss(nn.Module):
    """Push labeled-long logits above not-long logits inside each batch.

    Focal loss optimizes pointwise classification and RankICLoss aligns scores
    with forward returns. This auxiliary term directly targets the failure mode
    seen in bad folds: positive and negative score distributions compressing or
    reversing. It is intentionally batch-local and disabled unless configured.
    """

    def __init__(self, *, margin: float = 0.25) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float().flatten()
        targets = targets.float().flatten()
        valid = torch.isfinite(logits) & torch.isfinite(targets)
        if valid.sum() < 3:
            return logits.new_tensor(0.0)
        logits = logits[valid]
        targets = targets[valid]
        positive = logits[targets > 0.5]
        negative = logits[targets <= 0.5]
        if positive.numel() == 0 or negative.numel() == 0:
            return logits.new_tensor(0.0)
        pairwise_margin = positive[:, None] - negative[None, :]
        return F.softplus(self.margin - pairwise_margin).mean()


class PairwiseReturnOrderLoss(nn.Module):
    """Encourage logit ordering to match forward-return ordering.

    This is a local pairwise ranking term. Unlike label-margin loss, it does
    not force every long-label row above every not-long row; it only asks the
    model to preserve the ordering implied by realized forward returns inside
    the batch. That makes it a narrower candidate for Rank IC and bad-fold
    score-separation stability.
    """

    def __init__(self, *, margin: float = 0.05, min_return_diff: float = 0.0005, return_scale: float = 0.005) -> None:
        super().__init__()
        self.margin = float(margin)
        self.min_return_diff = float(min_return_diff)
        self.return_scale = max(float(return_scale), 1e-8)

    def forward(self, logits: torch.Tensor, forward_returns: torch.Tensor) -> torch.Tensor:
        logits = logits.float().flatten()
        returns = forward_returns.float().flatten()
        valid = torch.isfinite(logits) & torch.isfinite(returns)
        if valid.sum() < 3:
            return logits.new_tensor(0.0)
        logits = logits[valid]
        returns = returns[valid]

        return_delta = returns[:, None] - returns[None, :]
        upper = torch.triu(torch.ones_like(return_delta, dtype=torch.bool), diagonal=1)
        meaningful = upper & (return_delta.abs() >= self.min_return_diff)
        if meaningful.sum() == 0:
            return logits.new_tensor(0.0)

        logit_delta = logits[:, None] - logits[None, :]
        direction = torch.sign(return_delta[meaningful])
        ordered_logit_delta = direction * logit_delta[meaningful]
        weights = torch.tanh(return_delta[meaningful].abs() / self.return_scale).detach()
        return (weights * F.softplus(self.margin - ordered_logit_delta)).sum() / (weights.sum() + 1e-8)
