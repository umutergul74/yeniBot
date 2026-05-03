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
