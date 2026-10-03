"""Single-branch controls using the same components as the existing hybrid."""
from __future__ import annotations

import torch
from torch import nn

from yenibot.models import HybridEncoder
from yenibot.models.tcn import CausalTCN


class SingleEncoder(nn.Module):
    def __init__(self, n_features: int, architecture: str, settings: dict) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "gru":
            width = settings["gru_hidden"]
            layers = settings["gru_layers"]
            self.encoder = nn.GRU(n_features, width, num_layers=layers,
                                  dropout=settings["dropout"] if layers > 1 else 0,
                                  batch_first=True, bidirectional=False)
        elif architecture == "tcn":
            width = settings["tcn_channels"]
            self.encoder = CausalTCN(n_features, width,
                                    kernel_size=settings["tcn_kernel_size"],
                                    dilations=settings["tcn_dilations"], dropout=settings["dropout"])
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, settings["fusion_hidden"]),
                                  nn.GELU(), nn.Dropout(settings["dropout"]),
                                  nn.Linear(settings["fusion_hidden"], 1))

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> torch.Tensor:
        encoded = self.encoder(x)
        if self.architecture == "gru":
            encoded = encoded[0]
        logits = self.head(encoded[:, -1]).squeeze(-1)
        return logits if return_logits else torch.sigmoid(logits)


def build_advisor_model(n_features: int, architecture: str, settings: dict) -> nn.Module:
    if architecture == "tcn_gru":
        return HybridEncoder(n_features, **settings)
    return SingleEncoder(n_features, architecture, settings)
