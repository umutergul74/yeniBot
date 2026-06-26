from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        forward_returns: np.ndarray,
        *,
        seq_len: int,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        if len(features) != len(labels) or len(features) != len(forward_returns):
            raise ValueError("features, labels, and forward_returns must have equal length")
        if sample_weights is not None and len(sample_weights) != len(features):
            raise ValueError("sample_weights must have the same length as features")
        if len(features) < seq_len:
            raise ValueError("Not enough rows for requested sequence length")
        self.features = features.astype("float32")
        self.labels = labels.astype("float32")
        self.forward_returns = forward_returns.astype("float32")
        self.sample_weights = (
            np.ones(len(features), dtype="float32")
            if sample_weights is None
            else sample_weights.astype("float32")
        )
        self.seq_len = seq_len
        self.end_positions = np.arange(seq_len - 1, len(features))

    def __len__(self) -> int:
        return len(self.end_positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self.end_positions[idx]
        start = end - self.seq_len + 1
        return (
            torch.from_numpy(self.features[start : end + 1]),
            torch.tensor(self.labels[end], dtype=torch.float32),
            torch.tensor(self.forward_returns[end], dtype=torch.float32),
            torch.tensor(end, dtype=torch.long),
            torch.tensor(self.sample_weights[end], dtype=torch.float32),
        )
