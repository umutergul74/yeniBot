from __future__ import annotations

import numpy as np
import pandas as pd
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
        timestamps=None,
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
        if timestamps is not None:
            times = pd.Series(pd.to_datetime(timestamps, utc=True)).reset_index(
                drop=True
            )
            if (
                len(times) != len(features)
                or times.isna().any()
                or not times.is_monotonic_increasing
                or times.duplicated().any()
            ):
                raise ValueError(
                    "Sequence timestamps must be ordered, unique and aligned"
                )
            gaps = times.diff().ne(pd.Timedelta(hours=1)).astype(int)
            gaps.iloc[0] = 0
            cumulative = gaps.cumsum().to_numpy()
            ends = self.end_positions
            self.end_positions = ends[
                cumulative[ends] == cumulative[ends - seq_len + 1]
            ]
            if not len(self.end_positions):
                raise ValueError("No contiguous hourly sequence is available")

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
