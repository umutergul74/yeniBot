"""Causal trajectory weight averaging for a single fitted fold model."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class WeightAveragingSettings:
    enabled: bool = False
    strategy: str = "swa"
    start_epoch: int = 10
    update_interval_epochs: int = 1
    min_snapshots: int = 3


def weight_averaging_settings(config: Any) -> WeightAveragingSettings:
    training = (
        config.get("training", {})
        if isinstance(config, dict)
        else getattr(config, "training", {})
    ) or {}
    raw = (
        training.get("weight_averaging", {})
        if isinstance(training, dict)
        else getattr(training, "weight_averaging", {})
    ) or {}

    def read(name: str, default: Any) -> Any:
        return raw.get(name, default) if isinstance(raw, dict) else getattr(raw, name, default)

    settings = WeightAveragingSettings(
        enabled=bool(read("enabled", False)),
        strategy=str(read("strategy", "swa")),
        start_epoch=int(read("start_epoch", 10)),
        update_interval_epochs=int(read("update_interval_epochs", 1)),
        min_snapshots=int(read("min_snapshots", 3)),
    )
    if settings.strategy != "swa":
        raise ValueError("training.weight_averaging.strategy must be 'swa'")
    if settings.start_epoch < 1:
        raise ValueError("training.weight_averaging.start_epoch must be at least 1")
    if settings.update_interval_epochs < 1:
        raise ValueError(
            "training.weight_averaging.update_interval_epochs must be at least 1"
        )
    if settings.min_snapshots < 2:
        raise ValueError("training.weight_averaging.min_snapshots must be at least 2")
    return settings


class TrajectoryWeightAverager:
    """Equal-weight model-state average collected from training-only epochs.

    Epoch numbers are one-based at the public boundary so the configuration and
    audit are readable. Floating tensors are averaged online; non-floating
    buffers are copied from the latest snapshot. The retained model remains a
    single checkpoint, not an inference-time ensemble.
    """

    def __init__(self, settings: WeightAveragingSettings) -> None:
        self.settings = settings
        self.snapshot_count = 0
        self.first_snapshot_epoch: int | None = None
        self.last_snapshot_epoch: int | None = None
        self._state: OrderedDict[str, torch.Tensor] = OrderedDict()

    @property
    def ready(self) -> bool:
        return self.snapshot_count >= self.settings.min_snapshots

    def should_update(self, epoch: int) -> bool:
        if not self.settings.enabled or epoch < self.settings.start_epoch:
            return False
        return (
            epoch - self.settings.start_epoch
        ) % self.settings.update_interval_epochs == 0

    @torch.no_grad()
    def update(self, model: nn.Module, *, epoch: int) -> None:
        if not self.should_update(epoch):
            raise ValueError(f"Epoch {epoch} is not eligible for a weight snapshot")
        state = model.state_dict()
        if self.snapshot_count == 0:
            self._state = OrderedDict(
                (name, tensor.detach().clone()) for name, tensor in state.items()
            )
            self.first_snapshot_epoch = int(epoch)
        else:
            next_count = self.snapshot_count + 1
            for name, tensor in state.items():
                source = tensor.detach()
                target = self._state[name]
                if target.is_floating_point() or target.is_complex():
                    target.add_((source - target) / float(next_count))
                else:
                    target.copy_(source)
        self.snapshot_count += 1
        self.last_snapshot_epoch = int(epoch)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError(
                "Weight average is not ready: "
                f"{self.snapshot_count}/{self.settings.min_snapshots} snapshots"
            )
        return OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in self._state.items()
        )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict(), strict=True)

    @torch.no_grad()
    def mean_absolute_parameter_delta(self, model: nn.Module) -> float:
        if not self.ready:
            return float("nan")
        live = dict(model.named_parameters())
        deltas = []
        for name, averaged in self._state.items():
            parameter = live.get(name)
            if parameter is None or not averaged.is_floating_point():
                continue
            deltas.append(
                (parameter.detach() - averaged).abs().mean().to(dtype=torch.float64)
            )
        if not deltas:
            return 0.0
        return float(torch.stack(deltas).mean().cpu())


def validate_weight_averaging_horizon(
    settings: WeightAveragingSettings,
    *,
    epochs: int,
) -> None:
    if not settings.enabled:
        return
    last_required_epoch = settings.start_epoch + (
        settings.min_snapshots - 1
    ) * settings.update_interval_epochs
    if epochs < last_required_epoch:
        raise ValueError(
            "training.epochs cannot produce the preregistered weight average: "
            f"need epoch {last_required_epoch}, configured {epochs}"
        )


def clone_model_with_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> nn.Module:
    averaged_model = copy.deepcopy(model)
    averaged_model.load_state_dict(state, strict=True)
    return averaged_model
