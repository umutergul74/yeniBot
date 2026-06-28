from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def _named_parameter_groups(
    model: nn.Module,
) -> tuple[
    list[tuple[str, nn.Parameter]],
    list[tuple[str, nn.Parameter]],
    list[tuple[str, nn.Parameter]],
]:
    shared: list[tuple[str, nn.Parameter]] = []
    primary_head: list[tuple[str, nn.Parameter]] = []
    auxiliary_head: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("output."):
            primary_head.append((name, parameter))
        elif name.startswith("auxiliary_return_output."):
            auxiliary_head.append((name, parameter))
        else:
            shared.append((name, parameter))
    if not shared or not primary_head or not auxiliary_head:
        raise ValueError(
            "Gradient projection requires shared, primary-head, and "
            "auxiliary-head parameters"
        )
    return shared, primary_head, auxiliary_head


def _gradient_map(
    loss: torch.Tensor,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    pairs = list(named_parameters)
    gradients = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in pairs],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        name: gradient.detach()
        for (name, _parameter), gradient in zip(pairs, gradients)
        if gradient is not None
    }


def apply_primary_preserving_projection(
    model: nn.Module,
    *,
    primary_loss: torch.Tensor,
    weighted_auxiliary_loss: torch.Tensor,
) -> dict[str, Any]:
    """Assign gradients while removing auxiliary conflict with the primary task.

    The primary gradient is never changed. When the shared auxiliary gradient
    has a negative global dot product with it, only the auxiliary component is
    projected onto the normal plane of the primary gradient.
    """

    shared, primary_head, auxiliary_head = _named_parameter_groups(model)
    primary_pairs = [*shared, *primary_head]
    auxiliary_pairs = [*shared, *auxiliary_head]
    primary_gradients = _gradient_map(
        primary_loss,
        primary_pairs,
        retain_graph=True,
    )
    auxiliary_gradients = _gradient_map(
        weighted_auxiliary_loss,
        auxiliary_pairs,
        retain_graph=False,
    )

    shared_names = [name for name, _parameter in shared]
    common_names = [
        name
        for name in shared_names
        if name in primary_gradients and name in auxiliary_gradients
    ]
    if not common_names:
        raise ValueError("Primary and auxiliary tasks share no trainable gradients")

    dot = sum(
        torch.sum(primary_gradients[name] * auxiliary_gradients[name])
        for name in common_names
    )
    primary_norm_sq = sum(
        torch.sum(primary_gradients[name].square())
        for name in common_names
    )
    auxiliary_norm_sq = sum(
        torch.sum(auxiliary_gradients[name].square())
        for name in common_names
    )
    denominator = torch.sqrt(primary_norm_sq * auxiliary_norm_sq).clamp_min(1e-12)
    cosine_before = dot / denominator
    conflict = bool(dot.detach().cpu() < 0)
    projection_scale = (
        dot / primary_norm_sq.clamp_min(1e-12)
        if conflict
        else dot.new_tensor(0.0)
    )

    projected_auxiliary: dict[str, torch.Tensor] = {}
    for name in common_names:
        gradient = auxiliary_gradients[name]
        if conflict:
            gradient = gradient - projection_scale * primary_gradients[name]
        projected_auxiliary[name] = gradient

    parameter_lookup = dict(model.named_parameters())
    for parameter in parameter_lookup.values():
        parameter.grad = None
    for name, gradient in primary_gradients.items():
        parameter_lookup[name].grad = gradient.clone()
    for name, gradient in auxiliary_gradients.items():
        if name in projected_auxiliary:
            gradient = projected_auxiliary[name]
        parameter = parameter_lookup[name]
        parameter.grad = (
            gradient.clone()
            if parameter.grad is None
            else parameter.grad + gradient
        )

    projected_dot = sum(
        torch.sum(primary_gradients[name] * projected_auxiliary[name])
        for name in common_names
    )
    projected_auxiliary_norm_sq = sum(
        torch.sum(projected_auxiliary[name].square())
        for name in common_names
    )
    projected_denominator = torch.sqrt(
        primary_norm_sq * projected_auxiliary_norm_sq
    ).clamp_min(1e-12)
    cosine_after = projected_dot / projected_denominator
    primary_norm = torch.sqrt(primary_norm_sq)
    auxiliary_norm = torch.sqrt(auxiliary_norm_sq)

    return {
        "conflict": conflict,
        "cosine_before": float(cosine_before.detach().cpu()),
        "cosine_after": float(cosine_after.detach().cpu()),
        "primary_shared_grad_norm": float(primary_norm.detach().cpu()),
        "weighted_auxiliary_shared_grad_norm": float(
            auxiliary_norm.detach().cpu()
        ),
        "auxiliary_to_primary_grad_norm_ratio": float(
            (auxiliary_norm / primary_norm.clamp_min(1e-12)).detach().cpu()
        ),
    }
