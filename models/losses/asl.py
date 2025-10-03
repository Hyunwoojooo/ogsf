"""Action sensitivity loss components for event-aware training."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["asl_loss"]

Reduction = Literal["none", "mean", "sum"]


def asl_loss(
    logits: Tensor,
    targets: Tensor,
    sensitivity: Tensor,
    *,
    base_weight: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute the action-sensitivity loss with per-frame weights."""

    targets = targets.to(dtype=logits.dtype, device=logits.device)
    sensitivity = sensitivity.to(dtype=logits.dtype, device=logits.device)

    weight = base_weight + sensitivity
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weight

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")
