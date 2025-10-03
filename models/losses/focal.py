"""Focal loss implementations tailored for NLQ classification heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["focal_loss"]

Reduction = Literal["none", "mean", "sum"]


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute the binary focal loss in a numerically stable manner.

    Parameters
    ----------
    logits:
        Predicted logits of shape ``[..., T]``.
    targets:
        Binary labels broadcastable with *logits*.
    alpha:
        Class-balancing weight. ``None`` disables the weighting term.
    gamma:
        Focusing parameter emphasising hard examples.
    reduction:
        One of ``"none"``, ``"mean"`` or ``"sum"``.
    """

    targets = targets.to(dtype=logits.dtype, device=logits.device)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = torch.where(targets >= 0.5, prob, 1 - prob)
    modulating = (1 - p_t).clamp(min=0) ** gamma
    loss = modulating * ce

    if alpha is not None:
        alpha_pos = loss.new_tensor(alpha)
        alpha_neg = loss.new_tensor(1 - alpha)
        alpha_factor = torch.where(targets >= 0.5, alpha_pos, alpha_neg)
        loss = loss * alpha_factor

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")
