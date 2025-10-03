"""Segment IoU-based losses including IoU, GIoU, and DIoU."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = ["temporal_iou", "iou_loss", "giou_loss", "diou_loss"]

Reduction = Literal["none", "mean", "sum"]


def _prepare_segments(segments: Tensor) -> tuple[Tensor, Tensor]:
    start, end = segments.split(1, dim=-1)
    start, end = torch.minimum(start, end), torch.maximum(start, end)
    return start.squeeze(-1), end.squeeze(-1)


def temporal_iou(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Return the IoU between 1D temporal segments."""

    pred_start, pred_end = _prepare_segments(pred)
    target_start, target_end = _prepare_segments(target)

    inter_start = torch.maximum(pred_start, target_start)
    inter_end = torch.minimum(pred_end, target_end)
    intersection = (inter_end - inter_start).clamp(min=0.0)

    pred_length = (pred_end - pred_start).clamp(min=0.0)
    target_length = (target_end - target_start).clamp(min=0.0)
    union = pred_length + target_length - intersection
    return intersection / (union + eps)


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def iou_loss(pred: Tensor, target: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    """Standard 1 - IoU loss."""

    loss = 1.0 - temporal_iou(pred, target)
    return _reduce(loss, reduction)


def diou_loss(pred: Tensor, target: Tensor, *, reduction: Reduction = "mean", eps: float = 1e-6) -> Tensor:
    """Distance-IoU loss accounting for centre distance."""

    pred_start, pred_end = _prepare_segments(pred)
    target_start, target_end = _prepare_segments(target)

    iou = temporal_iou(pred, target, eps=eps)

    pred_center = 0.5 * (pred_start + pred_end)
    target_center = 0.5 * (target_start + target_end)
    center_dist = (pred_center - target_center) ** 2

    enclosing_start = torch.minimum(pred_start, target_start)
    enclosing_end = torch.maximum(pred_end, target_end)
    enclosing_length = (enclosing_end - enclosing_start).clamp(min=eps)

    penalty = center_dist / (enclosing_length ** 2)
    loss = 1.0 - iou + penalty
    return _reduce(loss, reduction)


def giou_loss(pred: Tensor, target: Tensor, *, reduction: Reduction = "mean", eps: float = 1e-6) -> Tensor:
    """Generalised IoU loss for 1D segments."""

    pred_start, pred_end = _prepare_segments(pred)
    target_start, target_end = _prepare_segments(target)

    iou = temporal_iou(pred, target, eps=eps)

    enclosing_start = torch.minimum(pred_start, target_start)
    enclosing_end = torch.maximum(pred_end, target_end)
    enclosing_length = (enclosing_end - enclosing_start).clamp(min=eps)

    inter_start = torch.maximum(pred_start, target_start)
    inter_end = torch.minimum(pred_end, target_end)
    intersection = (inter_end - inter_start).clamp(min=0.0)

    pred_length = (pred_end - pred_start).clamp(min=0.0)
    target_length = (target_end - target_start).clamp(min=0.0)
    union = pred_length + target_length - intersection

    giou = iou - (enclosing_length - union) / (enclosing_length + eps)
    loss = 1.0 - giou
    return _reduce(loss, reduction)
