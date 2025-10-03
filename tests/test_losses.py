"""Numerical unit tests for loss functions."""

from __future__ import annotations

import math

import pytest

from groundnlq.em.models.losses import asl, focal, iou1d


def test_focal_loss_matches_hand_computation() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.tensor([0.0, 1.5], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = focal.focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean")

    prob = torch.sigmoid(logits)
    ce = -(targets * torch.log(prob) + (1 - targets) * torch.log(1 - prob))
    p_t = torch.where(targets >= 0.5, prob, 1 - prob)
    modulating = (1 - p_t) ** 2.0
    alpha_factor = torch.where(targets >= 0.5, torch.tensor(0.25), torch.tensor(0.75))
    expected = (ce * modulating * alpha_factor).mean()

    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-5, abs_tol=1e-6)


def test_iou_losses_known_segments() -> None:
    torch = pytest.importorskip("torch")

    pred = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    iou = iou1d.temporal_iou(pred, target)
    assert math.isclose(iou.item(), 0.5, rel_tol=1e-5)

    iou_loss_val = iou1d.iou_loss(pred, target)
    assert math.isclose(iou_loss_val.item(), 0.5, rel_tol=1e-5)

    diou = iou1d.diou_loss(pred, target)
    assert diou.item() > iou_loss_val.item()

    giou = iou1d.giou_loss(pred, target)
    assert giou.item() >= 0.0


def test_asl_weighting_applies_sensitivity() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.zeros(1, 3)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    sensitivity = torch.tensor([[0.0, 1.0, 2.0]])

    loss = asl.asl_loss(logits, targets, sensitivity, base_weight=1.0, reduction="none")
    base = torch.full((1, 3), fill_value=torch.log(torch.tensor(2.0)))
    expected = base * (1.0 + sensitivity)

    assert torch.allclose(loss, expected)
