"""Smoke tests for MVP model."""

from __future__ import annotations

import pytest

from groundnlq.em.models import mvp


def test_mvp_forward_and_loss() -> None:
    torch = pytest.importorskip("torch")

    config = mvp.MVPConfig(d_v=16, d_t=12, hidden=32, heads=4, num_layers=2, dropout=0.0)
    model = mvp.MVPModel(config)

    video = torch.randn(2, 10, 16)
    text = torch.randn(2, 6, 12)

    outputs = model(video, text)
    assert outputs["scores"].shape == (2, 1, 10)
    assert outputs["bounds"].shape == (2, 2, 10)

    targets = {
        "scores": torch.rand(2, 10),
        "bounds": torch.rand(2, 2, 10),
    }

    loss, components = model.compute_loss(outputs, targets)
    assert loss.requires_grad
    loss.backward()

    assert set(components.keys()) == {"score", "bounds"}
