"""Configuration coverage tests for OGSF model."""

from __future__ import annotations

import itertools

import pytest

from groundnlq.em.models import ogsf


@pytest.mark.parametrize(
    "use_objects,use_multiscale,use_asl",
    list(itertools.product([False, True], repeat=3)),
)
def test_ogsf_forward_and_loss(use_objects: bool, use_multiscale: bool, use_asl: bool) -> None:
    torch = pytest.importorskip("torch")

    config = ogsf.OGSFConfig(
        d_v=20,
        d_t=12,
        hidden=16,
        heads=4,
        num_layers=1,
        dropout=0.0,
        use_objects=use_objects,
        use_multiscale=use_multiscale,
        use_asl=use_asl,
        object_dim=8,
        fpn_levels=3,
    )

    model = ogsf.OGSFModel(config)

    batch = 2
    time = 12
    video = torch.randn(batch, time, 20)
    text = torch.randn(batch, 5, 12)
    objects = torch.randn(batch, time, 8) if use_objects else None

    outputs = model(video, text, object_features=objects)
    assert outputs["scores"].shape[-1] == time
    assert outputs["bounds"].shape[-1] == time

    targets = {
        "scores": torch.rand(batch, time),
        "bounds": torch.rand(batch, 2, time),
    }
    if use_asl:
        targets["sensitivity"] = torch.rand(batch, time)

    loss, components = model.compute_loss(outputs, targets)
    assert loss.requires_grad
    loss.backward()

    assert "score" in components and "bounds" in components
    if use_multiscale:
        assert "multiscale" in components
    else:
        assert "multiscale" not in components
