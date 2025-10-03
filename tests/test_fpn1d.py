"""Tests for the 1D feature pyramid network layer module."""

from __future__ import annotations

import pytest

from groundnlq.em.models.layers import fpn1d


def test_temporal_fpn_produces_expected_scales() -> None:
    torch = pytest.importorskip("torch")

    config = fpn1d.TemporalFPNConfig(in_channels=32, out_channels=64, num_levels=3)
    module = fpn1d.TemporalFPN(config)

    features = torch.randn(2, 64, 32)
    outputs = module(features)

    assert len(outputs) == 3
    lengths = [tensor.shape[1] for tensor in outputs]
    channels = [tensor.shape[2] for tensor in outputs]
    assert lengths == [64, 32, 16]
    assert channels.count(64) == 3


def test_temporal_fpn_supports_backprop() -> None:
    torch = pytest.importorskip("torch")

    config = fpn1d.TemporalFPNConfig(in_channels=16, out_channels=32, num_levels=3)
    module = fpn1d.TemporalFPN(config)

    features = torch.randn(1, 32, 16, requires_grad=True)
    outputs = module(features)
    loss = sum(out.sum() for out in outputs)
    loss.backward()

    assert features.grad is not None
