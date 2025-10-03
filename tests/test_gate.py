"""Tests for the object gating layer module."""

from __future__ import annotations

import pytest

from groundnlq.em.models.layers import gate


def test_object_gate_modulates_logits() -> None:
    torch = pytest.importorskip("torch")

    config = gate.ObjectGateConfig(input_dim=8, hidden_dim=1)
    module = gate.ObjectGate(config)
    with torch.no_grad():
        module.mlp[0].weight.fill_(0.5)
        module.mlp[0].bias.zero_()
        module.mlp[-1].weight.fill_(1.0)
        module.mlp[-1].bias.fill_(-4.0)

    attn = torch.ones(2, 1, 4, 6)
    feats = torch.zeros(2, 4, 8)
    feats[1] = 1.0

    gated, gates = module(attn, feats)

    assert gated.shape == attn.shape
    assert gates.shape == (2, 4, 1)
    assert torch.all((gates >= 0.0) & (gates <= 1.0))

    mean_empty = gated[0].mean().item()
    mean_present = gated[1].mean().item()
    assert mean_present > mean_empty


def test_object_gate_respects_valid_mask() -> None:
    torch = pytest.importorskip("torch")

    config = gate.ObjectGateConfig(input_dim=4, hidden_dim=4)
    module = gate.ObjectGate(config)

    attn = torch.ones(1, 5, 3)
    feats = torch.ones(1, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.bool)

    gated, gates = module(attn, feats, valid_mask=mask)

    assert gated[:, 2].abs().sum().item() == 0.0
    assert gated[:, 3].abs().sum().item() == 0.0
    assert gates[0, 4] > 0.0
