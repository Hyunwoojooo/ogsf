"""Tests for the NLQ prediction head."""

from __future__ import annotations

import numpy as np
import pytest

from groundnlq.em.models.heads import nlq_head


def test_nlq_head_output_shapes_and_ranges() -> None:
    torch = pytest.importorskip("torch")

    config = nlq_head.NLQHeadConfig(embed_dim=64, hidden_dim=32, dropout=0.0)
    head = nlq_head.NLQHead(config)

    features = torch.randn(2, 20, 64, requires_grad=True)
    scores, bounds = head(features)

    assert scores.shape == (2, 1, 20)
    assert bounds.shape == (2, 2, 20)
    assert torch.all(bounds.abs() <= 1.0 + 1e-6)

    loss = scores.mean() + bounds.sum()
    loss.backward()
    assert features.grad is not None
