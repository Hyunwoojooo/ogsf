"""Tests for the cross-attention layer module."""

from __future__ import annotations

import numpy as np
import pytest

from groundnlq.em.models.layers import xattn


def _make_inputs(embed_dim: int, tgt_len: int, src_len: int, batch: int = 2):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(1)
    query = torch.rand(batch, tgt_len, embed_dim, generator=generator)
    key = torch.rand(batch, src_len, embed_dim, generator=generator)
    value = torch.rand(batch, src_len, embed_dim, generator=generator)
    return query, key, value


def test_cross_attention_applies_masks_without_nan() -> None:
    torch = pytest.importorskip("torch")

    embed_dim, heads = 32, 4
    layer = xattn.CrossAttention(xattn.CrossAttentionConfig(embed_dim=embed_dim, num_heads=heads, dropout=0.0, use_flash=False))
    layer.eval()

    query, key, value = _make_inputs(embed_dim, tgt_len=5, src_len=7)
    key_padding_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    additive_mask = torch.zeros(5, 7)
    additive_mask[0, 0] = float("-inf")

    output = layer(query, key, value, key_padding_mask=key_padding_mask, attn_mask=additive_mask)

    assert output.shape == (2, 5, embed_dim)
    assert torch.isfinite(output).all()


def test_flash_attention_matches_reference_path() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("Flash attention path requires scaled_dot_product_attention")

    config = xattn.CrossAttentionConfig(embed_dim=24, num_heads=3, dropout=0.0, use_flash=False)
    reference = xattn.CrossAttention(config)
    flash_cfg = xattn.CrossAttentionConfig(embed_dim=24, num_heads=3, dropout=0.0, use_flash=True)
    flash = xattn.CrossAttention(flash_cfg)
    flash.load_state_dict(reference.state_dict())

    query, key, value = _make_inputs(24, tgt_len=4, src_len=6)
    mask = torch.zeros(4, 6)
    mask[1, 0] = float("-inf")

    out_ref = reference(query, key, value, attn_mask=mask)
    out_flash = flash(query, key, value, attn_mask=mask)

    np.testing.assert_allclose(out_flash.detach().numpy(), out_ref.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_cross_attention_is_torch_compile_compatible() -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    config = xattn.CrossAttentionConfig(embed_dim=16, num_heads=4, dropout=0.0, use_flash=hasattr(torch.nn.functional, "scaled_dot_product_attention"))
    module = xattn.CrossAttention(config)

    compiled = torch.compile(module)
    query, key, value = _make_inputs(16, tgt_len=3, src_len=5, batch=1)
    output = compiled(query, key, value)

    assert output.shape == (1, 3, 16)
    assert torch.isfinite(output).all()
