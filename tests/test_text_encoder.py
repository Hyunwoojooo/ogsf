"""Tests for the text encoder helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import importlib.metadata as importlib_metadata
import numpy as np
import pytest

try:
    _torch_version = importlib_metadata.version("torch")
except importlib_metadata.PackageNotFoundError:
    pytest.skip("PyTorch not installed", allow_module_level=True)
else:
    major = int(_torch_version.split(".")[0])
    if major < 2 and np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0"):
        pytest.skip("PyTorch <2.0 is incompatible with NumPy >=2.0", allow_module_level=True)

import torch
import torch.nn as nn

from em.features import text_encoder


class DummyTokenizer:
    """Tokenizer stub returning sequential token IDs."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(
        self,
        texts: Sequence[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict:
        del padding, truncation, max_length, return_tensors
        batch = len(texts)
        token_ids = torch.arange(batch * self.dim, dtype=torch.long).reshape(batch, self.dim)
        attention = torch.ones_like(token_ids)
        return {"input_ids": token_ids, "attention_mask": attention}


class DummyModelOutput:
    def __init__(self, embedding: torch.Tensor) -> None:
        self.last_hidden_state = embedding.unsqueeze(1)
        self.pooler_output = embedding


class DummyModel(nn.Module):
    """Simple encoder producing deterministic embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.base = nn.Linear(dim, dim, bias=False)
        self.adapter_layer = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.base.weight)
        nn.init.eye_(self.adapter_layer.weight)
        nn.init.constant_(self.adapter_layer.bias, 0.0)

    def forward(self, input_ids: torch.Tensor, **_) -> DummyModelOutput:
        one_hot = torch.nn.functional.one_hot(
            input_ids % self.base.weight.shape[0], num_classes=self.base.weight.shape[0]
        ).float()
        pooled = one_hot.mean(dim=1)
        embedding = self.adapter_layer(self.base(pooled))
        return DummyModelOutput(embedding)


@dataclass
class DummyBuilder:
    dim: int = 8

    def __call__(self, config: text_encoder.TextEncoderConfig):
        tokenizer = DummyTokenizer(dim=self.dim)
        model = DummyModel(dim=self.dim)
        return tokenizer, model, config.device


def test_encode_returns_unit_norm_embeddings():
    config = text_encoder.TextEncoderConfig(model_name="dummy", max_length=16)
    encoder = text_encoder.load_text_encoder(config, builder=DummyBuilder(dim=8))

    sentences = ["the first query", "second query", "third query"]
    embeddings = encoder.encode(sentences)

    assert embeddings.shape == (3, 8)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)
    assert encoder.trainable_parameter_count() == 0


def test_qlora_enables_only_adapter_parameters():
    config = text_encoder.TextEncoderConfig(model_name="dummy", qlora=True, frozen=False)
    encoder = text_encoder.load_text_encoder(config, builder=DummyBuilder(dim=4))

    adapter_params = 0
    other_params = 0
    for name, param in encoder.model.named_parameters():
        if "adapter" in name.lower():
            adapter_params += param.numel()
            assert param.requires_grad
        else:
            other_params += param.numel()
            assert not param.requires_grad

    assert encoder.trainable_parameter_count() == adapter_params
    assert other_params > 0


def test_encode_handles_empty_input():
    config = text_encoder.TextEncoderConfig(model_name="dummy")
    encoder = text_encoder.load_text_encoder(config, builder=DummyBuilder(dim=6))

    embeddings = encoder.encode([])
    assert embeddings.shape[0] == 0
    assert embeddings.shape[1] == 6
