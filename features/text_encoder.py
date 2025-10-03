"""Text encoder inference helpers for NLQ."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional torch dependency
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - torch available
    TORCH_IMPORT_ERROR = None

Tokenizer = Any
Model = nn.Module if nn is not None else Any


@dataclass
class TextEncoderConfig:
    """Configuration describing how to build a text encoder."""

    model_name: str
    device: str = "cpu"
    frozen: bool = True
    qlora: bool = False
    max_length: int = 64
    normalize: bool = True


class TextEncoder:
    """Wrapper around a tokenizer+model pair providing embedding utilities."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        model: Model,
        config: TextEncoderConfig,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:  # pragma: no cover - enforced at runtime
            raise RuntimeError("PyTorch is required to use TextEncoder") from TORCH_IMPORT_ERROR

        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.device = device or config.device

        self.model.to(self.device)
        _apply_training_policy(self.model, config)
        if config.frozen and not config.qlora:
            self.model.eval()

    def encode(self, texts: Sequence[str], *, normalize: Optional[bool] = None) -> np.ndarray:
        """Tokenize *texts* and return pooled embeddings as `np.ndarray`."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to use TextEncoder") from TORCH_IMPORT_ERROR

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        norm = self.config.normalize if normalize is None else normalize
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        inference_context = (
            torch.no_grad()
            if self.config.frozen and not self.config.qlora
            else contextlib.nullcontext()
        )

        was_training = self.model.training
        if self.config.frozen and not self.config.qlora:
            self.model.eval()

        with inference_context:
            outputs = self.model(**inputs)

        embedding = _extract_embeddings(outputs)
        if embedding.device != torch.device("cpu"):
            embedding = embedding.cpu()

        if norm:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        embedding_cpu = embedding.detach().cpu().to(torch.float32)
        array = np.asarray(embedding_cpu.tolist(), dtype=np.float32)

        if was_training and (self.config.qlora or not self.config.frozen):
            self.model.train()

        return array

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the encoder outputs."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required") from TORCH_IMPORT_ERROR
        for param in self.model.parameters():
            if param.ndim >= 2:
                return param.shape[-1]
        raise RuntimeError("Unable to infer embedding dimension from model parameters")

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters respecting the policy."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required") from TORCH_IMPORT_ERROR
        total = 0
        for param in self.model.parameters():
            if getattr(param, "requires_grad", False):
                total += int(param.numel())
        return total


def load_text_encoder(
    config: TextEncoderConfig,
    *,
    builder: Optional[Callable[[TextEncoderConfig], Tuple[Tokenizer, Model, str]]] = None,
) -> TextEncoder:
    """Instantiate a :class:`TextEncoder` using the provided *builder*."""
    if builder is None:
        builder = _default_builder
    tokenizer, model, device = builder(config)
    return TextEncoder(tokenizer=tokenizer, model=model, config=config, device=device)


def _default_builder(config: TextEncoderConfig) -> Tuple[Tokenizer, Model, str]:  # pragma: no cover - heavy dependency
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("PyTorch is required for the default text encoder builder") from TORCH_IMPORT_ERROR

    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers must be installed to load text encoders") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    return tokenizer, model, config.device


def _extract_embeddings(outputs: Any) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from TORCH_IMPORT_ERROR

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise RuntimeError("Unsupported model outputs for text encoder")


def _apply_training_policy(model: Model, config: TextEncoderConfig) -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from TORCH_IMPORT_ERROR

    if config.qlora:
        _enable_qlora_mode(model)
        model.train()
        return

    requires_grad = not config.frozen
    for param in model.parameters():
        param.requires_grad = requires_grad


def _enable_qlora_mode(model: Model) -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from TORCH_IMPORT_ERROR

    adapter_found = False
    for name, param in model.named_parameters():
        lower = name.lower()
        if "adapter" in lower or "lora" in lower:
            param.requires_grad = True
            adapter_found = True
        else:
            param.requires_grad = False

    if adapter_found:
        return

    bias_enabled = False
    for name, param in model.named_parameters():
        if "bias" in name.lower():
            param.requires_grad = True
            bias_enabled = True
        else:
            param.requires_grad = False

    if not bias_enabled:
        first_param: Optional[torch.nn.Parameter] = None
        for param in model.parameters():
            first_param = param
            break
        if first_param is None:
            raise RuntimeError("Model has no parameters to enable for QLoRA")
        first_param.requires_grad = True
