"""Object gating layers for spatial-temporal fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["ObjectGateConfig", "ObjectGate"]


@dataclass
class ObjectGateConfig:
    """Configuration describing the object gate MLP."""

    input_dim: int
    hidden_dim: int = 128
    dropout: float = 0.0


class ObjectGate(nn.Module):
    """Predicts per-frame gates to modulate attention logits."""

    def __init__(self, config: ObjectGateConfig) -> None:
        super().__init__()
        self.config = config

        layers = [nn.Linear(config.input_dim, config.hidden_dim), nn.GELU()]
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Encourage low confidence in absence of objects.
        final_linear = self.mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, -2.0)

    def forward(
        self,
        attn_logits: Tensor,
        object_features: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return gated logits and the gate values.

        Parameters
        ----------
        attn_logits:
            Attention logits to modulate before the softmax. Shape ``[B, ..., T, L]``.
        object_features:
            Per-frame object descriptors with shape ``[B, T, F]``.
        valid_mask:
            Optional boolean mask ``[B, T]`` indicating which frames contain
            valid objects.
        """

        if object_features.ndim != 3:
            raise ValueError("object_features must be shaped [B, T, F]")

        gates = torch.sigmoid(self.mlp(object_features))  # [B, T, 1]
        gates = gates.to(device=attn_logits.device, dtype=attn_logits.dtype)
        
        if valid_mask is not None:
            if valid_mask.shape != object_features.shape[:2]:
                raise ValueError("valid_mask must match [B, T] dimensions")
            mask = valid_mask.unsqueeze(-1).to(device=gates.device, dtype=gates.dtype)
            gates = gates * mask

        expanded = gates
        while expanded.ndim < attn_logits.ndim:
            expanded = expanded.unsqueeze(1)

        gated = attn_logits * expanded
        return gated, gates
