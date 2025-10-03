"""Heads for NLQ score prediction and temporal boundary regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["NLQHeadConfig", "NLQHead"]


@dataclass
class NLQHeadConfig:
    """Configuration for the NLQ prediction head."""

    embed_dim: int
    hidden_dim: int = 256
    dropout: float = 0.1
    kernel_size: int = 3


class NLQHead(nn.Module):
    """Temporal head that outputs confidence scores and boundary offsets."""

    def __init__(self, config: NLQHeadConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2

        self.shared = nn.Sequential(
            nn.Conv1d(config.embed_dim, config.hidden_dim, kernel_size=config.kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.score_head = nn.Conv1d(config.hidden_dim, 1, kernel_size=1)
        self.bounds_head = nn.Conv1d(config.hidden_dim, 2, kernel_size=1)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Return score logits ``[B, 1, T]`` and boundary deltas ``[B, 2, T]``."""

        if features.ndim != 3:
            raise ValueError("features must be shaped [B, T, D]")

        x = features.permute(0, 2, 1)  # [B, D, T]
        x = self.shared(x)
        scores = self.score_head(x)
        bounds = torch.tanh(self.bounds_head(x))
        return scores, bounds
