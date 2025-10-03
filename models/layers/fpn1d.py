"""One-dimensional feature pyramid network layers for temporal aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["TemporalFPNConfig", "TemporalFPN"]


@dataclass
class TemporalFPNConfig:
    """Configuration for a 1D temporal feature pyramid."""

    in_channels: int
    out_channels: int
    num_levels: int = 3


class TemporalFPN(nn.Module):
    """Constructs a temporal feature pyramid from a single-resolution input."""

    def __init__(self, config: TemporalFPNConfig) -> None:
        super().__init__()
        if config.num_levels < 1:
            raise ValueError("num_levels must be >= 1")

        self.config = config
        self.stem = nn.Conv1d(config.in_channels, config.out_channels, kernel_size=3, padding=1)

        self.bottom_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for level in range(config.num_levels):
            stride = 1 if level == 0 else 2
            conv = nn.Conv1d(config.out_channels, config.out_channels, kernel_size=3, stride=stride, padding=1)
            self.bottom_convs.append(conv)
            self.lateral_convs.append(nn.Conv1d(config.out_channels, config.out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv1d(config.out_channels, config.out_channels, kernel_size=3, padding=1))

    def forward(self, features: Tensor) -> List[Tensor]:
        """Return a list of multi-scale features with descending resolutions."""

        if features.ndim != 3:
            raise ValueError("features must be shaped [B, T, C]")

        x = features.permute(0, 2, 1)  # [B, C, T]
        x = F.relu(self.stem(x))

        bottoms: List[Tensor] = []
        current = x
        for conv in self.bottom_convs:
            current = F.relu(conv(current))
            bottoms.append(current)

        results: List[Optional[Tensor]] = [None] * len(bottoms)
        prev = None

        for idx in reversed(range(len(bottoms))):
            lateral = self.lateral_convs[idx](bottoms[idx])
            if prev is not None:
                prev = F.interpolate(prev, size=lateral.shape[-1], mode="linear", align_corners=False)
                lateral = lateral + prev
            output = self.output_convs[idx](lateral)
            results[idx] = output
            prev = lateral

        return [tensor.permute(0, 2, 1) for tensor in results if tensor is not None]
