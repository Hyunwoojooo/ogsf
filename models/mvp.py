"""NLQ MVP model definitions using early cross-attention fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .heads import NLQHead, NLQHeadConfig
from .layers import xattn
from .losses import focal, iou1d

__all__ = ["MVPConfig", "MVPModel"]


@dataclass
class MVPConfig:
    """Configuration for the MVP fusion model."""

    d_v: int
    d_t: int
    hidden: int = 256
    heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_flash_attn: bool = False


class _FusionLayer(nn.Module):
    """Single cross-attention block with feed-forward network."""

    def __init__(self, hidden: int, heads: int, dropout: float, use_flash: bool) -> None:
        super().__init__()
        self.attn = xattn.CrossAttention(
            xattn.CrossAttentionConfig(
                embed_dim=hidden,
                num_heads=heads,
                dropout=dropout,
                use_flash=use_flash,
            )
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        attn_out = self.attn(query, key, key)
        query = self.norm1(query + attn_out)
        ff_out = self.ff(query)
        return self.norm2(query + ff_out)


class MVPModel(nn.Module):
    """Minimal viable pipeline combining cross-attention and NLQ head."""

    def __init__(self, config: MVPConfig) -> None:
        super().__init__()
        self.config = config

        self.video_proj = nn.Linear(config.d_v, config.hidden)
        self.text_proj = nn.Linear(config.d_t, config.hidden)

        self.layers = nn.ModuleList(
            [_FusionLayer(config.hidden, config.heads, config.dropout, config.use_flash_attn) for _ in range(config.num_layers)]
        )

        self.head = NLQHead(
            NLQHeadConfig(
                embed_dim=config.hidden,
                hidden_dim=config.hidden,
                dropout=config.dropout,
            )
        )

    def encode(self, video: Tensor, text: Tensor) -> Tensor:
        """Return fused temporal features shaped ``[B, T, hidden]``."""

        video_emb = self.video_proj(video)
        text_emb = self.text_proj(text)

        fused = video_emb
        for layer in self.layers:
            fused = layer(fused, text_emb)
        return fused

    def forward(self, video: Tensor, text: Tensor) -> Dict[str, Tensor]:
        features = self.encode(video, text)
        scores, bounds = self.head(features)
        return {"scores": scores, "bounds": bounds, "features": features}

    def compute_loss(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute focal + IoU regression losses for MVP outputs."""

        score_logits = predictions["scores"].squeeze(1)
        score_targets = targets["scores"].to(device=score_logits.device, dtype=score_logits.dtype)
        score_loss = focal.focal_loss(score_logits, score_targets)

        pred_bounds = predictions["bounds"].transpose(1, 2)  # [B, T, 2]
        target_bounds = targets["bounds"].transpose(1, 2).to(device=pred_bounds.device, dtype=pred_bounds.dtype)
        bound_loss = iou1d.iou_loss(pred_bounds, target_bounds, reduction="mean") + F.l1_loss(predictions["bounds"], targets["bounds"].to(predictions["bounds"].device))

        total = score_loss + bound_loss
        return total, {"score": score_loss, "bounds": bound_loss}
