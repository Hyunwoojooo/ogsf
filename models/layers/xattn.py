"""Cross-attention layers compatible with FlashAttention backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None


__all__ = ["CrossAttention", "CrossAttentionConfig"]


@dataclass
class CrossAttentionConfig:
    """Configuration for :class:`CrossAttention`."""

    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    use_flash: bool = False


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional FlashAttention fast path."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        if torch is None:  # pragma: no cover - exercised when torch missing
            raise RuntimeError("PyTorch is required for cross attention") from TORCH_IMPORT_ERROR

        super().__init__()
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(p=config.dropout)

        self._flash_available = bool(config.use_flash and hasattr(F, "scaled_dot_product_attention"))

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        *,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply cross-attention to *query* using *key*/*value* memories."""

        if key is None:
            key = query
        if value is None:
            value = key

        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key and value must be rank-3 [B, T, D] tensors")

        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        mask = self._build_attention_mask(
            bsz=bsz,
            tgt_len=tgt_len,
            src_len=src_len,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            device=q.device,
            dtype=q.dtype,
        )

        if self._flash_available:
            dropout_p = self.dropout.p if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)

    def _shape(self, tensor: Tensor) -> Tensor:
        bsz, seq_len, _ = tensor.shape
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _build_attention_mask(
        self,
        *,
        bsz: int,
        tgt_len: int,
        src_len: int,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if key_padding_mask is None and attn_mask is None:
            return None

        expanded_shape = (bsz, self.num_heads, tgt_len, src_len)
        mask = None

        if key_padding_mask is not None:
            padding = key_padding_mask.to(torch.bool).view(bsz, 1, 1, src_len)
            mask_val = torch.finfo(dtype).min if torch.is_floating_point(torch.empty(0, dtype=dtype)) else -1e4
            mask = torch.zeros(expanded_shape, device=device, dtype=dtype)
            mask = mask.masked_fill(padding, mask_val)

        if attn_mask is not None:
            attn_mask = self._expand_attn_mask(attn_mask, expanded_shape, device, dtype)
            mask = attn_mask if mask is None else mask + attn_mask

        if mask is not None and mask.dtype != dtype:
            mask = mask.to(dtype)
        return mask

    def _expand_attn_mask(
        self,
        attn_mask: Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.masked_fill(attn_mask, torch.finfo(dtype).min)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        attn_mask = attn_mask.to(device=device, dtype=dtype)
        if attn_mask.shape != shape:
            attn_mask = attn_mask.expand(shape)
        return attn_mask
