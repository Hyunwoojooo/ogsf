"""ROI embedding extraction utilities for tracked objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

import numpy as np

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


__all__ = ["TrackFeatureConfig", "TrackFeatureExtractor", "extract_track_features"]


@dataclass
class TrackFeatureConfig:
    """Configuration controlling ROI feature extraction."""

    embedding_dim: int = 128
    pool_size: int = 14
    normalize: bool = True
    device: str = "cpu"
    dtype: torch.dtype | None = None if torch is None else torch.float32


class _RoiEncoder(nn.Module):  # pragma: no cover - thin inference network
    """Small CNN used to embed cropped regions of interest."""

    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(32, embedding_dim // 2)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden, embedding_dim)

    def forward(self, roi: Tensor) -> Tensor:  # (N, C, H, W) -> (N, D)
        features = self.backbone(roi)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)


class TrackFeatureExtractor:
    """Helper that embeds tracked object crops."""

    def __init__(self, config: TrackFeatureConfig) -> None:
        if torch is None:  # pragma: no cover - exercised in tests when torch missing
            raise RuntimeError("PyTorch is required for track feature extraction") from TORCH_IMPORT_ERROR

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype or torch.float32
        self.encoder = _RoiEncoder(in_channels=3, embedding_dim=config.embedding_dim)
        self.encoder.to(self.device, dtype=self.dtype)
        self.encoder.eval()

    def extract(
        self,
        frames: Iterable[Tensor],
        tracks_per_frame: Iterable[Sequence[MutableMapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Return per-track embeddings suitable for serialization."""

        results: List[Dict[str, Any]] = []
        for frame_index, (frame, tracks) in enumerate(zip(frames, tracks_per_frame)):
            prepared = self._prepare_frame(frame)
            for track in tracks:
                bbox = track.get("bbox")
                if bbox is None:
                    continue
                roi = self._crop_roi(prepared, bbox)
                roi = roi.unsqueeze(0)
                embedding = self.encoder(roi)
                if self.config.normalize:
                    embedding = F.normalize(embedding, p=2, dim=1)
                embedding_cpu = embedding.detach().cpu().to(torch.float32)
                record = {
                    "track_id": int(track.get("track_id", -1)),
                    "t": int(track.get("t", frame_index)),
                    "bbox": [float(x) for x in bbox],
                    "embedding": np.asarray(embedding_cpu[0].tolist(), dtype=np.float32),
                    "embedding_norm": float(embedding_cpu[0].norm(p=2).item()),
                }
                results.append(record)
        return results

    def _prepare_frame(self, frame: Tensor) -> Tensor:
        if frame.ndim != 3:
            raise ValueError("Expected frame tensor with shape [C, H, W]")
        if frame.dtype.is_floating_point:
            tensor = frame.to(device=self.device, dtype=self.dtype)
        else:
            tensor = frame.to(device=self.device, dtype=self.dtype) / 255.0
        return tensor

    def _crop_roi(self, frame: Tensor, bbox: Sequence[float]) -> Tensor:
        if len(bbox) != 4:
            raise ValueError("Bounding box must contain four coordinates")

        c, h, w = frame.shape
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(w, np.floor(x1))))
        x2 = int(max(0, min(w, np.ceil(x2))))
        y1 = int(max(0, min(h, np.floor(y1))))
        y2 = int(max(0, min(h, np.ceil(y2))))

        if x2 <= x1 or y2 <= y1:
            return torch.zeros((c, self.config.pool_size, self.config.pool_size), device=self.device, dtype=self.dtype)

        crop = frame[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            return torch.zeros((c, self.config.pool_size, self.config.pool_size), device=self.device, dtype=self.dtype)

        crop = crop.unsqueeze(0)
        resized = F.interpolate(
            crop,
            size=(self.config.pool_size, self.config.pool_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)


def extract_track_features(
    frame: Tensor,
    tracks: Sequence[MutableMapping[str, Any]],
    *,
    config: Optional[TrackFeatureConfig] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper that encodes a single frame."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for track feature extraction") from TORCH_IMPORT_ERROR

    extractor = TrackFeatureExtractor(config or TrackFeatureConfig())
    return extractor.extract([frame], [tracks])
