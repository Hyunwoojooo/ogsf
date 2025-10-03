"""Offline video feature loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = ["load_feature_matrix", "prepare_feature_sequence", "load_and_prepare_video_features"]


def load_feature_matrix(path: str | Path, *, mmap: bool = True) -> np.ndarray:
    """Load a feature matrix stored as `.npy` with optional memory mapping."""
    mmap_mode = "r" if mmap else None
    array = np.load(Path(path), mmap_mode=mmap_mode)
    if array.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    return array


def _l2_normalize(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def prepare_feature_sequence(
    features: np.ndarray,
    *,
    target_length: int,
    pad_value: float = 0.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize, pad, and mask a feature sequence to *target_length*."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    materialized = np.asarray(features, dtype=np.float32)
    if materialized.ndim != 2:
        raise ValueError("features must be two-dimensional")

    if normalize and len(materialized):
        materialized = _l2_normalize(materialized)

    num_steps, dim = materialized.shape
    if num_steps == 0:
        padded = np.full((target_length, dim), pad_value, dtype=np.float32)
        mask = np.zeros((target_length,), dtype=np.float32)
        return padded, mask

    if target_length <= num_steps:
        indices = np.linspace(0, num_steps - 1, target_length)
        indices = np.clip(indices.round().astype(int), 0, num_steps - 1)
        resized = materialized[indices]
        mask = np.ones((target_length,), dtype=np.float32)
        return resized.astype(np.float32), mask

    padded = np.full((target_length, dim), pad_value, dtype=np.float32)
    padded[:num_steps] = materialized
    mask = np.concatenate(
        [np.ones((num_steps,), dtype=np.float32), np.zeros((target_length - num_steps,), dtype=np.float32)]
    )
    return padded, mask


def load_and_prepare_video_features(
    path: str | Path,
    *,
    target_length: int,
    pad_value: float = 0.0,
    normalize: bool = True,
    mmap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load features from *path* and return a padded/normalized sequence."""
    matrix = load_feature_matrix(path, mmap=mmap)
    # Ensure we have an in-memory float32 array before further processing
    if isinstance(matrix, np.memmap):
        matrix = np.asarray(matrix, dtype=np.float32)
    return prepare_feature_sequence(
        matrix,
        target_length=target_length,
        pad_value=pad_value,
        normalize=normalize,
    )
