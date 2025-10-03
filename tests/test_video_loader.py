"""Tests for video feature loading utilities."""

from __future__ import annotations

import numpy as np

from em.features import video_loader


def test_load_feature_matrix_and_prepare(tmp_path):
    rng = np.random.default_rng(seed=0)
    matrix = rng.standard_normal((37, 512)).astype(np.float32)
    path = tmp_path / "feat.npy"
    np.save(path, matrix)

    loaded = video_loader.load_feature_matrix(path)
    assert loaded.shape == (37, 512)

    features_long, mask_long = video_loader.load_and_prepare_video_features(
        path, target_length=64
    )
    assert features_long.shape == (64, 512)
    assert mask_long.shape == (64,)
    assert int(mask_long.sum()) == 37

    non_zero = features_long[mask_long.astype(bool)]
    norms = np.linalg.norm(non_zero, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)

    features_short, mask_short = video_loader.load_and_prepare_video_features(
        path, target_length=16
    )
    assert features_short.shape == (16, 512)
    assert mask_short.shape == (16,)
    assert int(mask_short.sum()) == 16


def test_load_with_empty_file(tmp_path):
    empty = np.zeros((0, 256), dtype=np.float32)
    path = tmp_path / "empty.npy"
    np.save(path, empty)

    features, mask = video_loader.load_and_prepare_video_features(path, target_length=10)
    assert features.shape == (10, 256)
    assert mask.sum() == 0
