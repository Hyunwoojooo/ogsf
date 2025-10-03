"""Tests for ROI track feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from em.objects import track_features


def test_track_feature_extractor_normalises_embeddings() -> None:
    torch = pytest.importorskip("torch")

    frame = torch.rand(3, 24, 24)
    tracks = [
        {"track_id": 7, "t": 0, "bbox": [2.0, 2.0, 12.0, 16.0]},
        {"track_id": 8, "t": 0, "bbox": [10.0, 6.0, 20.0, 20.0]},
    ]

    config = track_features.TrackFeatureConfig(embedding_dim=32, pool_size=8, normalize=True)
    extractor = track_features.TrackFeatureExtractor(config)
    outputs = extractor.extract([frame], [tracks])

    assert len(outputs) == len(tracks)
    for record in outputs:
        emb = record["embedding"]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (config.embedding_dim,)
        norm = record["embedding_norm"]
        np.testing.assert_allclose(np.linalg.norm(emb), norm, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_extract_track_features_helper_matches_class() -> None:
    torch = pytest.importorskip("torch")

    frame = torch.rand(3, 18, 18)
    tracks = [{"track_id": 1, "t": 3, "bbox": [0.0, 0.0, 4.0, 4.0]}]

    config = track_features.TrackFeatureConfig(embedding_dim=16, pool_size=6)
    outputs = track_features.extract_track_features(frame, tracks, config=config)

    extractor = track_features.TrackFeatureExtractor(config)
    outputs_cls = extractor.extract([frame], [tracks])

    assert len(outputs) == len(outputs_cls) == 1
    np.testing.assert_allclose(outputs[0]["embedding"], outputs_cls[0]["embedding"], atol=1e-6)

