"""Tests for shard writing and reading utilities."""

from __future__ import annotations

import numpy as np
import pytest

from em.io import schema, shards


def _dummy_sample(idx: int, *, length: int = 4, dim: int = 16) -> schema.NLQSample:
    rng = np.random.default_rng(seed=idx)
    video = rng.standard_normal((length, dim)).astype(np.float32)
    text = rng.standard_normal((2, dim // 2)).astype(np.float32)
    return schema.NLQSample(
        video_id=f"video-{idx:04d}",
        qid=f"query-{idx:04d}",
        video_feat=video.tolist(),
        text_feat=text.tolist(),
        labels=[{"start": float(idx), "end": float(idx) + 0.5}],
        fps=30.0,
    )


def test_write_and_read_roundtrip(tmp_path):
    samples = [_dummy_sample(i) for i in range(100)]
    shard_dir = tmp_path / "output"
    try:
        shard_paths = shards.write_shards(
        samples,
        shard_dir,
        prefix="nlqtest",
        min_shard_size=16 * 1024,
        max_shard_size=64 * 1024,
        target_shard_size=32 * 1024,
    )

    except RuntimeError as exc:
        if "SHM" in str(exc):
            pytest.skip("Shared memory unavailable in sandbox")
        raise

    assert shard_paths, "expected at least one shard to be written"
    index_file = shard_dir / "nlqtest.sha256"
    assert index_file.is_file(), "sha256 index file should be generated"

    restored = []
    for path in shard_paths:
        restored.extend(shards.read_shard(path))

    assert len(restored) == len(samples)

    for original, loaded in zip(samples, restored):
        assert original["video_id"] == loaded["video_id"]
        assert original["qid"] == loaded["qid"]
        np.testing.assert_allclose(
            np.asarray(original["video_feat"], dtype=np.float32),
            np.asarray(loaded["video_feat"], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(original["text_feat"], dtype=np.float32),
            np.asarray(loaded["text_feat"], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_write_shards_respects_size_constraints(tmp_path):
    samples = [_dummy_sample(i, length=2, dim=4) for i in range(10)]
    shard_dir = tmp_path / "size"

    with pytest.raises(ValueError):
        shards.write_shards(samples, shard_dir, min_shard_size=64, max_shard_size=32)
