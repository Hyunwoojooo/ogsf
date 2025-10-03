"""Tests for the WebDataset reader pipeline."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from em.io import schema, shards, webdataset_reader


def _make_sample(idx: int) -> schema.NLQSample:
    time_steps = 4 + (idx % 3)
    dim_video = 8
    dim_text = 6
    rng = np.random.default_rng(seed=idx)
    video_feat = rng.standard_normal((time_steps, dim_video)).astype(np.float32)
    text_feat = rng.standard_normal((2, dim_text)).astype(np.float32)
    return schema.NLQSample(
        video_id=f"vid-{idx:04d}",
        qid=f"q-{idx:04d}",
        video_feat=video_feat.tolist(),
        text_feat=text_feat.tolist(),
        labels=[{"start": float(idx), "end": float(idx) + 1.0}],
        fps=24.0,
    )


def test_pipeline_batches_and_stats(tmp_path):
    samples: List[schema.NLQSample] = [_make_sample(i) for i in range(50)]
    shard_dir = tmp_path / "shards"
    shard_paths = shards.write_shards(
        samples,
        shard_dir,
        prefix="pipe",
        min_shard_size=8 * 1024,
        max_shard_size=32 * 1024,
        target_shard_size=16 * 1024,
    )
    assert shard_paths, "expected at least one shard to be created"

    pattern = str(shard_dir / "pipe-*.tar")
    pipeline = webdataset_reader.create_pipeline(
        pattern,
        batch_size=8,
        shuffle_buf=5,
        pin_memory=False,
    )

    all_batches = list(pipeline)
    assert all_batches, "pipeline should yield at least one batch"

    total = sum(len(batch) for batch in all_batches)
    assert total == len(samples)

    for batch in all_batches:
        assert len(batch) <= 8
        for sample in batch:
            assert "video_feat" in sample and "text_feat" in sample
            np.testing.assert_allclose(
                sample["fps"], 24.0, rtol=0, atol=1e-5
            )

    assert len(pipeline.stats["data_time"]) == len(all_batches)
    assert len(pipeline.stats["step_time"]) == len(all_batches)
    assert all(value >= 0 for value in pipeline.stats["data_time"])
    assert all(value >= 0 for value in pipeline.stats["step_time"])

    shard_size_bytes = [path.stat().st_size for path in shard_paths]
    assert all(size > 0 for size in shard_size_bytes)
    assert not math.isnan(sum(pipeline.stats["data_time"]))
