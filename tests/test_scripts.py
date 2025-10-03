"""Tests for lightweight CLI helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from groundnlq.em.io import schema, shards
from groundnlq.em.scripts import build_tracks, eval_submit, extract_objects, make_wds


def _sample(idx: int) -> schema.NLQSample:
    rng = np.random.default_rng(seed=idx)
    video = rng.standard_normal((4, 4)).astype(np.float32)
    text = rng.standard_normal((2, 3)).astype(np.float32)
    return schema.NLQSample(
        video_id=f"vid-{idx:03d}",
        qid=f"q-{idx:03d}",
        video_feat=video.tolist(),
        text_feat=text.tolist(),
        labels=[{"start": float(idx), "end": float(idx) + 0.5}],
        fps=24.0,
    )


def test_make_wds_roundtrip(tmp_path):
    samples = [_sample(i) for i in range(10)]
    shard_dir = tmp_path / "wds"
    shard_paths = make_wds.build_shards(samples, shard_dir, prefix="test", min_shard_size=8 * 1024, max_shard_size=16 * 1024, target_shard_size=12 * 1024)

    assert shard_paths
    restored = list(shards.iter_shard_samples(shard_paths))
    assert len(restored) == len(samples)
    assert restored[0]["video_id"] == samples[0]["video_id"]


def test_extract_and_build_tracks(tmp_path):
    frames = np.random.rand(5, 4, 4).astype(np.float32)
    detections = extract_objects.generate_detections(frames)
    det_path = tmp_path / "detections.pkl"
    extract_objects.save_detections(detections, det_path)

    payload = pickle.loads(det_path.read_bytes())
    assert "detections" in payload
    tracks = build_tracks.build_tracks(payload["detections"])
    track_path = tmp_path / "tracks.pkl"
    build_tracks.save_tracks(tracks, track_path)

    saved = pickle.loads(track_path.read_bytes())
    assert set(saved.keys()) == {"tracks"}
    assert all({"track_id", "t", "bbox"} <= set(item.keys()) for item in saved["tracks"])


def test_eval_submit_schema():
    predictions = [
        {
            "video_id": "vid-001",
            "qid": "q-001",
            "segments": [
                {"start": 0.0, "end": 1.0, "score": 0.8},
            ],
        }
    ]

    submission = eval_submit.build_submission(predictions)
    assert submission[0]["segments"][0]["start"] == 0.0
    text = json.dumps(submission)
    assert "vid-001" in text

