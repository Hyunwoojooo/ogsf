"""Tests for tracking utilities."""

from __future__ import annotations

from typing import List

import numpy as np

from em.objects import tracking


def _det(bbox, conf=0.9, cls=0):
    return {"bbox": list(map(float, bbox)), "conf": conf, "cls": cls}


def test_tracking_id_persistence_with_occlusion():
    detections: List[List[dict]] = [
        [_det([0, 0, 2, 2], conf=0.9), _det([8, 0, 10, 2], conf=0.9)],
        [_det([1, 0, 3, 2], conf=0.9)],
        [_det([2, 0, 4, 2], conf=0.9), _det([7, 0, 9, 2], conf=0.9)],
        [_det([3, 0, 5, 2], conf=0.9)],
    ]

    config = tracking.TrackerConfig(backend="simple", max_age=2, match_iou_threshold=0.1)
    outputs = tracking.run_tracking(detections, config=config)

    track_ids = {record["track_id"] for record in outputs if record["cls"] == 0}
    assert len(track_ids) == 2

    track_0 = [rec for rec in outputs if rec["track_id"] == min(track_ids)]
    ts = [rec["t"] for rec in track_0]
    assert ts == sorted(ts)


def test_tracking_interpolates_missing_frames():
    detections = [
        [_det([0, 0, 2, 2])],
        [],
        [_det([2, 0, 4, 2])],
    ]

    config = tracking.TrackerConfig(
        backend="simple",
        interpolate_missing=True,
        interpolate_max_gap=2,
        match_iou_threshold=0.0,
    )
    outputs = tracking.run_tracking(detections, config=config)

    interpolated = [rec for rec in outputs if rec["score"] is None]
    assert len(interpolated) == 1
    interp = interpolated[0]
    assert interp["t"] == 1
    np.testing.assert_allclose(interp["bbox"], [1.0, 0.0, 3.0, 2.0], atol=1e-6)
