"""Tests for metric computation and post-processing modules."""

from __future__ import annotations

import numpy as np
import pytest

from groundnlq.em.metrics import postprocess, recall_iou


def test_recall_at_iou_matches_expected() -> None:
    preds = [
        [
            {"start": 0.0, "end": 1.0, "score": 0.9},
            {"start": 1.0, "end": 2.0, "score": 0.8},
        ],
        [
            {"start": 0.2, "end": 1.2, "score": 0.7},
        ],
    ]
    gts = [
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ],
        [
            {"start": 0.0, "end": 1.0},
        ],
    ]

    recall = recall_iou.compute_recall_at_iou(preds, gts, thresholds=[0.5, 0.75], max_predictions=2)
    assert pytest.approx(recall[0.5], rel=1e-5) == 1.0
    assert recall[0.75] < 1.0

    r_at_1 = recall_iou.recall_at_k(preds, gts, ks=[1], iou_threshold=0.5)
    assert pytest.approx(r_at_1[1], rel=1e-5) == 2 / 3


def test_soft_nms_temporal_reduces_duplicates() -> None:
    proposals = [
        {"start": 0.0, "end": 1.0, "score": 1.0},
        {"start": 0.1, "end": 1.1, "score": 0.95},
        {"start": 2.0, "end": 3.0, "score": 0.8},
    ]

    filtered = postprocess.soft_nms_temporal(proposals, sigma=0.5, iou_threshold=0.5, min_score=0.05)
    assert len(filtered) == 3
    assert filtered[0].score >= filtered[1].score >= filtered[2].score
    assert filtered[1].score < 0.95


def test_tta_aggregation_merges_segments() -> None:
    tta_predictions = [
        [
            {"start": 0.0, "end": 1.0, "score": 0.9},
            {"start": 2.0, "end": 3.0, "score": 0.5},
        ],
        [
            {"start": 0.1, "end": 1.1, "score": 0.8},
            {"start": 4.0, "end": 5.0, "score": 0.4},
        ],
    ]

    merged = postprocess.aggregate_tta(tta_predictions, iou_threshold=0.5, soft_nms_sigma=None)
    assert len(merged) == 3
    starts = np.array([seg.start for seg in merged])
    assert np.isclose(starts.min(), 0.05, atol=0.1)
