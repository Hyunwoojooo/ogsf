"""Post-processing utilities including Soft-NMS and TTA aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from .recall_iou import Segment

__all__ = [
    "soft_nms_temporal",
    "aggregate_tta",
]


def _temporal_iou(a: Segment, b: Segment) -> float:
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    intersection = max(0.0, inter_end - inter_start)
    union = max(0.0, a.end - a.start) + max(0.0, b.end - b.start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _ensure_segment(item: Mapping[str, float] | Segment) -> Segment:
    if isinstance(item, Segment):
        return item
    return Segment(start=float(item["start"]), end=float(item["end"]), score=float(item.get("score", 1.0)))


def soft_nms_temporal(
    proposals: Sequence[Mapping[str, float] | Segment],
    *,
    sigma: float = 0.5,
    iou_threshold: float = 0.5,
    min_score: float = 1e-3,
    top_k: int | None = None,
) -> List[Segment]:
    """Apply a Gaussian Soft-NMS pass to temporal proposals."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    remaining = [_ensure_segment(item) for item in proposals]
    remaining.sort(key=lambda seg: seg.score, reverse=True)
    results: List[Segment] = []

    while remaining and (top_k is None or len(results) < top_k):
        current = remaining.pop(0)
        results.append(current)

        new_remaining: List[Segment] = []
        for proposal in remaining:
            iou = _temporal_iou(current, proposal)
            if iou > iou_threshold:
                weight = math.exp(- (iou ** 2) / sigma)
                proposal = Segment(start=proposal.start, end=proposal.end, score=proposal.score * weight)
            if proposal.score >= min_score:
                new_remaining.append(proposal)
        new_remaining.sort(key=lambda seg: seg.score, reverse=True)
        remaining = new_remaining

    return results


def aggregate_tta(
    predictions: Sequence[Sequence[Mapping[str, float] | Segment]],
    *,
    iou_threshold: float = 0.6,
    soft_nms_sigma: float | None = 0.5,
) -> List[Segment]:
    """Aggregate TTA predictions by averaging aligned segments."""

    pooled: List[Segment] = []
    for batch in predictions:
        pooled.extend(_ensure_segment(item) for item in batch)

    if not pooled:
        return []

    pooled.sort(key=lambda seg: seg.score, reverse=True)
    merged: List[Segment] = []

    for proposal in pooled:
        match_idx = None
        for idx, existing in enumerate(merged):
            if _temporal_iou(existing, proposal) >= iou_threshold:
                match_idx = idx
                break
        if match_idx is None:
            merged.append(proposal)
        else:
            existing = merged[match_idx]
            weight = existing.score + proposal.score
            if weight <= 0:
                continue
            start = (existing.start * existing.score + proposal.start * proposal.score) / weight
            end = (existing.end * existing.score + proposal.end * proposal.score) / weight
            merged[match_idx] = Segment(start=start, end=end, score=weight / 2)

    if soft_nms_sigma is not None:
        merged = soft_nms_temporal(merged, sigma=soft_nms_sigma, iou_threshold=iou_threshold)

    return merged
