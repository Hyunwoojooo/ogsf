"""Recall metrics for temporal IoU evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import torch

from ..models.losses.iou1d import temporal_iou

__all__ = [
    "Segment",
    "compute_recall_at_iou",
    "recall_at_k",
]


@dataclass(frozen=True)
class Segment:
    """Temporal segment prediction."""

    start: float
    end: float
    score: float = 1.0


def _to_segment(item: Mapping[str, float] | Segment) -> Segment:
    if isinstance(item, Segment):
        return item
    return Segment(start=float(item["start"]), end=float(item["end"]), score=float(item.get("score", 1.0)))


def _prepare_segments(values: Sequence[Mapping[str, float] | Segment]) -> List[Segment]:
    return sorted((_to_segment(item) for item in values), key=lambda seg: seg.score, reverse=True)


def _iou(pred: Segment, target: Segment) -> float:
    pred_tensor = torch.tensor([[pred.start, pred.end]], dtype=torch.float32)
    target_tensor = torch.tensor([[target.start, target.end]], dtype=torch.float32)
    return float(temporal_iou(pred_tensor, target_tensor).item())


def compute_recall_at_iou(
    predictions: Sequence[Sequence[Mapping[str, float] | Segment]],
    ground_truths: Sequence[Sequence[Mapping[str, float] | Segment]],
    *,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    max_predictions: int | None = 100,
) -> MutableMapping[float, float]:
    """Return recall for each IoU threshold across the dataset."""

    thresholds = list(sorted(thresholds))
    matched = {thr: 0 for thr in thresholds}
    total_gt = 0

    for pred_list, gt_list in zip(predictions, ground_truths):
        preds = _prepare_segments(pred_list)
        if max_predictions is not None:
            preds = preds[:max_predictions]
        gts = [_to_segment(gt) for gt in gt_list]
        total_gt += len(gts)

        for gt in gts:
            best = max((_iou(pred, gt) for pred in preds), default=0.0)
            for thr in thresholds:
                if best >= thr:
                    matched[thr] += 1

    if total_gt == 0:
        return {thr: 0.0 for thr in thresholds}
    return {thr: matched[thr] / total_gt for thr in thresholds}


def recall_at_k(
    predictions: Sequence[Sequence[Mapping[str, float] | Segment]],
    ground_truths: Sequence[Sequence[Mapping[str, float] | Segment]],
    *,
    ks: Iterable[int] = (1, 5),
    iou_threshold: float = 0.5,
) -> MutableMapping[int, float]:
    """Compute Recall@K for a single IoU threshold."""

    results: MutableMapping[int, float] = {}
    for k in ks:
        results[k] = compute_recall_at_iou(
            predictions,
            ground_truths,
            thresholds=[iou_threshold],
            max_predictions=k,
        )[iou_threshold]
    return results
