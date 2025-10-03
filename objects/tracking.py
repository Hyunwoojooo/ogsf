"""Object tracking wrappers for ByteTrack and DeepSORT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableSequence, Sequence

import numpy as np


@dataclass
class TrackerConfig:
    """Configuration governing tracking behaviour."""

    backend: str = "bytetrack"
    max_age: int = 30
    match_iou_threshold: float = 0.3
    interpolate_missing: bool = False
    interpolate_max_gap: int = 5

    def __post_init__(self) -> None:
        supported = {"bytetrack", "deepsort"}
        if self.backend not in supported:
            raise ValueError(f"Unsupported tracker backend: {self.backend}")
        if self.max_age < 0:
            raise ValueError("max_age must be non-negative")
        if not (0.0 <= self.match_iou_threshold <= 1.0):
            raise ValueError("match_iou_threshold must be between 0 and 1")
        if self.interpolate_max_gap < 0:
            raise ValueError("interpolate_max_gap must be non-negative")


Detection = Dict[str, Any]
FrameDetections = Sequence[Detection]


def run_tracking(
    frame_detections: Sequence[FrameDetections],
    *,
    config: TrackerConfig,
) -> List[Detection]:
    """Track detections across frames and return per-frame track assignments."""
    active_tracks: List[Dict[str, Any]] = []
    outputs: List[Detection] = []
    next_id = 1

    for frame_index, detections in enumerate(frame_detections):
        matches, unmatched_tracks, unmatched_dets = _match_detections(
            active_tracks, detections, config.match_iou_threshold
        )

        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track["missed"] += 1

        for track_idx, detection_idx in matches:
            track = active_tracks[track_idx]
            detection = detections[detection_idx]
            previous_frame = track["last_frame"]
            previous_bbox = track["last_bbox"]

            gap = frame_index - previous_frame
            if (
                config.interpolate_missing
                and gap > 1
                and gap - 1 <= config.interpolate_max_gap
            ):
                interpolated = _interpolate_track(
                    track_id=track["id"],
                    start_frame=previous_frame,
                    end_frame=frame_index,
                    start_bbox=previous_bbox,
                    end_bbox=detection["bbox"],
                )
                outputs.extend(interpolated)

            record = _make_track_record(track["id"], frame_index, detection["bbox"], detection)
            outputs.append(record)

            track["last_bbox"] = detection["bbox"]
            track["last_frame"] = frame_index
            track["missed"] = 0

        # Remove expired tracks
        active_tracks = [track for track in active_tracks if track["missed"] <= config.max_age]

        # Handle unmatched detections
        for detection_index in unmatched_dets:
            detection = detections[detection_index]
            track_id = next_id
            next_id += 1
            track_state = {
                "id": track_id,
                "last_bbox": detection["bbox"],
                "last_frame": frame_index,
                "missed": 0,
            }
            active_tracks.append(track_state)
            outputs.append(_make_track_record(track_id, frame_index, detection["bbox"], detection))

    outputs.sort(key=lambda item: (item["t"], item["track_id"]))
    return outputs


def _make_track_record(track_id: int, frame_index: int, bbox: Sequence[float], detection: Detection) -> Detection:
    return {
        "track_id": track_id,
        "t": frame_index,
        "bbox": list(map(float, bbox)),
        "cls": detection.get("cls"),
        "score": detection.get("conf", detection.get("score")),
    }


def _match_detections(
    tracks: List[Dict[str, Any]],
    detections: Sequence[Detection],
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    scores: List[tuple[float, int, int]] = []
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            score = _iou(track["last_bbox"], detection["bbox"])
            scores.append((score, track_idx, det_idx))

    scores.sort(reverse=True, key=lambda item: item[0])

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: List[tuple[int, int]] = []

    for score, track_idx, det_idx in scores:
        if score < threshold:
            break
        if track_idx in matched_tracks or det_idx in matched_dets:
            continue
        matched_tracks.add(track_idx)
        matched_dets.add(det_idx)
        matches.append((track_idx, det_idx))

    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _interpolate_track(
    *,
    track_id: int,
    start_frame: int,
    end_frame: int,
    start_bbox: Sequence[float],
    end_bbox: Sequence[float],
) -> List[Detection]:
    gap = end_frame - start_frame
    if gap <= 1:
        return []

    start = np.asarray(start_bbox, dtype=np.float32)
    end = np.asarray(end_bbox, dtype=np.float32)

    interpolated: List[Detection] = []
    for step, frame in enumerate(range(start_frame + 1, end_frame)):
        alpha = (step + 1) / gap
        bbox = (1 - alpha) * start + alpha * end
        interpolated.append({
            "track_id": track_id,
            "t": frame,
            "bbox": bbox.tolist(),
            "cls": None,
            "score": None,
        })
    return interpolated
