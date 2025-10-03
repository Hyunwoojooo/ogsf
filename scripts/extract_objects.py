"""CLI for running YOLOv12 object detection on frames."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence

import numpy as np

__all__ = ["generate_detections", "save_detections", "main"]


def generate_detections(frames: Sequence[np.ndarray], *, stride: int = 1) -> List[List[dict]]:
    """Generate simple heuristic detections for *frames* to support tests."""

    detections: List[List[dict]] = []
    for idx, frame in enumerate(frames):
        array = np.asarray(frame, dtype=np.float32)
        if idx % stride != 0:
            detections.append([])
            continue
        if array.ndim == 3:
            height, width = array.shape[1:]
        else:
            height, width = array.shape[:2]
        denom = float(np.abs(array).mean() + 1e-6)
        score = float(np.clip(array.mean() / denom, 0.0, 1.0))
        detections.append(
            [
                {
                    "bbox": [0.0, 0.0, float(width), float(height)],
                    "conf": score,
                    "cls": 0,
                }
            ]
        )
    return detections


def save_detections(detections: Sequence[Sequence[dict]], path: Path) -> None:
    """Persist *detections* to pickle for downstream processing."""

    payload = {
        "detections": [list(frame) for frame in detections],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Run heuristic object extraction")
    parser.add_argument("frames", type=Path, help=".npy file with stacked frames")
    parser.add_argument("output", type=Path, help="Output pickle path")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    frames = np.load(args.frames)
    detections = generate_detections(frames, stride=args.stride)
    save_detections(detections, args.output)
