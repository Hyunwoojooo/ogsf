"""Tests for YOLO inference utilities."""

from __future__ import annotations

import numpy as np

from em.objects import yolo_infer


class DummyDetector:
    def __init__(self) -> None:
        self.invocations: list[int] = []

    def __call__(self, frame: np.ndarray):
        frame_index = len(self.invocations) * 2
        self.invocations.append(frame_index)
        bbox = [float(frame_index), 0.0, float(frame_index + 2), 2.0]
        return [{"bbox": bbox, "cls": 1, "conf": 0.9}]


def test_run_yolo_inference_stride_and_sorting():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(6)]
    config = yolo_infer.YoloInferenceConfig(frame_stride=2, conf_threshold=0.5)
    detector = DummyDetector()

    detections = yolo_infer.run_yolo_inference(frames, config=config, detector=detector)

    assert len(detections) == 3
    assert [d["t"] for d in detections] == [0, 2, 4]
    for detection in detections:
        assert detection["conf"] >= 0.5
        assert len(detection["bbox"]) == 4
