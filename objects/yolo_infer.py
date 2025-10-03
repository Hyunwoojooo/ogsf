"""YOLOv12 inference utilities for episodic memory frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except ModuleNotFoundError:  # pragma: no cover - OpenCV optional
    cv2 = None  # type: ignore[assignment]


@dataclass
class YoloInferenceConfig:
    """Runtime options for YOLO inference."""

    model_path: str | None = None
    device: str = "cpu"
    frame_stride: int = 1
    conf_threshold: float = 0.25


Detection = MutableMapping[str, Any]


def _default_detector(model_path: str | None, device: str) -> Callable[[np.ndarray], Sequence[Any]]:
    try:  # pragma: no cover - heavy dependency
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics must be installed to run YOLO inference") from exc

    model_reference = model_path or "yolov8n-world.pt"
    model = YOLO(model_reference)
    model.to(device)

    def _run(frame: np.ndarray) -> Sequence[Any]:
        return model(frame)

    return _run


def _iter_frames(source: Iterable[np.ndarray] | str | Path) -> Iterable[np.ndarray]:
    if isinstance(source, (str, Path)):
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("OpenCV is required to decode video sources")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
    else:
        for frame in source:
            yield frame


def _normalize_detection(raw: Any, timestamp: int, threshold: float) -> List[Detection]:
    detections: List[Detection] = []

    if isinstance(raw, Mapping):
        conf = float(raw.get("conf", raw.get("confidence", 0.0)))
        if conf < threshold:
            return []
        bbox = list(map(float, raw.get("bbox", raw.get("xyxy", []))))
        if len(bbox) != 4:
            raise ValueError("Detections must provide a 4-element bounding box")
        detections.append({
            "t": timestamp,
            "bbox": bbox,
            "cls": int(raw.get("cls", raw.get("class", -1))),
            "conf": conf,
        })
        return detections

    if hasattr(raw, "boxes"):
        boxes = raw.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.asarray([])
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.asarray([])
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.asarray([])
        for bbox, score, label in zip(xyxy, conf, cls):
            if float(score) < threshold:
                continue
            detections.append({
                "t": timestamp,
                "bbox": bbox.astype(float).tolist(),
                "cls": int(label),
                "conf": float(score),
            })
        return detections

    if isinstance(raw, Sequence):
        for item in raw:
            detections.extend(_normalize_detection(item, timestamp, threshold))
        return detections

    raise TypeError(f"Unsupported detection output type: {type(raw)}")


def run_yolo_inference(
    frames: Iterable[np.ndarray] | str | Path,
    *,
    config: YoloInferenceConfig,
    detector: Callable[[np.ndarray], Sequence[Any]] | None = None,
) -> List[Detection]:
    """Run YOLO detection over *frames* and return sorted detections."""
    if config.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    detector_fn = detector or _default_detector(config.model_path, config.device)

    results: List[Detection] = []
    for index, frame in enumerate(_iter_frames(frames)):
        if index % config.frame_stride != 0:
            continue
        outputs = detector_fn(frame)
        detections = _normalize_detection(outputs, index, config.conf_threshold)
        results.extend(detections)

    results.sort(key=lambda item: (item["t"], -item["conf"]))
    return results
