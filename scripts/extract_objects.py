"""CLI for running YOLOv12 object detection on ego-centric videos."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

from ..objects import yolo_infer

__all__ = ["run_object_detection", "save_detections", "main"]


def _iter_videos(source: Path, pattern: str | None, limit: int | None) -> Iterable[Path]:
    if source.is_dir():
        videos = sorted(source.glob(pattern or "*.mp4"))
    else:
        videos = [source]

    for index, video in enumerate(videos):
        if limit is not None and index >= limit:
            break
        if video.is_file():
            yield video


def run_object_detection(video_path: Path, config: yolo_infer.YoloInferenceConfig) -> List[List[yolo_infer.Detection]]:
    """Run YOLO inference on *video_path* using *config*."""

    flat_detections = yolo_infer.run_yolo_inference(str(video_path), config=config)
    if not flat_detections:
        return []

    max_frame = int(max(det.get("t", 0) for det in flat_detections))
    grouped: List[List[yolo_infer.Detection]] = [[] for _ in range(max_frame + 1)]
    for det in flat_detections:
        frame_index = int(det.get("t", 0))
        grouped[frame_index].append(det)
    return grouped


def save_detections(
    detections: Sequence[Sequence[yolo_infer.Detection]],
    *,
    output_path: Path,
    video_path: Path,
    config: yolo_infer.YoloInferenceConfig,
) -> None:
    """Persist YOLO detections to pickle for downstream processing."""

    payload = {
        "video": str(video_path),
        "detections": list(detections),
        "config": asdict(config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(payload))


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Run YOLOv12 object detection on Ego4D videos")
    parser.add_argument("--source", type=Path, required=True, help="Video file or directory of videos")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory for detection pickles")
    parser.add_argument("--model", type=Path, help="YOLO checkpoint (defaults to ultralytics built-in weights)")
    parser.add_argument("--device", default="cuda:0", help="Torch device string (e.g. cuda:0 or cpu)")
    parser.add_argument("--frame-stride", type=int, default=3, help="Frame sampling stride for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern when --source is a directory")
    parser.add_argument("--limit", type=int, help="Optional maximum number of videos to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing detection files")
    args = parser.parse_args()

    source = args.source
    output_root = args.output

    treat_as_directory = source.is_dir() or (
        source.is_file()
        and (
            (output_root.exists() and output_root.is_dir())
            or output_root.suffix == ""
        )
    )

    if source.is_file() and not treat_as_directory and output_root.suffix != ".pkl":
        raise SystemExit("When processing a single video, --output must be a .pkl file or a directory.")

    config = yolo_infer.YoloInferenceConfig(
        model_path=str(args.model) if args.model else None,
        device=args.device,
        frame_stride=args.frame_stride,
        conf_threshold=args.conf,
    )

    for video_path in _iter_videos(source, args.pattern, args.limit):
        output_path = output_root / f"{video_path.stem}.pkl" if treat_as_directory else output_root

        if output_path.exists() and not args.overwrite:
            print(f"[extract_objects] Skipping existing file: {output_path}")
            continue

        try:
            detections = run_object_detection(video_path, config)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"[extract_objects] Failed to process {video_path.name}: {exc}")
            continue

        save_detections(detections, output_path=output_path, video_path=video_path, config=config)
        print(f"[extract_objects] Wrote {output_path} ({len(detections)} detections)")


if __name__ == "__main__":  # pragma: no cover
    main()
