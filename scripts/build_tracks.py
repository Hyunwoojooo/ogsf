"""CLI for building and persisting object tracks."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Iterator, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ..objects import tracking

__all__ = ["build_tracks", "save_tracks", "main"]


def load_detections(path: Path) -> List[List[dict]]:
    payload = pickle.loads(path.read_bytes())
    return [list(frame) for frame in payload.get("detections", [])]


def build_tracks(
    detections: Sequence[Sequence[dict]],
    *,
    config: tracking.TrackerConfig | None = None,
) -> List[dict]:
    """Run tracking on detection sequences and return track records."""

    tracker_config = config or tracking.TrackerConfig()
    return tracking.run_tracking(detections, config=tracker_config)


def save_tracks(tracks: Sequence[dict], path: Path) -> None:
    payload = {"tracks": list(tracks)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))


def _iter_detection_files(source: Path, pattern: str | None, limit: int | None) -> Iterator[Path]:
    if source.is_dir():
        candidates = sorted(source.glob(pattern or "*.pkl"))
    else:
        candidates = [source]

    for index, item in enumerate(candidates):
        if limit is not None and index >= limit:
            break
        if item.is_file():
            yield item


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Build object tracks from detections")
    parser.add_argument("--source", type=Path, required=True, help="Detection pickle or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory for tracks")
    parser.add_argument("--pattern", default="*.pkl", help="Glob when --source is a directory")
    parser.add_argument("--limit", type=int, help="Optional maximum number of files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing track pickles")
    parser.add_argument("--backend", default="bytetrack", choices=["bytetrack", "simple"], help="Tracker backend to use")
    parser.add_argument("--frame-rate", type=float, default=30.0, help="Frame rate hint for tracker backends")
    parser.add_argument("--tracker-cfg", type=Path, help="Optional tracker configuration YAML path")
    parser.add_argument("--max-age", type=int, default=30, help="Max age for simple tracker (frames)")
    parser.add_argument("--match-iou", type=float, default=0.3, help="IoU match threshold for simple tracker")
    parser.add_argument("--interpolate-missing", action="store_true", help="Enable gap interpolation for simple tracker")
    parser.add_argument("--interpolate-gap", type=int, default=5, help="Maximum gap for interpolation")
    args = parser.parse_args()

    source = args.source
    output = args.output

    treat_as_directory = source.is_dir() or (
        source.is_file()
        and (
            output.is_dir()
            or output.suffix == ""
            or (output.exists() and output.is_dir())
        )
    )

    if source.is_file() and not treat_as_directory and output.suffix != ".pkl":
        raise SystemExit("For a single detection file, --output must be a .pkl file or directory.")

    if treat_as_directory:
        output.mkdir(parents=True, exist_ok=True)

    tracker_cfg = tracking.TrackerConfig(
        backend=args.backend,
        frame_rate=args.frame_rate,
        tracker_config_path=str(args.tracker_cfg) if args.tracker_cfg else None,
        max_age=args.max_age,
        match_iou_threshold=args.match_iou,
        interpolate_missing=args.interpolate_missing,
        interpolate_max_gap=args.interpolate_gap,
    )

    for det_path in _iter_detection_files(source, args.pattern, args.limit):
        out_path = output / det_path.name if treat_as_directory else output
        if out_path.exists() and not args.overwrite:
            print(f"[build_tracks] Skipping existing file: {out_path}")
            continue
        detections = load_detections(det_path)
        tracks_output = build_tracks(detections, config=tracker_cfg)
        save_tracks(tracks_output, out_path)
        print(f"[build_tracks] Wrote {out_path} ({len(tracks_output)} tracks)")


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
