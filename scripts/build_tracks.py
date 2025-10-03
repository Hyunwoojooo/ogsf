"""CLI for building and persisting object tracks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence

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


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Build object tracks from detections")
    parser.add_argument("detections", type=Path, help="Path to detection pickle")
    parser.add_argument("output", type=Path, help="Path to write tracks pickle")
    args = parser.parse_args()

    detections = load_detections(args.detections)
    tracks_output = build_tracks(detections)
    save_tracks(tracks_output, args.output)

