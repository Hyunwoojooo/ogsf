"""CLI for generating submission files from inference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping

__all__ = ["build_submission", "main"]


def build_submission(
    predictions: Iterable[Mapping[str, object]],
) -> List[dict]:
    """Convert model predictions into the official submission structure."""

    formatted: List[dict] = []
    for entry in predictions:
        video_id = str(entry["video_id"])
        qid = str(entry["qid"])
        segments_raw = entry.get("segments", [])
        segments: List[dict] = []
        for seg in segments_raw:
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "score": float(seg.get("score", 1.0)),
                }
            )
        formatted.append({"video_id": video_id, "qid": qid, "segments": segments})
    return formatted


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Convert predictions to submission JSON")
    parser.add_argument("input", type=Path, help="Path to prediction JSON")
    parser.add_argument("output", type=Path, help="Path to submission JSON")
    args = parser.parse_args()

    predictions = json.loads(args.input.read_text(encoding="utf-8"))
    submission = build_submission(predictions)
    args.output.write_text(json.dumps(submission, indent=2), encoding="utf-8")

