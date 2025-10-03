"""CLI for converting features into WebDataset shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from ..io import schema, shards

__all__ = ["build_shards", "main"]


def _load_samples(path: Path) -> List[schema.NLQSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input JSON must contain a list of samples")
    samples: List[schema.NLQSample] = []
    for sample in data:
        schema.validate(sample)
        samples.append(schema.NLQSample(**sample))
    return samples


def build_shards(
    samples: Iterable[schema.NLQSample],
    output_dir: Path,
    *,
    prefix: str = "nlq",
    min_shard_size: int = 64 * 1024,
    max_shard_size: int = 128 * 1024,
    target_shard_size: int = 96 * 1024,
) -> List[Path]:
    """Convert *samples* into shards and return created paths."""

    return shards.write_shards(
        list(samples),
        output_dir,
        prefix=prefix,
        min_shard_size=min_shard_size,
        max_shard_size=max_shard_size,
        target_shard_size=target_shard_size,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Convert NLQ samples into WebDataset shards")
    parser.add_argument("input", type=Path, help="Path to JSON file containing NLQ samples")
    parser.add_argument("output", type=Path, help="Directory to write shards into")
    parser.add_argument("--prefix", default="nlq", help="Shard filename prefix")
    args = parser.parse_args()

    samples = _load_samples(args.input)
    build_shards(samples, args.output, prefix=args.prefix)
