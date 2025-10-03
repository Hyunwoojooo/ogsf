"""WebDataset reader pipeline components."""

from __future__ import annotations

import glob
import random
import time
from collections.abc import Iterable, Iterator
from typing import Any, Dict, List, Sequence

import numpy as np

from . import schema, shards

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch optional for tests
    torch = None

__all__ = ["create_pipeline"]


def _expand_shards(pattern: str | Sequence[str]) -> List[str]:
    if isinstance(pattern, (list, tuple)):
        files: List[str] = []
        for item in pattern:
            files.extend(sorted(glob.glob(str(item))))
        return files
    return sorted(glob.glob(str(pattern)))


def _shuffle_iterator(samples: Iterator[schema.NLQSample], buffer_size: int) -> Iterator[schema.NLQSample]:
    if buffer_size <= 0:
        yield from samples
        return
    buffer: List[schema.NLQSample] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            index = random.randrange(len(buffer))
            yield buffer.pop(index)
    while buffer:
        yield buffer.pop()


def _map_sample(sample: schema.NLQSample, *, pin_memory: bool) -> Dict[str, Any]:
    video = np.asarray(sample["video_feat"], dtype=np.float32)
    text = np.asarray(sample["text_feat"], dtype=np.float32)

    record: Dict[str, Any] = {
        "video_id": sample["video_id"],
        "qid": sample["qid"],
        "video_feat": video,
        "text_feat": text,
        "labels": sample["labels"],
        "fps": float(sample["fps"]),
    }

    if pin_memory:
        if torch is None:
            raise RuntimeError("pin_memory requested but torch is not available")
        record["video_feat"] = torch.from_numpy(video).pin_memory()
        record["text_feat"] = torch.from_numpy(text).pin_memory()

    return record


def _collate_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Direct list passthrough keeps variable-length support.
    return batch


class DataPipeline:
    """Iterable over batches with basic timing statistics."""

    def __init__(
        self,
        sample_iter: Iterator[schema.NLQSample],
        *,
        batch_size: int,
        pin_memory: bool,
    ) -> None:
        self._sample_iter = sample_iter
        self._batch_size = max(1, batch_size)
        self._pin_memory = pin_memory
        self.stats: Dict[str, List[float]] = {"data_time": [], "step_time": []}

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        last_yield = time.perf_counter()
        while True:
            data_start = time.perf_counter()
            batch: List[Dict[str, Any]] = []
            try:
                for _ in range(self._batch_size):
                    sample = next(self._sample_iter)
                    batch.append(_map_sample(sample, pin_memory=self._pin_memory))
            except StopIteration:
                if batch:
                    data_elapsed = time.perf_counter() - data_start
                    self.stats["data_time"].append(data_elapsed)
                    batch_out = _collate_batch(batch)
                    now = time.perf_counter()
                    self.stats["step_time"].append(now - last_yield)
                    yield batch_out
                break

            data_elapsed = time.perf_counter() - data_start
            self.stats["data_time"].append(data_elapsed)
            batch_out = _collate_batch(batch)
            now = time.perf_counter()
            self.stats["step_time"].append(now - last_yield)
            last_yield = now
            yield batch_out


def create_pipeline(
    shards_glob: str | Sequence[str],
    *,
    batch_size: int,
    num_workers: int = 0,  # noqa: ARG001 - placeholder for future parallelism
    prefetch: int = 2,  # noqa: ARG001 - retained for interface compatibility
    shuffle_buf: int = 0,
    pin_memory: bool = False,
) -> DataPipeline:
    """Build an iterable pipeline over shards matching *shards_glob*."""
    shard_paths = _expand_shards(shards_glob)
    if not shard_paths:
        raise FileNotFoundError(f"No shards matched pattern: {shards_glob}")

    sample_iter = shards.iter_shard_samples(shard_paths)
    shuffled_iter = _shuffle_iterator(sample_iter, shuffle_buf)
    pipeline = DataPipeline(shuffled_iter, batch_size=batch_size, pin_memory=pin_memory)
    return pipeline
