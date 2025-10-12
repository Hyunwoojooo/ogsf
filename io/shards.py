"""Shard management helpers for WebDataset archives."""

from __future__ import annotations

import hashlib
import os
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

from . import schema

__all__ = [
    "DEFAULT_MIN_SHARD_SIZE",
    "DEFAULT_MAX_SHARD_SIZE",
    "DEFAULT_TARGET_SHARD_SIZE",
    "write_shards",
    "read_shard",
    "iter_shard_samples",
]

DEFAULT_MIN_SHARD_SIZE = 256 * 1024 * 1024
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024
DEFAULT_TARGET_SHARD_SIZE = 512 * 1024 * 1024


def _sanitize_token(token: str) -> str:
    allowed = "-_."
    return "".join(ch if ch.isalnum() or ch in allowed else "-" for ch in token)


def _to_array(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise AssertionError("feature matrix must be two-dimensional")
    return array


def _serialize_sample(sample: Mapping[str, object], index: int) -> Dict[str, bytes]:
    schema.validate(sample)

    video = _to_array(sample["video_feat"])  # type: ignore[index]
    text = _to_array(sample["text_feat"])  # type: ignore[index]

    base_name = f"{index:08d}_{_sanitize_token(str(sample['video_id']))}_{_sanitize_token(str(sample['qid']))}"

    meta = {
        "video_id": sample["video_id"],
        "qid": sample["qid"],
        "labels": [
            {"start": float(label["start"]), "end": float(label["end"])}  # type: ignore[index]
            for label in sample["labels"]  # type: ignore[index]
        ],
        "fps": float(sample["fps"]),
        "video_feat_shape": list(video.shape),
        "text_feat_shape": list(text.shape),
    }

    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    video_buffer = io.BytesIO()
    np.save(video_buffer, video, allow_pickle=False)
    video_bytes = video_buffer.getvalue()

    text_buffer = io.BytesIO()
    np.save(text_buffer, text, allow_pickle=False)
    text_bytes = text_buffer.getvalue()

    payloads: Dict[str, bytes] = {
        f"{base_name}/meta.json": meta_bytes,
        f"{base_name}/video.npy": video_bytes,
        f"{base_name}/text.npy": text_bytes,
    }

    object_feat = sample.get("object_feat")
    if object_feat is not None:
        obj_array = _to_array(object_feat)  # type: ignore[arg-type]
        meta["object_feat_shape"] = list(obj_array.shape)
        object_buffer = io.BytesIO()
        np.save(object_buffer, obj_array.astype(np.float32), allow_pickle=False)
        payloads[f"{base_name}/objects.npy"] = object_buffer.getvalue()

    object_mask = sample.get("object_mask")
    if object_mask is not None:
        mask_array = np.asarray(object_mask, dtype=np.float32).reshape(-1)
        meta["object_mask_shape"] = [int(mask_array.shape[0])]
        mask_buffer = io.BytesIO()
        np.save(mask_buffer, mask_array, allow_pickle=False)
        payloads[f"{base_name}/object_mask.npy"] = mask_buffer.getvalue()

    payloads[f"{base_name}/meta.json"] = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return payloads


def write_shards(
    samples: Sequence[schema.NLQSample],
    output_dir: str | Path,
    *,
    prefix: str = "nlq",
    min_shard_size: int = DEFAULT_MIN_SHARD_SIZE,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    target_shard_size: int = DEFAULT_TARGET_SHARD_SIZE,
) -> List[Path]:
    """Persist *samples* into WebDataset-compatible tar shards."""
    if min_shard_size > max_shard_size:
        raise ValueError("min_shard_size must be <= max_shard_size")
    if not (min_shard_size <= target_shard_size <= max_shard_size):
        raise ValueError("target_shard_size must lie between min and max shard size")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_paths: List[Path] = []
    tar_handle: tarfile.TarFile | None = None
    current_path: Path | None = None
    current_size = 0

    def _start_new_shard() -> None:
        nonlocal tar_handle, current_path, current_size
        shard_index = len(shard_paths)
        current_path = output_path / f"{prefix}-{shard_index:06d}.tar"
        tar_handle = tarfile.open(current_path, mode="w")
        current_size = 0

    def _finish_current_shard() -> None:
        nonlocal tar_handle, current_path
        if tar_handle is not None:
            tar_handle.close()
            if current_path is not None:
                shard_paths.append(current_path)
        tar_handle = None
        current_path = None

    for index, sample in enumerate(samples):
        payloads = _serialize_sample(sample, index)
        sample_size = sum(len(blob) for blob in payloads.values()) + 512 * len(payloads)

        if tar_handle is None:
            _start_new_shard()

        if (
            current_size >= target_shard_size
            and current_size >= min_shard_size
            and current_size + sample_size > max_shard_size
        ):
            _finish_current_shard()
            _start_new_shard()

        assert tar_handle is not None  # for type checkers
        for name, data in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(time.time())
            tar_handle.addfile(info, io.BytesIO(data))
            current_size += info.size + (512 - (info.size % 512 or 512))

    _finish_current_shard()

    if not shard_paths:
        return []

    index_lines = []
    for path in shard_paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        index_lines.append(f"{digest.hexdigest()}  {path.name}")

    index_file = output_path / f"{prefix}.sha256"
    index_file.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    return shard_paths


def read_shard(path: str | Path) -> List[schema.NLQSample]:
    """Load a shard from *path* and return contained samples."""
    shard_path = Path(path)
    samples: Dict[str, Dict[str, bytes]] = {}
    order: List[str] = []

    with tarfile.open(shard_path, mode="r") as tar_handle:
        for member in tar_handle:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            base = parts[0]
            filename = parts[-1]
            file_obj = tar_handle.extractfile(member)
            if file_obj is None:
                continue
            data = file_obj.read()
            if base not in samples:
                samples[base] = {}
                order.append(base)
            samples[base][filename] = data

    result: List[schema.NLQSample] = []
    for base in order:
        bundle = samples[base]
        meta = json.loads(bundle["meta.json"].decode("utf-8"))
        video = np.load(io.BytesIO(bundle["video.npy"]))
        text = np.load(io.BytesIO(bundle["text.npy"]))

        restored: Dict[str, object] = dict(
            schema.NLQSample(
                video_id=str(meta["video_id"]),
                qid=str(meta["qid"]),
                video_feat=video.astype(np.float32).tolist(),
                text_feat=text.astype(np.float32).tolist(),
                labels=[{"start": float(lbl["start"]), "end": float(lbl["end"])} for lbl in meta["labels"]],
                fps=float(meta["fps"]),
            )
        )

        if "objects.npy" in bundle:
            object_feat = np.load(io.BytesIO(bundle["objects.npy"]))
            restored["object_feat"] = object_feat.astype(np.float32).tolist()
        if "object_mask.npy" in bundle:
            object_mask = np.load(io.BytesIO(bundle["object_mask.npy"]))
            restored["object_mask"] = object_mask.astype(np.float32).tolist()

        result.append(restored)  # type: ignore[arg-type]

    return result


def iter_shard_samples(paths: Iterable[str | Path]) -> Iterator[schema.NLQSample]:
    """Yield samples from the provided shard *paths* sequentially."""
    for shard in paths:
        for sample in read_shard(shard):
            yield sample
