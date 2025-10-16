"""CLI helpers for packing NLQ data into WebDataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from ..features import video_loader

try:  # pragma: no cover - optional dependency guard
    import yaml
except ModuleNotFoundError:  # pragma: no cover - kept optional for legacy behaviour
    yaml = None  # type: ignore[assignment]

from ..io import schema, shards

__all__ = ["build_shards", "main"]

OBJECT_FEATURE_DIM_DEFAULT = 4


@dataclass
class PathsConfig:
    raw_dir: Path
    anno_dir: Path
    nas_features: Path
    wds_shards: Path
    output_dir: Path


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


def _load_paths(paths_file: Path) -> PathsConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse --paths")

    payload = yaml.safe_load(paths_file.read_text(encoding="utf-8")) or {}
    required = ["RAW_DIR", "ANNO_DIR", "NAS_FEATURES", "WDS_SHARDS", "OUTPUT_DIR"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"paths file is missing keys: {', '.join(missing)}")

    return PathsConfig(
        raw_dir=Path(payload["RAW_DIR"]),
        anno_dir=Path(payload["ANNO_DIR"]),
        nas_features=Path(payload["NAS_FEATURES"]),
        wds_shards=Path(payload["WDS_SHARDS"]),
        output_dir=Path(payload["OUTPUT_DIR"]),
    )


def _load_manifest(manifest_path: Path, split: str) -> Optional[Sequence[str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = split.lower()

    if isinstance(data, Mapping):
        candidates = (
            data.get(split)
            or data.get(split.upper())
            or data.get(normalized)
        )
        if candidates is None:
            return None
        return _normalize_manifest_entries(candidates)

    if isinstance(data, Sequence):
        return _normalize_manifest_entries(data, split_hint=normalized)

    raise ValueError("manifest must be a mapping or a sequence")


def _normalize_manifest_entries(entries: Sequence, *, split_hint: Optional[str] = None) -> Sequence[str]:
    result: List[str] = []
    for item in entries:
        if isinstance(item, Mapping):
            if split_hint and item.get("split", split_hint).lower() != split_hint:
                continue
            value = (
                item.get("video_id")
                or item.get("clip_uid")
                or item.get("clip_id")
            )
            if value is not None:
                result.append(str(value))
        else:
            result.append(str(item))
    return result


def _iter_annotations(anno_path: Path) -> Iterator[Dict[str, object]]:
    data = json.loads(anno_path.read_text(encoding="utf-8"))
    videos = data.get("videos") if isinstance(data, Mapping) else None
    if not isinstance(videos, Sequence):
        raise ValueError("annotation JSON must contain a 'videos' list")

    for video_entry in videos:
        if not isinstance(video_entry, Mapping):
            continue
        video_uid = video_entry.get("video_uid") or video_entry.get("video_id")
        clips = video_entry.get("clips")
        fps = float(video_entry.get("fps", 30.0))
        if not isinstance(clips, Sequence):
            continue
        for clip in clips:
            if not isinstance(clip, Mapping):
                continue
            clip_uid = str(clip.get("clip_uid", clip.get("clip_id", "")))
            if not clip_uid:
                continue
            annotations = clip.get("annotations")
            clip_fps = float(clip.get("fps", fps))
            if not isinstance(annotations, Sequence):
                continue
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    continue
                ann_uid = str(annotation.get("annotation_uid", annotation.get("annotation_id", "")))
                queries = annotation.get("language_queries")
                if not isinstance(queries, Sequence):
                    continue
                for idx, query in enumerate(queries):
                    if not isinstance(query, Mapping):
                        continue
                    payload = {
                        "clip_uid": clip_uid,
                        "fps": clip_fps,
                        "annotation_uid": ann_uid,
                        "query_index": idx,
                        "query": query.get("query", ""),
                        "query_id": query.get("query_id", query.get("query_uid")),
                        "clip_start_sec": query.get("clip_start_sec", query.get("start_sec", query.get("start_time"))),
                        "clip_end_sec": query.get("clip_end_sec", query.get("end_sec", query.get("end_time"))),
                    }
                    if video_uid:
                        payload["video_uid"] = str(video_uid)
                    source_clip_uid = clip.get("source_clip_uid") or clip.get("clip_source_uid")
                    if source_clip_uid:
                        payload["source_clip_uid"] = str(source_clip_uid)
                    yield payload


def _load_video_feature(feat_dir: Path, video_id: str, cache: MutableMapping[str, np.ndarray]) -> Optional[np.ndarray]:
    if video_id in cache:
        return cache[video_id]

    candidates = [
        feat_dir / f"{video_id}.npy",
        feat_dir / f"{video_id}.npz",
        feat_dir / "video" / f"{video_id}.npy",
        feat_dir / "video" / f"{video_id}.npz",
    ]

    array: Optional[np.ndarray] = None
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "video" in data:
                    array = data["video"].astype(np.float32)
                elif "arr_0" in data:
                    array = data["arr_0"].astype(np.float32)
        else:
            array = np.load(path).astype(np.float32)
        if array is not None:
            if array.ndim == 1:
                array = array.reshape(1, -1)
            elif array.ndim >= 3:
                array = array.reshape(array.shape[0], -1)
            break

    if array is not None:
        cache[video_id] = array
        return array

    return None


def _load_text_feature(
    feat_dir: Path,
    video_id: str,
    qid: str,
    query_text: str,
) -> np.ndarray:
    candidates = [
        feat_dir / f"{video_id}.npz",
        feat_dir / "text" / f"{qid}.npy",
        feat_dir / "text" / f"{video_id}.npy",
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key_options = [
                    "text",
                    f"text_{qid}",
                    f"text_{video_id}",
                    "arr_1",
                ]
                for key in key_options:
                    if key in data:
                        array = data[key].astype(np.float32)
                        return array if array.ndim == 2 else array.reshape(array.shape[0], -1)
        else:
            array = np.load(path).astype(np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            return array

    return _deterministic_text_embedding(query_text)


def _deterministic_text_embedding(text: str, *, dim: int = 512) -> np.ndarray:
    tokens = text.split()
    token_count = max(1, len(tokens))
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((token_count, dim), dtype=np.float32)


def _load_object_features(
    track_dir: Optional[Path],
    video_id: str,
    target_length: int,
    *,
    object_dim: int = OBJECT_FEATURE_DIM_DEFAULT,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if track_dir is None or target_length <= 0:
        return None, None

    track_path = track_dir / f"{video_id}.pkl"
    if not track_path.exists():
        return None, None

    try:
        payload = pickle.loads(track_path.read_bytes())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[make_wds] warning: failed to read tracks for {video_id}: {exc}")
        return None, None

    records = payload.get("tracks", [])
    if not records:
        return (
            np.zeros((target_length, object_dim), dtype=np.float32),
            np.zeros((target_length,), dtype=np.float32),
        )

    max_frame = max(int(rec.get("t", 0)) for rec in records) + 1
    frame_count = max(max_frame, 1)
    counts = np.zeros(frame_count, dtype=np.float32)
    sum_scores = np.zeros(frame_count, dtype=np.float32)
    max_scores = np.zeros(frame_count, dtype=np.float32)

    for rec in records:
        t = int(rec.get("t", 0))
        if t < 0 or t >= frame_count:
            continue
        score = float(rec.get("score", rec.get("conf", 0.0)) or 0.0)
        counts[t] += 1.0
        sum_scores[t] += score
        if score > max_scores[t]:
            max_scores[t] = score

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_scores = np.divide(sum_scores, counts, out=np.zeros_like(sum_scores), where=counts > 0)

    stats = np.zeros((frame_count, object_dim), dtype=np.float32)
    stats[:, 0] = np.log1p(counts)
    if object_dim > 1:
        stats[:, 1] = max_scores
    if object_dim > 2:
        stats[:, 2] = mean_scores
    if object_dim > 3:
        stats[:, 3] = sum_scores

    mask = (counts > 0).astype(np.float32)

    features, _ = video_loader.prepare_feature_sequence(
        stats,
        target_length=target_length,
        normalize=False,
        pad_value=0.0,
    )
    mask_matrix, _ = video_loader.prepare_feature_sequence(
        mask.reshape(-1, 1),
        target_length=target_length,
        normalize=False,
        pad_value=0.0,
    )
    mask_sequence = mask_matrix.reshape(-1)
    return features.astype(np.float32), mask_sequence.astype(np.float32)


def _assemble_samples(
    *,
    anno_path: Path,
    feat_dir: Path,
    manifest_ids: Optional[Sequence[str]],
    take: Optional[int],
    tracks_dir: Optional[Path],
    object_dim: int,
) -> List[schema.NLQSample]:
    allowed = set(manifest_ids) if manifest_ids else None
    cache: Dict[str, np.ndarray] = {}
    samples: List[schema.NLQSample] = []

    for payload in _iter_annotations(anno_path):
        clip_uid = payload["clip_uid"]  # type: ignore[index]
        if allowed is not None and clip_uid not in allowed:
            continue

        video_feat = _load_video_feature(feat_dir, clip_uid, cache)
        if video_feat is None:
            video_uid = payload.get("video_uid")
            if isinstance(video_uid, str):
                video_feat = _load_video_feature(feat_dir, video_uid, cache)
        if video_feat is None:
            source_clip_uid = payload.get("source_clip_uid")
            if isinstance(source_clip_uid, str):
                video_feat = _load_video_feature(feat_dir, source_clip_uid, cache)
        if video_feat is None:
            print(f"[make_wds] warning: missing video features for {clip_uid}")
            continue

        qid = payload.get("query_id")
        if qid is None:
            qid = f"{payload['annotation_uid']}_{payload['query_index']}"
        qid = str(qid)

        start = payload.get("clip_start_sec")
        end = payload.get("clip_end_sec")
        if start is None or end is None:
            print(f"[make_wds] warning: missing temporal label for {qid} ({clip_uid})")
            continue

        text_feat = _load_text_feature(feat_dir, clip_uid, qid, str(payload.get("query", "")))

        sample: Dict[str, Any] = dict(
            schema.NLQSample(
            video_id=str(clip_uid),
            qid=qid,
            video_feat=video_feat,
            text_feat=text_feat,
            labels=[{"start": float(start), "end": float(end)}],
            fps=float(payload.get("fps", 30.0)),
        ))

        object_features, object_mask = _load_object_features(
            tracks_dir,
            clip_uid,
            video_feat.shape[0],
            object_dim=object_dim,
        )
        if object_features is not None and object_mask is not None:
            sample["object_feat"] = object_features.tolist()
            sample["object_mask"] = object_mask.tolist()
        samples.append(sample)

        if take is not None and len(samples) >= take:
            break

    return samples


def _run_structured_flow(args: argparse.Namespace) -> None:
    paths = _load_paths(args.paths)

    feat_dir = Path(args.feat_dir) if args.feat_dir else paths.nas_features
    anno_path = Path(args.anno) if args.anno else paths.anno_dir / f"nlq_{args.split}.json"

    manifest_ids = None
    if args.manifest:
        manifest_ids = _load_manifest(Path(args.manifest), args.split)

    tracks_dir = Path(args.tracks) if args.tracks else None

    samples = _assemble_samples(
        anno_path=anno_path,
        feat_dir=feat_dir,
        manifest_ids=manifest_ids,
        take=args.take,
        tracks_dir=tracks_dir,
        object_dim=args.object_dim,
    )

    if not samples:
        raise RuntimeError("no samples were produced; check annotations and feature paths")

    shard_root = paths.wds_shards / args.split
    print(f"[make_wds] collected {len(samples)} samples for split '{args.split}'")
    print(f"[make_wds] writing shards to {shard_root}")
    shard_root.mkdir(parents=True, exist_ok=True)

    target = int(args.shard_size_mb * 1024 * 1024)
    min_target = max(1024 * 1024, int(target * 0.8))
    max_target = int(target * 1.2)

    build_shards(
        samples,
        shard_root,
        prefix=f"nlq_{args.split}",
        min_shard_size=min_target,
        max_shard_size=max_target,
        target_shard_size=target,
    )


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Convert NLQ samples into WebDataset shards")
    parser.add_argument("input", nargs="?", type=Path, help="Path to pre-built NLQ sample JSON")
    parser.add_argument("output", nargs="?", type=Path, help="Directory to write shards into")
    parser.add_argument("--prefix", default="nlq", help="Shard filename prefix")

    parser.add_argument("--paths", type=Path, help="YAML file describing shared dataset paths")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--mode", choices=["features"], default="features")
    parser.add_argument("--feat_dir", type=Path, help="Directory containing precomputed features")
    parser.add_argument("--anno", type=Path, help="Annotation JSON to consume")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest with split subsets")
    parser.add_argument("--take", type=int, help="Optional number of samples to retain")
    parser.add_argument("--shard_size_mb", type=int, default=512, help="Target shard size in MB")
    parser.add_argument("--tracks", type=Path, help="Directory containing object track pickles")
    parser.add_argument("--object-dim", type=int, default=OBJECT_FEATURE_DIM_DEFAULT, help="Per-frame object feature dimension")

    args = parser.parse_args()

    if args.paths:
        _run_structured_flow(args)
        return

    if args.input is None or args.output is None:
        raise SystemExit("either provide --paths or specify input/output positional arguments")

    samples = _load_samples(args.input)
    build_shards(samples, args.output, prefix=args.prefix)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
