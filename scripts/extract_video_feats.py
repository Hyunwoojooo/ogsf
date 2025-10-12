"""InternVideo 기반 비디오 피처 추출 스크립트.

이 스크립트는 InternVideo 모델을 이용해 영상으로부터 피처를 추출해
`.npy` 파일로 저장합니다. 200개 샘플 Smoke 테스트나 실제 학습 파이프라인
준비에 바로 활용할 수 있도록 GPU 한 장에서 순차 처리하는 구조로
구현했습니다.

추출 설정(백본, FPS, 클립 길이 등)은 명령행 인자로 조정할 수 있으며,
`make_wds`가 기대하는 디렉터리 구조에 맞춰 저장합니다.

필수 종속성: torch, torchvision, internvideo. 모델 체크포인트는
`--checkpoint`로 지정하거나, 기본 위치(환경 변수 `INTERNVIDEO_CKPT`)를
이용하십시오.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


def _import_internvideo_module():
    try:
        import InternVideo.internvideo as internvideo  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "InternVideo 모듈을 찾을 수 없습니다. PYTHONPATH에 "
            "InternVideo/InternVideo1/Pretrain/Multi-Modalities-Pretraining 경로를 추가했는지 확인하세요."
        ) from exc
    return internvideo


@dataclass
class ExtractionConfig:
    src_dir: Path
    out_dir: Path
    backbone: str
    fps: float
    clip_len: int
    stride: int
    fp16: bool
    pattern: Optional[str]
    overwrite: bool
    limit: Optional[int]
    resize: int
    checkpoint: Optional[Path]
    device: torch.device


def _iter_videos(directory: Path, *, pattern: Optional[str], limit: Optional[int]) -> Iterable[Path]:
    if pattern:
        files = sorted(directory.glob(pattern))
    else:
        files = sorted(
            file
            for file in directory.iterdir()
            if file.suffix.lower() in VIDEO_EXTENSIONS and file.is_file()
        )

    count = 0
    for file in files:
        yield file
        count += 1
        if limit is not None and count >= limit:
            break


def _load_video(path: Path) -> Tuple[np.ndarray, float]:
    import decord  # type: ignore

    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    array = vr.get_batch(range(len(vr))).asnumpy()
    return array, fps


def _sample_frames(frames: np.ndarray, *, fps_original: float, fps_target: float) -> np.ndarray:
    if fps_target <= 0:
        return frames
    fps_original = max(fps_original, 1e-3)
    step = max(int(round(fps_original / fps_target)), 1)
    sampled = frames[::step]
    if sampled.size == 0:
        return frames
    return sampled


def _split_into_clips(frames: np.ndarray, clip_len: int, stride: int) -> np.ndarray:
    total = frames.shape[0]
    if total == 0:
        return np.zeros((0, clip_len, *frames.shape[1:]), dtype=frames.dtype)

    stride = max(1, stride)
    clips: List[np.ndarray] = []
    for start in range(0, total - clip_len + 1, stride):
        clips.append(frames[start:start + clip_len])
    if not clips:
        pad = np.repeat(frames[-1:], clip_len, axis=0)
        clips.append(pad)
    return np.stack(clips, axis=0)


def _prepare_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    tensor = tensor / 255.0
    return tensor


def _load_backbone(checkpoint: Optional[Path], device: torch.device):
    internvideo = _import_internvideo_module()
    ckpt_path = str(checkpoint) if checkpoint else os.environ.get("INTERNVIDEO_CKPT")
    if not ckpt_path:
        raise RuntimeError("InternVideo 체크포인트 경로를 --checkpoint 또는 환경변수 INTERNVIDEO_CKPT로 지정해 주세요.")
    model = internvideo.load_model(ckpt_path)
    model.to(device)
    model.eval()
    return model


def _extract_internvideo_features(
    model: torch.nn.Module,
    clips: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for clip in clips:
        tensor = _prepare_tensor(clip, device=device)
        tensor = tensor.unsqueeze(0)  # [1, T, C, H, W]
        with torch.no_grad():
            feats = model(tensor)
        feat_np = feats.detach().cpu().to(torch.float32).numpy()
        outputs.append(feat_np.squeeze(0))
    return np.stack(outputs, axis=0)


def _save_feature_array(path: Path, features: np.ndarray, fp16: bool) -> None:
    array = features.astype(np.float16 if fp16 else np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def extract_single(config: ExtractionConfig, model: torch.nn.Module, video_path: Path) -> None:
    try:
        frames, fps_original = _load_video(video_path)
    except Exception as exc:
        print(f"[extract] failed to read {video_path.name}: {exc}", file=sys.stderr)
        return

    sampled = _sample_frames(frames, fps_original=fps_original, fps_target=config.fps)
    clips = _split_into_clips(sampled, config.clip_len, config.stride)
    feats = _extract_internvideo_features(model, clips, config.device)

    out_path = config.out_dir / f"{video_path.stem}.npy"
    if out_path.exists() and not config.overwrite:
        print(f"[extract] skipping existing file: {out_path}")
        return

    _save_feature_array(out_path, feats, config.fp16)
    print(f"[extract] wrote {out_path.name} -> clips={feats.shape[0]}, dim={feats.shape[-1] if feats.ndim >= 2 else 0}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVideo 기반 비디오 피처 추출기")
    parser.add_argument("--src_dir", type=Path, required=True, help="영상이 위치한 디렉터리")
    parser.add_argument("--out_dir", type=Path, required=True, help="피처 출력 디렉터리")
    parser.add_argument("--backbone", default="internvideo_l", help="InternVideo 백본 이름")
    parser.add_argument("--checkpoint", type=Path, help="모델 체크포인트 경로")
    parser.add_argument("--fps", type=float, default=2.0, help="타깃 프레임 샘플링 속도")
    parser.add_argument("--clip_len", type=int, default=16, help="클립당 프레임 수")
    parser.add_argument("--stride", type=int, default=8, help="클립 추출 보폭")
    parser.add_argument("--fp16", action="store_true", help=" FP16으로 저장")
    parser.add_argument("--pattern", help="글롭 패턴 (예: '*.mp4')")
    parser.add_argument("--overwrite", action="store_true", help="기존 결과를 덮어쓰기")
    parser.add_argument("--limit", type=int, help="처리할 최대 영상 수 (테스트용)")
    parser.add_argument("--device", default="cuda", help="torch 디바이스 (예: cuda:0 또는 cpu)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    config = ExtractionConfig(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        backbone=args.backbone,
        fps=args.fps,
        clip_len=args.clip_len,
        stride=args.stride,
        fp16=args.fp16,
        pattern=args.pattern,
        overwrite=args.overwrite,
        limit=args.limit,
        resize=0,
        checkpoint=args.checkpoint,
        device=device,
    )

    if not config.src_dir.exists():
        raise SystemExit(f"source directory does not exist: {config.src_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(_iter_videos(config.src_dir, pattern=config.pattern, limit=config.limit))
    if not videos:
        raise SystemExit("no video files found to process")

    model = _load_backbone(config.backbone, config.checkpoint, config.device)

    print(f"[extract] processing {len(videos)} videos from {config.src_dir}")
    for idx, video_path in enumerate(videos, start=1):
        print(f"[extract] ({idx}/{len(videos)}) {video_path.name}")
        extract_single(config, model, video_path)


if __name__ == "__main__":  # pragma: no cover
    main()
