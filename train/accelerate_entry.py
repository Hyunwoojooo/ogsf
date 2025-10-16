"""Accelerate launcher entry points for distributed training."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch

from ..common import config as config_utils
from ..features import video_loader
from ..io import webdataset_reader
from ..models import MVPConfig, MVPModel, OGSFConfig, OGSFModel
from ..train import amp, loop, optimizer

__all__ = ["AccelerateConfig", "create_accelerator", "main"]


@dataclass
class AccelerateConfig:
    """Configuration parameters for accelerator setup."""

    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1


class _DummyAccelerator:
    """Fallback accelerator used when ``accelerate`` is unavailable."""

    def __init__(self, config: AccelerateConfig) -> None:
        self.config = config

    def prepare(self, *objects: Any):
        if len(objects) == 1:
            return objects[0]
        return objects

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def wait_for_everyone(self) -> None:  # pragma: no cover - trivial
        return None


def create_accelerator(config: AccelerateConfig):
    """Create an ``Accelerator`` if available, otherwise a dummy wrapper."""

    try:  # pragma: no cover - accelerate optional
        from accelerate import Accelerator
    except ModuleNotFoundError:  # pragma: no cover
        return _DummyAccelerator(config)

    return Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )


def _load_paths(paths_file: str | None) -> Dict[str, str]:
    if not paths_file:
        return {}
    import yaml  # lazy import to avoid hard dependency during tests

    data = yaml.safe_load(Path(paths_file).read_text(encoding="utf-8")) or {}
    env_pairs = {str(key): str(value) for key, value in data.items()}
    for key, value in env_pairs.items():
        os.environ.setdefault(key, value)
    return env_pairs


def _build_model(model_cfg: Mapping[str, Any]) -> torch.nn.Module:
    name = model_cfg.get("name", "mvp").lower()
    if name == "mvp":
        cfg = MVPConfig(
            d_v=int(model_cfg.get("d_v", 1024)),
            d_t=int(model_cfg.get("d_t", 768)),
            hidden=int(model_cfg.get("hidden", 512)),
            heads=int(model_cfg.get("heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_flash_attn=bool(model_cfg.get("use_flash_attn", False)),
        )
        return MVPModel(cfg)

    if name == "ogsf":
        cfg = OGSFConfig(
            d_v=int(model_cfg.get("d_v", 1024)),
            d_t=int(model_cfg.get("d_t", 768)),
            hidden=int(model_cfg.get("hidden", 512)),
            heads=int(model_cfg.get("heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_flash_attn=bool(model_cfg.get("use_flash_attn", False)),
            use_objects=bool(model_cfg.get("use_objects", True)),
            use_multiscale=bool(model_cfg.get("use_multiscale", True)),
            use_asl=bool(model_cfg.get("use_asl", True)),
            object_dim=int(model_cfg.get("object_dim", 0)),
            fpn_levels=int(model_cfg.get("fpn_levels", 3)),
        )
        return OGSFModel(cfg)

    raise ValueError(f"Unsupported model name: {name}")


def _expand_path(value: str | None) -> Optional[str]:
    if not value:
        return None
    return os.path.expandvars(os.path.expanduser(value))


def _prepare_sample_batch(
    batch: List[Mapping[str, Any]],
    *,
    sequence_length: int,
    text_tokens: int,
) -> Dict[str, Any]:
    """Convert a list of raw samples into stacked tensors."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if text_tokens <= 0:
        raise ValueError("text_tokens must be positive")

    video_tensors: List[torch.Tensor] = []
    text_tensors: List[torch.Tensor] = []
    score_targets: List[torch.Tensor] = []
    bound_targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] = []
    object_features: List[Optional[torch.Tensor]] = []
    object_masks: List[Optional[torch.Tensor]] = []
    object_feature_dim: Optional[int] = None
    has_objects = False

    denom = max(sequence_length - 1, 1)

    for sample in batch:
        video_raw = np.asarray(sample["video_feat"], dtype=np.float32)
        if video_raw.ndim == 1:
            video_raw = video_raw.reshape(1, -1)
        video_seq, mask = video_loader.prepare_feature_sequence(
            video_raw,
            target_length=sequence_length,
            normalize=True,
        )
        video_tensors.append(torch.from_numpy(video_seq))
        masks.append(torch.from_numpy(mask))

        text_raw = np.asarray(sample["text_feat"], dtype=np.float32)
        if text_raw.ndim == 1:
            text_raw = text_raw.reshape(1, -1)
        text_seq, _ = video_loader.prepare_feature_sequence(
            text_raw,
            target_length=text_tokens,
            normalize=False,
        )
        text_tensors.append(torch.from_numpy(text_seq))

        labels = sample.get("labels") or [{"start": 0.0, "end": 0.0}]
        start_sec = float(labels[0]["start"])
        end_sec = float(labels[0]["end"])
        if end_sec < start_sec:
            end_sec = start_sec
        duration = max(max(label["end"] for label in labels), 1e-3)
        start_frac = max(0.0, min(1.0, start_sec / duration))
        end_frac = max(start_frac, min(1.0, end_sec / duration))
        start_idx = int(round(start_frac * denom))
        end_idx = int(round(end_frac * denom))

        scores = torch.zeros(sequence_length, dtype=torch.float32)
        scores[start_idx : end_idx + 1] = 1.0
        bounds = torch.zeros((2, sequence_length), dtype=torch.float32)
        bounds[0].fill_(start_frac * 2.0 - 1.0)
        bounds[1].fill_(end_frac * 2.0 - 1.0)

        obj_raw = sample.get("object_feat")
        mask_raw = sample.get("object_mask")
        if obj_raw is not None:
            obj_np = np.asarray(obj_raw, dtype=np.float32)
            if obj_np.ndim == 1:
                obj_np = obj_np.reshape(-1, 1)
            if obj_np.shape[0] != sequence_length:
                obj_np, _ = video_loader.prepare_feature_sequence(
                    obj_np,
                    target_length=sequence_length,
                    normalize=False,
                )
            if object_feature_dim is None:
                object_feature_dim = obj_np.shape[1]
            obj_tensor = torch.from_numpy(obj_np)

            if mask_raw is not None:
                mask_np = np.asarray(mask_raw, dtype=np.float32).reshape(-1, 1)
                if mask_np.shape[0] != sequence_length:
                    mask_np, _ = video_loader.prepare_feature_sequence(
                        mask_np,
                        target_length=sequence_length,
                        normalize=False,
                    )
                mask_vector = torch.from_numpy(mask_np.reshape(-1))
            else:
                mask_vector = (obj_tensor[:, 0] > 0).to(torch.float32)

            object_features.append(obj_tensor)
            object_masks.append(mask_vector)
            has_objects = True
        else:
            object_features.append(None)
            object_masks.append(None)

        score_targets.append(scores)
        bound_targets.append(bounds)
        metadata.append(
            {
                "video_id": sample.get("video_id"),
                "qid": sample.get("qid"),
                "start": start_sec,
                "end": end_sec,
                "duration": duration,
            }
        )

    batch_dict: Dict[str, Any] = {
        "video_feat": torch.stack(video_tensors),
        "text_feat": torch.stack(text_tensors),
        "scores": torch.stack(score_targets),
        "bounds": torch.stack(bound_targets),
        "mask": torch.stack(masks),
        "meta": metadata,
    }

    if has_objects:
        feature_dim = object_feature_dim or (object_features[0].shape[1] if object_features[0] is not None else 1)
        filled_features: List[torch.Tensor] = []
        filled_masks: List[torch.Tensor] = []
        for feat, msk in zip(object_features, object_masks):
            if feat is None:
                filled_features.append(torch.zeros(sequence_length, feature_dim, dtype=torch.float32))
            else:
                filled_features.append(feat.to(torch.float32))
            if msk is None:
                filled_masks.append(torch.zeros(sequence_length, dtype=torch.float32))
            else:
                filled_masks.append(msk.to(torch.float32))

        batch_dict["object_feat"] = torch.stack(filled_features)
        batch_dict["object_mask"] = torch.stack(filled_masks)

    return batch_dict


def _dataset_iterator(
    pattern: str,
    *,
    batch_size: int,
    shuffle_buf: int,
    sequence_length: int,
    text_tokens: int,
) -> Iterator[Dict[str, Any]]:
    pipeline = webdataset_reader.create_pipeline(
        pattern,
        batch_size=batch_size,
        shuffle_buf=shuffle_buf,
        pin_memory=False,
    )
    for batch in pipeline:
        yield _prepare_sample_batch(
            batch,
            sequence_length=sequence_length,
            text_tokens=text_tokens,
        )


def _synthetic_batches(
    *,
    batch_size: int,
    time_steps: int,
    text_tokens: int,
    d_v: int,
    d_t: int,
    steps: int,
) -> Iterable[Mapping[str, torch.Tensor]]:
    for _ in range(steps):
        video = torch.randn(batch_size, time_steps, d_v)
        text = torch.randn(batch_size, text_tokens, d_t)
        scores = torch.rand(batch_size, time_steps)
        bounds = torch.rand(batch_size, 2, time_steps)
        mask = torch.ones(batch_size, time_steps)
        yield {
            "video_feat": video,
            "text_feat": text,
            "scores": scores,
            "bounds": bounds,
            "mask": mask,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accelerate entry for GroundNLQ experiments")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--paths", help="Optional paths YAML to set environment variables")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile where available")
    parser.add_argument("--steps", type=int, default=50, help="Synthetic steps per epoch when data unavailable")
    parser.add_argument("--text-tokens", type=int, default=4, help="Synthetic text token count")
    parser.add_argument(
        "--experiment",
        help="Optional experiment name; defaults to config file stem when omitted",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device (e.g. cuda, cuda:0, cpu). 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abort when CUDA is unavailable instead of silently falling back to CPU.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSONL file to record per-step training metrics",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Write every N steps to --log-file (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _load_paths(args.paths)

    exp_config = config_utils.load_config(args.config).to_dict()
    model_cfg = exp_config.get("model", {})
    solver_cfg = exp_config.get("solver") or {}
    train_cfg = solver_cfg or exp_config.get("train", {})
    data_cfg: Mapping[str, Any] = exp_config.get("data", {})

    model = _build_model(model_cfg)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[arg-type]

    def _resolve_device(request: str) -> torch.device:
        if request == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        device_candidate = torch.device(request)
        if device_candidate.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested via --device={request} but no CUDA devices are available.")
        return device_candidate

    device = _resolve_device(args.device)
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("CUDA is required (--require-gpu) but no GPU devices were detected.")

    print(f"[accelerate_entry] using device: {device}")
    model.to(device)

    opt_cfg = optimizer.OptimizerConfig(
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        use_8bit=bool(train_cfg.get("optimizer_8bit", False)),
    )
    opt = optimizer.create_optimizer(model.parameters(), opt_cfg)

    scaler = amp.Scaler(enabled=bool(train_cfg.get("fp16", False)))

    batch_size = int(train_cfg.get("batch_size", 2))
    sequence_length = int(train_cfg.get("sequence_length", data_cfg.get("sequence_length", 16)))
    text_tokens = int(data_cfg.get("text_tokens", args.text_tokens))
    epochs = int(train_cfg.get("epochs", 1))
    shuffle_buf = int(data_cfg.get("shuffle_buf", 0))

    train_glob = _expand_path(data_cfg.get("shard_glob") or data_cfg.get("train_glob"))
    def _make_data_iter(pattern: Optional[str], *, shuffle: bool) -> Optional[Iterator[Dict[str, Any]]]:
        if not pattern:
            return None
        return _dataset_iterator(
            pattern,
            batch_size=batch_size,
            shuffle_buf=shuffle_buf if shuffle else 0,
            sequence_length=sequence_length,
            text_tokens=text_tokens,
        )

    log_handle_fp = None
    log_writer = None
    log_every = max(1, int(args.log_interval))
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle_fp = log_path.open("w", encoding="utf-8")

        def _write_log(record: Mapping[str, Any]) -> None:
            payload = json.dumps(record, ensure_ascii=False)
            assert log_handle_fp is not None
            log_handle_fp.write(payload + "\n")
            log_handle_fp.flush()

        log_writer = _write_log

    for epoch in range(epochs):
        train_iter = _make_data_iter(train_glob, shuffle=True)
        if train_iter is None:
            data_iter = _synthetic_batches(
                batch_size=batch_size,
                time_steps=sequence_length,
                text_tokens=text_tokens,
                d_v=int(model_cfg.get("d_v", 1024)),
                d_t=int(model_cfg.get("d_t", 768)),
                steps=args.steps,
            )
        else:
            data_iter = train_iter

        def step_fn(batch: Mapping[str, torch.Tensor]) -> loop.StepOutput:
            video = batch["video_feat"].to(device)
            text = batch["text_feat"].to(device)
            mask = batch["mask"].to(device)

            object_tensor: Optional[torch.Tensor] = None
            object_mask_tensor: Optional[torch.Tensor] = None
            if "object_feat" in batch:
                obj = batch["object_feat"].to(device)
                object_tensor = obj
                obj_mask = batch.get("object_mask")
                if obj_mask is not None:
                    object_mask_tensor = obj_mask.to(device)
                    object_tensor = object_tensor * object_mask_tensor.unsqueeze(-1)
            elif hasattr(model, "config") and getattr(model.config, "use_objects", False):
                object_dim = int(getattr(model.config, "object_dim", 0))
                if object_dim > 0:
                    object_tensor = torch.zeros(video.size(0), video.size(1), object_dim, device=device, dtype=video.dtype)
                    object_mask_tensor = torch.zeros(video.size(0), video.size(1), device=device, dtype=video.dtype)

            use_objects = (
                object_tensor is not None
                and hasattr(model, "config")
                and getattr(model.config, "use_objects", False)
                and hasattr(model, "object_gate")
            )

            if use_objects:
                predictions = model(video, text, object_features=object_tensor)
            else:
                predictions = model(video, text)
            mask_scores = mask.unsqueeze(1)
            masked_preds = {
                "scores": predictions["scores"] * mask_scores,
                "bounds": predictions["bounds"] * mask_scores,
            }
            targets = {
                "scores": batch["scores"].to(device) * mask,
                "bounds": batch["bounds"].to(device) * mask_scores,
            }

            loss, components = model.compute_loss(masked_preds, targets)
            metrics = {key: float(val.detach().item()) for key, val in components.items()}
            return loop.StepOutput(loss=loss, metrics=metrics)

        history = loop.train_epoch(
            data_iter,
            step_fn,
            opt,
            scaler=scaler if scaler.enabled else None,
        )
        avg_loss = sum(entry["loss"] for entry in history) / max(1, len(history))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        if log_writer is not None:
            timestamp = time.time()
            for entry in history:
                if entry["step"] % log_every != 0:
                    continue
                record = dict(entry)
                record["epoch"] = epoch + 1
                record["time"] = timestamp
                log_writer(record)

    output_root = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
    run_name = args.experiment or Path(args.config).stem
    ckpt_dir = output_root / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model_last.pth"
    model_to_save = model
    if hasattr(model_to_save, "_orig_mod"):  # torch.compile wrapper
        model_to_save = model_to_save._orig_mod  # type: ignore[attr-defined]

    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.enabled else {},
            "config": exp_config,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    if log_handle_fp is not None:
        log_handle_fp.close()

if __name__ == "__main__":  # pragma: no cover
    main()
