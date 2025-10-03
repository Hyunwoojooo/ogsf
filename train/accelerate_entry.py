"""Accelerate launcher entry points for distributed training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import torch

from ..common import config as config_utils
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
            "labels": scores,
            "bounds": bounds,
            "mask": mask,
        }


def _run_epoch(
    model: torch.nn.Module,
    data_iter: Iterable[Mapping[str, torch.Tensor]],
    opt: torch.optim.Optimizer,
    scaler: amp.Scaler | None,
    device: torch.device,
) -> list[MutableMapping[str, float]]:
    def step_fn(batch: Mapping[str, torch.Tensor]) -> loop.StepOutput:
        video = batch["video_feat"].to(device)
        text = batch["text_feat"].to(device)
        targets = {
            "scores": batch["labels"].to(device),
            "bounds": batch["bounds"].to(device),
        }

        predictions = model(video, text)
        preds = {"scores": predictions["scores"], "bounds": predictions["bounds"]}
        loss, components = model.compute_loss(preds, targets)
        metrics = {key: float(val.detach().item()) for key, val in components.items()}
        return loop.StepOutput(loss=loss, metrics=metrics)

    return loop.train_epoch(data_iter, step_fn, opt, scaler=scaler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accelerate entry for GroundNLQ experiments")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--paths", help="Optional paths YAML to set environment variables")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile where available")
    parser.add_argument("--steps", type=int, default=50, help="Synthetic steps per epoch when data unavailable")
    parser.add_argument("--text-tokens", type=int, default=4, help="Synthetic text token count")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _load_paths(args.paths)

    exp_config = config_utils.load_config(args.config).to_dict()
    model_cfg = exp_config.get("model", {})
    train_cfg = exp_config.get("train", {})

    model = _build_model(model_cfg)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[arg-type]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    time_steps = int(train_cfg.get("sequence_length", 16))
    epochs = int(train_cfg.get("epochs", 1))

    for epoch in range(epochs):
        data_iter = _synthetic_batches(
            batch_size=batch_size,
            time_steps=time_steps,
            text_tokens=args.text_tokens,
            d_v=int(model_cfg.get("d_v", 1024)),
            d_t=int(model_cfg.get("d_t", 768)),
            steps=args.steps,
        )
        history = _run_epoch(model, data_iter, opt, scaler if scaler.enabled else None, device)
        avg_loss = sum(entry["loss"] for entry in history) / max(1, len(history))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
