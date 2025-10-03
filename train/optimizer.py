"""Optimizers and schedulers including 8-bit variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import optim

__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "create_optimizer",
    "create_scheduler",
]


@dataclass
class OptimizerConfig:
    """Configuration for optimizer construction."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    use_8bit: bool = False


@dataclass
class SchedulerConfig:
    """Configuration for learning-rate scheduler construction."""

    name: str = "cosine"
    warmup_steps: int = 0
    total_steps: int = 1000
    min_lr: float = 0.0


def _maybe_import_bitsandbytes():
    try:  # pragma: no cover - optional dependency
        import bitsandbytes as bnb  # type: ignore

    except ModuleNotFoundError:  # pragma: no cover
        return None
    return bnb


def create_optimizer(parameters: Iterable[torch.nn.Parameter], config: OptimizerConfig) -> optim.Optimizer:
    """Instantiate an optimizer honoring optional 8-bit mode."""

    params = list(parameters)
    if not params:
        raise ValueError("Parameter iterable is empty")

    if config.use_8bit:
        bnb = _maybe_import_bitsandbytes()
        if bnb is not None:
            return bnb.optim.AdamW8bit(params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)  # type: ignore[attr-defined]

    name = config.name.lower()
    if name == "adamw":
        return optim.AdamW(params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=config.lr, momentum=config.betas[0], weight_decay=config.weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.name}")


def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def create_scheduler(optimizer: optim.Optimizer, config: SchedulerConfig):
    """Return a simple scheduler callable supporting cosine or constant modes."""

    name = config.name.lower()

    if name == "constant":
        def constant_schedule(step: int) -> float:
            return _warmup_factor(step, config.warmup_steps)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_schedule)

    if name == "cosine":
        def schedule(step: int) -> float:
            warmup = _warmup_factor(step, config.warmup_steps)
            progress = min(1.0, max(0.0, (step + 1 - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            base = cosine * (1.0 - config.min_lr) + config.min_lr
            return base * warmup

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)

    raise ValueError(f"Unsupported scheduler: {config.name}")
