"""Pure training loop routines for a single epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch

from .amp import Scaler

__all__ = ["StepOutput", "train_epoch"]


@dataclass
class StepOutput:
    """Container returned by ``step_fn`` holding loss and metrics."""

    loss: torch.Tensor
    metrics: Mapping[str, float]


def train_epoch(
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    step_fn: Callable[[Mapping[str, torch.Tensor]], StepOutput],
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Optional[Scaler] = None,
    callbacks: Optional[Iterable[Callable[[int, Mapping[str, float]], None]]] = None,
) -> List[MutableMapping[str, float]]:
    """Execute a stateless training loop returning per-step metrics."""

    history: List[MutableMapping[str, float]] = []
    callbacks = list(callbacks or [])

    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad()

        if scaler is not None:
            with scaler.autocast():
                output = step_fn(batch)
            scaled_loss = scaler.scale(output.loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = step_fn(batch)
            output.loss.backward()
            optimizer.step()

        metrics = dict(output.metrics)
        metrics.setdefault("loss", float(output.loss.detach().item()))
        metrics["step"] = step
        history.append(metrics)

        for callback in callbacks:
            callback(step, metrics)

    return history
