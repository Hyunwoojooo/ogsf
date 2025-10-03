"""Automatic mixed precision helpers for fp16 training."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

__all__ = ["Scaler"]


@dataclass
class Scaler:
    """Thin wrapper around ``torch.cuda.amp.GradScaler`` with CPU fallback."""

    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = torch.cuda.is_available()
        self._scaler = None
        if self.enabled and torch.cuda.is_available():  # pragma: no cover - GPU specific
            self._scaler = torch.cuda.amp.GradScaler()

    def autocast(self):
        if self.enabled and torch.cuda.is_available():  # pragma: no cover - GPU specific
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self._scaler is not None:  # pragma: no branch
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer) -> None:
        if self._scaler is not None:  # pragma: no branch
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        if self._scaler is not None:  # pragma: no branch
            self._scaler.update()

    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state)
