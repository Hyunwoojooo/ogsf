"""Tests for training loop scaffolding modules."""

from __future__ import annotations

import torch
import pytest

from groundnlq.em.train import accelerate_entry, amp, loop, optimizer


def _dummy_dataloader(batch_size: int, steps: int):
    for _ in range(steps):
        yield {
            "video_feat": torch.randn(batch_size, 4),
            "text_feat": torch.randn(batch_size, 4),
            "labels": torch.randn(batch_size),
            "mask": torch.ones(batch_size),
        }


def test_train_epoch_runs_with_callbacks():
    model = torch.nn.Linear(4, 1)
    opt = optimizer.create_optimizer(model.parameters(), optimizer.OptimizerConfig(lr=0.1))

    records = []

    def step_fn(batch):
        preds = model(batch["video_feat"]).squeeze(-1)
        diff = (preds - batch["labels"]) * batch["mask"]
        loss = (diff ** 2).mean()
        return loop.StepOutput(loss=loss, metrics={"batch_loss": float(loss.detach())})

    callbacks = [lambda step, metrics: records.append((step, metrics["batch_loss"]))]
    history = loop.train_epoch(_dummy_dataloader(3, 2), step_fn, opt, callbacks=callbacks)

    assert len(history) == 2
    assert len(records) == 2
    assert all("batch_loss" in item for _, item in zip(range(2), history))


def test_optimizer_and_scheduler_step():
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = optimizer.create_optimizer([param], optimizer.OptimizerConfig(lr=0.1))
    sched = optimizer.create_scheduler(opt, optimizer.SchedulerConfig(name="constant", warmup_steps=1, total_steps=10))

    loss = param ** 2
    loss.backward()
    opt.step()
    sched.step()
    opt.zero_grad()

    assert param.item() != 1.0


def test_amp_scaler_cpu_fallback():
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=0.1)
    scaler = amp.Scaler(enabled=False)

    with scaler.autocast():
        loss = param ** 2
    scaled = scaler.scale(loss)
    scaled.backward()
    scaler.step(opt)
    scaler.update()

    assert param.grad is not None


def test_accelerate_entry_dummy_accelerator():
    acc = accelerate_entry.create_accelerator(accelerate_entry.AccelerateConfig())
    model = torch.nn.Linear(2, 2)
    prepared = acc.prepare(model)
    assert prepared is model
