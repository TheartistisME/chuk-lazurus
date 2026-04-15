"""Config-driven optimizer loader (EWS-9).

Lazy-import-clean: all mlx/torch imports are function-scope.
"""

from __future__ import annotations

import os
from typing import Any


def linear_warmup_schedule(initial_lr, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return initial_lr * (step + 1) / warmup_steps
        return initial_lr

    return schedule


def piecewise_scheduler(schedulers, milestones):
    def schedule(step):
        index = 0
        for milestone in milestones:
            if step < milestone:
                break
            index += 1
        return schedulers[index](step - milestones[index - 1] if index > 0 else step)

    return schedule


def _resolve_framework(framework: str | None) -> str:
    if framework:
        return framework
    return os.environ.get("CHUK_BACKEND", "mlx").lower() or "mlx"


def load_optimizer(
    optimizer_config: dict, total_iterations: int, framework: str | None = None
) -> Any:
    optimizer_name = optimizer_config["name"]
    initial_lr = float(optimizer_config["initial_lr"])
    lr_schedule_type = optimizer_config["lr_schedule"]["type"]
    betas = [float(b) for b in optimizer_config["betas"]]
    eps = float(optimizer_config["eps"])
    weight_decay = float(optimizer_config["weight_decay"])
    decay_steps = total_iterations
    fw = _resolve_framework(framework)

    if fw == "mlx":
        import mlx.optimizers as optim

        if lr_schedule_type == "cosine_decay":
            warmup_steps = int(optimizer_config["lr_schedule"].get("warmup_steps", 0))
            minimum = float(optimizer_config["lr_schedule"].get("minimum", 0.0))
            cosine_schedule = optim.cosine_decay(initial_lr, decay_steps, minimum)
            if warmup_steps > 0:
                warmup_schedule = linear_warmup_schedule(initial_lr, warmup_steps)
                lr_schedule = piecewise_scheduler(
                    [warmup_schedule, cosine_schedule], [warmup_steps]
                )
            else:
                lr_schedule = cosine_schedule
        elif lr_schedule_type == "exponential_decay":
            decay_rate = float(optimizer_config["lr_schedule"]["decay_rate"])
            lr_schedule = optim.exponential_decay(initial_lr, decay_rate, decay_steps)
        else:
            raise ValueError(f"Unsupported learning rate schedule: {lr_schedule_type}")

        if optimizer_name != "AdamW":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optim.AdamW(
            learning_rate=lr_schedule, betas=betas, eps=eps, weight_decay=weight_decay
        )

    # torch path — returns a callable that takes params and produces (optimizer, scheduler).
    import torch.optim as torch_optim

    if optimizer_name != "AdamW":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _build(params):
        optimizer = torch_optim.AdamW(
            params,
            lr=initial_lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )
        scheduler: Any
        if lr_schedule_type == "cosine_decay":
            minimum = float(optimizer_config["lr_schedule"].get("minimum", 0.0))
            scheduler = torch_optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min=minimum
            )
        elif lr_schedule_type == "exponential_decay":
            decay_rate = float(optimizer_config["lr_schedule"]["decay_rate"])
            scheduler = torch_optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        else:
            raise ValueError(f"Unsupported learning rate schedule: {lr_schedule_type}")
        return optimizer, scheduler

    return _build
