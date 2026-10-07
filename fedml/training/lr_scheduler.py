"""Learning rate scheduler factory and schedule builder."""

from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, SequentialLR


def get_lr_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    method: Optional[str] = "STATIC",
    warmup_steps: int = 0,
    initial_lr: Optional[float] = None,
    target_lr: Optional[float] = None,
    **kwargs: Optional[Dict],
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build and return a learning rate scheduler."""
    schedulers_list = []
    milestones = []

    if warmup_steps > 0:
        factor = (target_lr / initial_lr) ** (1 / warmup_steps)
        schedulers_list.append(ExponentialLR(optimizer=optimizer, gamma=factor))
        milestones.append(warmup_steps)

    if method == "STATIC":
        schedulers_list.append(
            MultiStepLR(
                optimizer=optimizer, milestones=[total_epochs + 1]
            )
        )
    elif method == "3-STEP":
        schedulers_list.append(
            MultiStepLR(
                optimizer=optimizer,
                milestones=[int(0.25 * total_epochs), int(0.5 * total_epochs), int(0.75 * total_epochs)],
                gamma=0.1,
            )
        )
    elif method == "EXPONENTIAL":
        schedulers_list.append(
            ExponentialLR(optimizer=optimizer,  **kwargs)
        )
    elif method == "CUSTOM":
        schedulers_list.append(
            MultiStepLR(optimizer=optimizer, **kwargs)
        )
    else:
        raise ValueError(f"Scheduler method '{method}' is not supported.")

    return SequentialLR(optimizer=optimizer, schedulers=schedulers_list, milestones=milestones)


def get_lr_schedule(
    total_rounds: int,
    method: Optional[str] = "STATIC",
    warmup_steps: int = 0,
    initial_lr: Optional[float] = None,
    target_lr: Optional[float] = None,
    scheduler_args: Optional[Dict] = None,
) -> List[float]:
    """Build and return a per-round learning rate schedule as a list."""
    schedule = []

    if warmup_steps > 0:
        factor = (target_lr / initial_lr) ** (1 / warmup_steps)
        schedule.extend([initial_lr * (factor ** step) for step in range(warmup_steps)])

    if method == "STATIC":
        schedule.extend([target_lr] * (total_rounds - warmup_steps))

    elif method in ("3-STEP", "CUSTOM"):
        if method == "3-STEP":
            decay_steps = [
                int(0.25 * total_rounds),
                int(0.50 * total_rounds),
                int(0.75 * total_rounds),
            ]
        else:
            decay_steps = [int(m * total_rounds) for m in scheduler_args["milestones"]]

        current_lr = target_lr
        for current_round in range(total_rounds):
            if current_round in decay_steps:
                current_lr *= scheduler_args["gamma"]
            if current_round >= warmup_steps:
                schedule.append(current_lr)

    elif method == "EXPONENTIAL":
        current_lr = initial_lr
        for current_round in range(total_rounds):
            if current_round >= warmup_steps:
                schedule.append(current_lr)
                current_lr *= scheduler_args["gamma"]

    return schedule