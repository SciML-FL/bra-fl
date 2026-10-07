"""Helper functions for building strategy configuration callables."""

import json
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from fedml.utils.typing import Parameters, Scalar
from fedml.models import load_model
from fedml.training import evaluate, get_criterion
from fedml.training.lr_scheduler import get_lr_schedule

import os
# num_workers = min(os.cpu_count(), 8)  # cap at 8, use all cores if fewer
num_workers = 0

def get_fit_config_fn(
    total_rounds: int,
    local_epochs: int,
    scheduler_args: Dict,
    local_batchsize: int,
    learning_rate: float,
    lr_scheduler: str,
    lr_warmup_steps: int,
    initial_lr: float,
    optimizer_str: str,
    criterion_str: str,
    perform_evals: bool,
    optim_kwargs: Dict,
    base_seed: int = 0,
) -> Callable:
    """Return a fit config function that provides per-round training configuration."""
    lr_schedule: List[float] = get_lr_schedule(
        total_rounds=total_rounds,
        method=lr_scheduler,
        warmup_steps=lr_warmup_steps,
        initial_lr=initial_lr,
        target_lr=learning_rate,
        scheduler_args=scheduler_args,
    )

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": str(server_round),
            "total_rounds": str(total_rounds),
            "epochs": str(local_epochs),
            "batch_size": str(local_batchsize),
            "learning_rate": str(lr_schedule[server_round - 1]),
            "optimizer": optimizer_str,
            "criterion": criterion_str,
            "perform_evals": perform_evals,
            "optim_kwargs": json.dumps(optim_kwargs),
            "base_seed": str(base_seed),
        }

    return fit_config


def get_evaluate_config_fn(
    total_rounds: int,
    evaluate_bs: int,
    criterion_str: str,
) -> Callable:
    """Return an evaluate config function that provides per-round eval configuration."""
    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": str(server_round),
            "total_rounds": str(total_rounds),
            "batch_size": str(evaluate_bs),
            "criterion": criterion_str,
        }

    return evaluate_config


def get_evaluate_fn(
    testset: Dataset,
    model,
    model_as_fn: bool,
    # model_configs: dict,
    device: str,
    criterion_str: str,
) -> Callable[[int, Parameters, Dict[str, Scalar]], Optional[Tuple[float, Dict]]]:
    """Return a centralized server-side evaluation function."""
    # model = load_model(model_configs=model_configs)
    if model_as_fn: model = model()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=num_workers)
    criterion = get_criterion(criterion_str=criterion_str)

    def evaluate_fn(
        server_round: int,
        weights: Parameters,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict]]:
        model.set_weights(weights)
        model.to(device)
        loss, accuracy, _ = evaluate(model, testloader, device=device, criterion=criterion)
        return loss, {"accuracy": accuracy}

    return evaluate_fn