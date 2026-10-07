"""Factory function for optimizers."""

from typing import Dict
import torch.nn as nn
from torch.optim import Adam, Optimizer, SGD


def get_optimizer(
    optimizer_str: str,
    local_model: nn.Module,
    learning_rate: float,
    **kwargs: Dict,
) -> Optimizer:
    """Return the requested optimizer for a given model."""
    if optimizer_str == "SGD":
        return SGD(local_model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_str == "ADAM":
        return Adam(local_model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_str}' requested.")