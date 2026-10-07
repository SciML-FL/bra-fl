"""Implementation of a simple multilayer perceptron network."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fedml.models import BaseModel


class DNNRegressor(BaseModel):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
