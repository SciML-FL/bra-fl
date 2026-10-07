"""A module to handle the model architecture for Generator part of GAN."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from fedml.models import BaseModel


class GeneratorRegressor(BaseModel):
    """Generative Adversarial Network model's generator part."""

    def __init__(self, latent_dim, output_shape, condition_dim=16, dropout_rate=0.4):
        super().__init__()
        # Encode the continuous target into a conditioning vector
        # (analogous to the soft-maxed label vector in the paper)
        self.condition_encoder = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim, condition_dim),
        )
        # Main generator body
        self.gen_layers = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout(dropout_rate),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Dropout(dropout_rate * 0.5),

            nn.Linear(128, np.prod(output_shape)),
            # nn.Tanh(),  # assuming normalised input features in [-1, 1]
        )

        self.output_shape= output_shape

    def forward(self, noise: Tensor, target_values: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        # target_values: [B, 1] continuous UPDRS scores (normalised)
        cond = self.condition_encoder(target_values)
        x = torch.cat([noise, cond], dim=1)
        output_x = self.gen_layers(x)
        return output_x.view(output_x.size(0), *self.output_shape)

