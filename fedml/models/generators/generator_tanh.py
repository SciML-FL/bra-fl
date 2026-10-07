"""A module to handle the model architecture for Generator part of GAN."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from fedml.models import ClassificationBaseModel


class GeneratorTanh(ClassificationBaseModel):
    """Generative Adversarial Network model's generator part."""

    def __init__(self, label_range, latent_dim, output_shape):
        super().__init__(num_classes=(label_range[1]-label_range[0]))
        self.output_shape = output_shape
        self.output_size = output_shape[1]  # assuming square output
        
        self.generator_layers = nn.Sequential(
            nn.Linear(latent_dim + self.num_classes, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, np.prod(output_shape)),
            nn.Tanh(),
        )

    def forward(self, input_noises: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        # emb_label = self.embedding_layers(input_labels)
        # input_x = torch.cat([input_noises, emb_label], dim = 1)
        x = torch.cat([input_noises, input_labels], dim=1)
        output_x = self.generator_layers(x)
        return output_x.view(output_x.size(0), *self.output_shape)
