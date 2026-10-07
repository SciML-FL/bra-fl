"""A module to handle the model architecture for Generator part of GAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fedml.models import ClassificationBaseModel


class GeneratorDCGAN(ClassificationBaseModel):
    """Generative Adversarial Network model's generator part.

    Reference Link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    
    """
    def __init__(
            self, 
            label_range, 
            latent_dim, 
            output_shape, 
            ngf=64,
            dropout_rate: float = 0.4,
        ):
        super().__init__(num_classes=(label_range[1]-label_range[0]))

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.output_size = output_shape[1]  # assuming square output
    
        self.early_layers = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim + self.num_classes, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            # Output is then interpolated to (num_classes x num_classes) in forward()
        )

        self.late_layers = nn.Sequential(
            # NxN -> 2N x 2N
            nn.ConvTranspose2d(ngf * 4 , ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.ConvTranspose2d(ngf * 4 + 1, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),  # slightly less dropout in later layers

            # 2N x 2N -> 4N x 4N
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),

            # Final channel projection (spatial size adjusted via interpolation in forward)
            nn.Conv2d(ngf, output_shape[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_noises: Tensor, input_labels: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_noises : [batch_size, latent_dim]       — sampled from noise distribution
            input_labels : [batch_size, num_classes]      — soft-maxed random vectors
                           (NOT integer class indices; the label is encoded implicitly
                            via argmax — the generator never sees it directly)
        Returns:
            Generated images of shape [batch_size, output_channels, output_size, output_size]
        """

        # 1. Concatenate noise + soft label vector, reshape to [B, C, 1, 1]
        x = torch.cat([input_noises, input_labels], dim=1)
        x = x[:, :, None, None]

        # 2. Early upsampling (to ~8x8)
        x = self.early_layers(x)

        # 3. Interpolate to (num_classes x num_classes) for matrix injection
        x = F.interpolate(x, size=(self.num_classes, self.num_classes), mode='bilinear', align_corners=False)

        # 4. Build and concatenate the NxN hot conditioning matrix
        # cond_matrix = self._build_conditioning_matrix(input_labels)
        # x = torch.cat([x, cond_matrix], dim=1)   # channel dim: ngf*4 + 1

        # 5. Late upsampling
        x = self.late_layers(x)

        # 6. Resize to the exact target output size
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return x
