"""Implementation of a simple multilayer perceptron network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from .model import ClassificationBaseModel


class Net2(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes, input_size, output_size) -> None:
        super(Net2, self).__init__()
        self.embedding_layers = nn.Embedding(num_classes, num_classes)
        self.generator_layers = nn.Sequential(
            nn.Linear(input_size+num_classes, 60),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(60, 120),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(120, 240),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(240, 480),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(480, 960),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(960, 1920),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1920, output_size),
            # nn.Tanh(),
        )
        self.num_classes = num_classes
        self.output_size= output_size

    def forward(self, input_z: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        emb_label = self.embedding_layers(input_labels)
        input_x = torch.cat([input_z, emb_label], dim = 1)
        output_x = self.generator_layers(input_x)
        return output_x.view(output_x.size(0), self.output_size)

    def get_weights(self):
        """Get model weights as a list of NumPy ndarrays."""
        weights: Tensor = nn.utils.parameters_to_vector(self.parameters(recurse=True))
        weights = weights.detach().clone()
        return weights

    def set_weights(self, weights, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: weights = weights.detach().clone()
        nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))

class Net3(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes, input_size, output_size) -> None:
        super(Net3, self).__init__()
        self.embedding_layers = nn.Embedding(num_classes, num_classes)
        self.generator_layers = nn.Sequential(
            nn.Linear(input_size+num_classes, 60),
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.Tanh(),
            nn.Linear(120, 240),
            nn.Tanh(),
            nn.Linear(240, 480),
            nn.Tanh(),
            nn.Linear(480, 960),
            nn.Tanh(),
            nn.Linear(960, 1920),
            nn.Tanh(),
            nn.Linear(1920, 1920),
            nn.Tanh(),
            nn.Linear(1920, 1920),
            nn.Tanh(),
            # nn.Linear(1920, 1920),
            # nn.Tanh(),
            # nn.Linear(1920, 1920),
            # nn.Tanh(),
            nn.Linear(1920, output_size),
            nn.Tanh(),
        )
        self.num_classes = num_classes
        self.output_size = output_size

    def forward(self, input_z: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        emb_label = self.embedding_layers(input_labels)
        input_x = torch.cat([input_z, emb_label], dim = 1)
        output_x = self.generator_layers(input_x)
        return output_x.view(output_x.size(0), self.output_size)

    def get_weights(self):
        """Get model weights as a list of NumPy ndarrays."""
        weights: Tensor = nn.utils.parameters_to_vector(self.parameters(recurse=True))
        weights = weights.detach().clone()
        return weights

    def set_weights(self, weights, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: weights = weights.detach().clone()
        nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))


class Net4(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes, latent_size, output_size) -> None:
        super(Net4, self).__init__()
        self.embedding_layers = nn.Embedding(num_classes, num_classes)
        self.layer_set_1 = nn.Sequential(
            nn.Linear(latent_size+num_classes, 256),
            nn.BatchNorm1d(256),  
            nn.ReLU(True),          
        )
        self.layer_set_2 = nn.Sequential(
            nn.Linear(256+num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.layer_set_3 = nn.Sequential(
            nn.Linear(256, 2),
            nn.Tanh(),
        )
        self.num_classes = num_classes
        self.output_size = output_size

    def forward(self, input_z: Tensor, input_labels: Tensor) -> Tensor:
        """Compute forward pass through the model"""
        emb_label = self.embedding_layers(input_labels)
        input_x = torch.cat([input_z, emb_label], dim = 1)
        output_x = self.layer_set_1(input_x)
        output_x = torch.cat([output_x, emb_label], dim = 1)
        output_x = self.layer_set_2(output_x)
        output_x = self.layer_set_3(output_x)
        return output_x.view(output_x.size(0), self.output_size)

    def get_weights(self):
        """Get model weights as a list of NumPy ndarrays."""
        weights: Tensor = nn.utils.parameters_to_vector(self.parameters(recurse=True))
        weights = weights.detach().clone()
        return weights

    def set_weights(self, weights, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: weights = weights.detach().clone()
        nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))
