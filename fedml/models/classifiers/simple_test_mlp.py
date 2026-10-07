"""Implementation of a simple multilayer perceptron network."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from .model import ClassificationBaseModel


class Net(nn.Module):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes) -> None:
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x = x.reshape(x.size(0), -1)
        # x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self):
        """Get model weights as a list of NumPy ndarrays."""
        weights: Tensor = nn.utils.parameters_to_vector(self.parameters(recurse=True))
        weights = weights.detach().clone()
        return weights

    def set_weights(self, weights, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: weights = weights.detach().clone()
        nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))
