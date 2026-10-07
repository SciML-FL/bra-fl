"""Base model implementation."""

from collections import OrderedDict
from unicodedata import name
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch import Tensor

from fedml.utils.typing import Parameters

class BaseModel(nn.Module):
    """Base class for Convolutional neural network."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        raise NotImplementedError(f"{self}.forward method not implemented")

    def get_weights(self) -> Parameters:
        """Get model weights as a list of NumPy ndarrays."""
        params: Tensor = nn.utils.parameters_to_vector(self.parameters(recurse=True))
        return params.detach().clone()

    def set_weights(self, weights: Parameters, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: 
            weights = weights.detach().clone()
        
        nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))
        
class ClassificationBaseModel(BaseModel):
    """Base class for classification models."""
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes