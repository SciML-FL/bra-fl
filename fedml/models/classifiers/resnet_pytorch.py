"""Implementation of a ResNet-18 neural network for 3."""

import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock

from fedml.utils.typing import Parameters

model_urls = "https://download.pytorch.org/models/resnet18-5c106cde.pth"

class Net(ResNet):
    """Multilayer percenptron (MLP) network."""
    def __init__(self, num_classes, pretrained=False) -> None:
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)

        # Replace the final fully connected layer to match the number of classes
        num_ftrs = self.fc.in_features
        self.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.num_classes = num_classes

        # Replace all batch normalization layers with group normalization to avoid issues in federated learning
        # self.replace_bn_with_gn(num_groups=32)

        # if pretrained:
        #     state_dict = model_zoo.load_url(model_urls)
        #     state_dict.pop("fc.weight", None)
        #     state_dict.pop("fc.bias", None)
        #     self.load_state_dict(state_dict, strict=False)

    def get_weights(self) -> Parameters:
        """Get model weights as a list of NumPy ndarrays."""
        params = torch.nn.utils.parameters_to_vector(self.parameters(recurse=True))
        return params.detach().clone()

    def set_weights(self, weights: Parameters, clone = False) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if clone: 
            weights = weights.detach().clone()
        
        torch.nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))

    def replace_bn_with_gn(self, num_groups=32):
        self._replace_bn_with_gn_recursive(self, num_groups)

    @staticmethod
    def _replace_bn_with_gn_recursive(module, num_groups):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                num_channels = child.num_features
                setattr(module, name, torch.nn.GroupNorm(
                    num_groups=min(num_groups, num_channels),
                    num_channels=num_channels
                ))
            else:
                Net._replace_bn_with_gn_recursive(child, num_groups)  # recurse deeper