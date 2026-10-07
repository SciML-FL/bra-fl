"""Implementation of a ConvNeXt neural network."""

import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.convnext import ConvNeXt

from fedml.utils.typing import Parameters


CONVNEXT_VARIANTS = {
    "tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights.DEFAULT),
    "small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
    "base":  (convnext_base,  ConvNeXt_Base_Weights.DEFAULT),
}


class Net(ConvNeXt):
    """ConvNeXt network."""

    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "tiny") -> None:
        if variant not in CONVNEXT_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(CONVNEXT_VARIANTS)}")

        factory_fn, weights = CONVNEXT_VARIANTS[variant]

        # Build the base architecture (no pretrained weights yet)
        base_model = factory_fn(weights=None)

        # Build the full model via the factory, then hijack its internals.
        # We bypass ConvNeXt.__init__ (which requires CNBlockConfig objects we
        # don't want to reconstruct) and just init the base nn.Module directly.
        torch.nn.Module.__init__(self)

        # Copy the fully built feature extractor & classifier from the factory model
        self.features   = base_model.features
        self.avgpool    = base_model.avgpool
        self.classifier = base_model.classifier

        self.num_classes = num_classes

        # Replace the final linear head to match the requested number of classes
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = torch.nn.Linear(in_features, num_classes)

        if pretrained:
            state_dict = factory_fn(weights=weights).state_dict()
            state_dict.pop("classifier.2.weight", None)
            state_dict.pop("classifier.2.bias",   None)
            state_dict.pop("features.0.0.weight", None)
            state_dict.pop("features.0.0.bias",   None)
            self.load_state_dict(state_dict, strict=False)

        # Always adapt stem — do this last so pretrained load doesn't overwrite it
        self._adapt_for_small_images()

    # ------------------------------------------------------------------
    # Federated-learning helpers (identical interface to your ResNet)
    # ------------------------------------------------------------------

    def get_weights(self) -> Parameters:
        """Get model weights as a flat parameter vector."""
        params = torch.nn.utils.parameters_to_vector(self.parameters(recurse=True))
        return params.detach().clone()

    def set_weights(self, weights: Parameters, clone: bool = False) -> None:
        """Set model weights from a flat parameter vector."""
        if clone:
            weights = weights.detach().clone()
        torch.nn.utils.vector_to_parameters(weights, self.parameters(recurse=True))

    def _adapt_for_small_images(self) -> None:
        """
        Replace the default stride-4 stem with a stride-2 stem so that
        64x64 inputs produce reasonable intermediate feature map sizes:
        Default:  64 -> 16 -> 8 -> 4 -> 2   (too aggressive)
        Modified: 64 -> 32 -> 16 -> 8 -> 4  (much healthier)
        """
        original_stem_conv = self.features[0][0]  # Conv2d(3, C, kernel=4, stride=4)
        self.features[0][0] = torch.nn.Conv2d(
            in_channels=original_stem_conv.in_channels,
            out_channels=original_stem_conv.out_channels,
            kernel_size=3,   # smaller kernel fits the smaller input better
            stride=2,
            padding=1,
            bias=original_stem_conv.bias is not None,
        )