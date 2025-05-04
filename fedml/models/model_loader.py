"""A function to load desired model for training."""

def load_model(model_configs: dict):
    """Load requested model."""
    if model_configs["MODEL_NAME"] == "SIMPLE-MLP":
        from .simple_mlp import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "SIMPLE-CNN":
        from .simple_cnn import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-1CH":
        from .lenet_1ch import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-1CH-BN":
        from .lenet_1ch_bn import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-3CH":
        from .lenet_3ch import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-3CH-BN":
        from .lenet_3ch_bn import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "RESNET-18-PYTORCH":
        from .resnet_pytorch import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "PRERESNET-20":
        from .preresnet import preresnet20
        return preresnet20(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "RESNET-18-CUSTOM":
        from .resnet_custom import ResNet18
        return ResNet18(num_classes = model_configs["NUM_CLASSES"])
    else:
        raise ValueError(f"Invalid model {model_configs['MODEL_NAME']} requested.")
