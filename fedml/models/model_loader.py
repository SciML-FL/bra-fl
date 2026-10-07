"""A function to load desired model for training."""

class _ModelFactory:
    """Named callable so ProcessPool can pickle it."""
    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def __call__(self):
        return self.cls(**self.kwargs)

def load_model(model_configs: dict, as_fn: bool = False):
    """Load requested model.
    
    :param model_configs: A dictionary containing model configuration parameters.
    :param as_fn: If True, returns a zero-argument callable that constructs the model (useful for passing to workers without instantiating the model yet).
    :returns: The requested model instance or a callable that constructs the model if as_fn is True.
    """

    def _make(cls, **kwargs):
        if as_fn:
            return _ModelFactory(cls, **kwargs)
        return cls(**kwargs)

    if model_configs["MODEL_NAME"] == "SIMPLE-MLP":
        from .classifiers.simple_mlp import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "SIMPLE-CNN":
        from .classifiers.simple_cnn import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-1CH":
        from .classifiers.lenet_1ch import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-1CH-BN":
        from .classifiers.lenet_1ch_bn import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-3CH":
        from .classifiers.lenet_3ch import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-3CH-BN":
        from .classifiers.lenet_3ch_bn import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "RESNET-18-PYTORCH":
        from .classifiers.resnet_pytorch import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "PRERESNET-20":
        from .classifiers.preresnet import preresnet20
        return _make(preresnet20, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "RESNET-18-CUSTOM":
        from .classifiers.resnet_custom import ResNet18
        return _make(ResNet18, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "CONVNEXT-TINY":
        from .classifiers.convnext import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"], variant="tiny")
    elif model_configs["MODEL_NAME"] == "CONVNEXT-SMALL":
        from .classifiers.convnext import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"], variant="small")
    elif model_configs["MODEL_NAME"] == "CONVNEXT-BASE":
        from .classifiers.convnext import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"], variant="base")
    elif model_configs["MODEL_NAME"] == "TEST-MLP":
        from .classifiers.simple_test_mlp import Net
        return _make(Net, num_classes=model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "DNN-REGRESSOR":
        from .regressors.dnn_regressor import DNNRegressor
        return _make(DNNRegressor, input_dim=model_configs["INPUT_DIM"])

    # The following are very special models
    # used as generators for GAN based filteration.
    elif model_configs["MODEL_NAME"] == "GEN-TANH":
        from .generators.generator_tanh import GeneratorTanh
        return _make(GeneratorTanh,
            label_range=model_configs["LABEL_RANGE"],
            latent_dim=model_configs["NOISE_DIMS"],
            output_shape=model_configs["OUTPUT_SHAPE"])
    elif model_configs["MODEL_NAME"] == "GEN-RELU":
        from .generators.generator_relu import GeneratorRelu
        return _make(GeneratorRelu,
            label_range=model_configs["LABEL_RANGE"],
            latent_dim=model_configs["NOISE_DIMS"],
            output_shape=model_configs["OUTPUT_SHAPE"])
    elif model_configs["MODEL_NAME"] == "GEN-SIGMOID":
        from .generators.generator_sigmoid import GeneratorSigmoid
        return _make(GeneratorSigmoid,
            label_range=model_configs["LABEL_RANGE"],
            latent_dim=model_configs["NOISE_DIMS"],
            output_shape=model_configs["OUTPUT_SHAPE"])
    elif model_configs["MODEL_NAME"] == "GEN-DCGAN":
        from .generators.generator_dcgan import GeneratorDCGAN
        return _make(GeneratorDCGAN,
            label_range=model_configs["LABEL_RANGE"],
            latent_dim=model_configs["NOISE_DIMS"],
            output_shape=model_configs["OUTPUT_SHAPE"])
    elif model_configs["MODEL_NAME"] == "GEN-REGRESSOR":
        from .generators.generator_regressor import GeneratorRegressor
        return _make(GeneratorRegressor,
            latent_dim=model_configs["NOISE_DIMS"],
            output_shape=model_configs["OUTPUT_SHAPE"])
    else:
        raise ValueError(f"Invalid model {model_configs['MODEL_NAME']} requested.")
