"""A function to create desired type of FL server."""
from typing import Optional, Callable

def create_client(
        client_type: Optional[str],
        client_id: int,
        trainset,
        testset,
        process: bool,
        configs: dict,
        # model_fn: Optional[Callable] = None,
        run_device: Optional[str] = None,
    ):
    """Function to create the appropriat FL server instance."""

    # Shared kwargs for all clients
    base_kwargs = dict(
        client_id=client_id,
        trainset=trainset,
        testset=testset,
        # model_fn=model_fn,
        process=process,
        batch_train=configs["BATCH_TRAINING"],
        run_device=run_device,
    )

    if (client_type == "HONEST") or (client_type is None):
        from .honest_client import HonestClient
        return HonestClient(**base_kwargs)

    # All malicious clients additionally require attack_config
    attack_kwargs = {**base_kwargs, "attack_config": configs["MAL_HYPER_PARAM"]}

    if client_type == "RANDOM":
        from .malicious.random import RandomUpdateClient
        return RandomUpdateClient(**attack_kwargs)
    elif client_type == "ALIE":
        from .malicious.alie import ALIEClient
        return ALIEClient(**attack_kwargs)
    elif client_type == "IPM":
        from .malicious.ipm import IPMClient
        return IPMClient(**attack_kwargs)
    elif client_type == "MIMIC":
        from .malicious.mimic import MIMICClient
        return MIMICClient(**attack_kwargs)
    elif client_type == "SIGNFLIP":
        from .malicious.signflip import SignFlipClient
        return SignFlipClient(**attack_kwargs)
    elif client_type == "MPAF":
        from .malicious.mpaf import ModelReplacementClient
        return ModelReplacementClient(**attack_kwargs)
    elif client_type == "LABELFLIP":
        from .malicious.labelflip import LabelFlippingClient
        return LabelFlippingClient(**attack_kwargs)
    elif client_type == "BACKDOOR":
        from .malicious.backdoor import BackdoorClient
        return BackdoorClient(**attack_kwargs)
    else:
        raise ValueError(f"Invalid client type '{client_type}' requested.")
