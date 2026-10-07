"""Factory function for creating FL defense filters."""

import math
from typing import Dict

from fedml.defenses.base_filter import Filter


def create_filter(user_configs: Dict) -> Filter:
    """Create and return the appropriate defense filter."""
    filter_configs = user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]
    filter_type = filter_configs["FILTER_TYPE"]
    filter_param = filter_configs["HYPER_PARAM"]

    if filter_type == "GAN-FILTERING":
        from .gan_filter import GenerativeFilter
        return GenerativeFilter(
            gen_configs=filter_param["GEN_ARGS"],
            dis_configs=user_configs["MODEL_CONFIGS"],
            train_configs=filter_param["TRAIN_GAN_PARAMS"],
            filter_configs=filter_param["FILTER_ARGS"],
            skip_rounds=filter_param["SKIP_ROUNDS"],
        )

    elif filter_type == "MixingGAN-FILTERING":
        from .mixing_gan_filter import MixingGenerativeFilter
        return MixingGenerativeFilter(
            gen_configs=filter_param["GEN_ARGS"],
            dis_configs=user_configs["MODEL_CONFIGS"],
            train_configs=filter_param["TRAIN_GAN_PARAMS"],
            filter_configs=filter_param["FILTER_ARGS"],
            skip_rounds=filter_param["SKIP_ROUNDS"],
        )

    elif filter_type == "MEAN-FILTERING":
        from .mean_filter import MeanFilter
        return MeanFilter(filter_configs=filter_param["FILTER_ARGS"])

    elif filter_type == "KRUM-FILTERING":
        from .krum_filter import KrumFilter
        if "num_malicious_clients" not in filter_param:
            filter_param["num_malicious_clients"] = math.ceil(
                user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"]
                * user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"]
            )
        if "num_clients_to_keep" not in filter_param:
            filter_param["num_clients_to_keep"] = (
                user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"]
                - filter_param["num_malicious_clients"]
            )
        return KrumFilter(**filter_param)

    else:
        raise ValueError(f"Invalid filter type '{filter_type}' requested.")