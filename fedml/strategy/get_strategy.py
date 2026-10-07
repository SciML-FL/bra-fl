"""Factory function for creating FL aggregation strategies."""

from asyncio import run
import math
from fedml.data import load_data

from fedml.strategy.helpers import (
    get_evaluate_fn,
    get_fit_config_fn,
    get_evaluate_config_fn
)
from fedml.strategy.metrics import (
    aggregate_fit_metrics,
    aggregate_evaluate_metrics,
)


def get_strategy(user_configs: dict, local_models: list, model_as_fn: bool=True, run_devices: list = ["cpu"]):
    """Build and return the configured aggregation strategy."""

    server_cfg = user_configs["SERVER_CONFIGS"]
    client_cfg = user_configs["CLIENT_CONFIGS"]
    dataset_cfg = user_configs["DATASET_CONFIGS"]
    model_cfg = user_configs["MODEL_CONFIGS"]
    experiment_cfg = user_configs["EXPERIMENT_CONFIGS"]

    # ------------------------------------------------------------------
    # Server-side evaluation function (optional)
    # ------------------------------------------------------------------
    eval_fn = None
    if server_cfg["EVALUATE_SERVER"]:
        _, testset = load_data(
            dataset_name=dataset_cfg["DATASET_NAME"],
            dataset_path=dataset_cfg["DATASET_PATH"],
            dataset_down=dataset_cfg["DATASET_DOWN"],
            random_seed=dataset_cfg["RANDOM_SEED"],
        )
        eval_fn = get_evaluate_fn(
            testset=testset,
            # model_configs=model_cfg,
            model=local_models[0],
            model_as_fn=model_as_fn,
            device=run_devices[0],
            criterion_str=client_cfg["CRITERION"],
        )

    # ------------------------------------------------------------------
    # Config functions
    # ------------------------------------------------------------------
    fit_config_fn = get_fit_config_fn(
        total_rounds=server_cfg["NUM_TRAIN_ROUND"],
        local_epochs=client_cfg["LOCAL_EPCH"],
        lr_scheduler=client_cfg["LR_SCHEDULER"],
        scheduler_args=client_cfg["SCHEDULER_ARGS"],
        local_batchsize=client_cfg["BATCH_SIZE"],
        learning_rate=client_cfg["LEARN_RATE"],
        initial_lr=client_cfg["INITIAL_LR"],
        lr_warmup_steps=client_cfg["WARMUP_RDS"],
        optimizer_str=client_cfg["OPTIMIZER"],
        criterion_str=client_cfg["CRITERION"],
        perform_evals=client_cfg["EVALUATE"],
        optim_kwargs=client_cfg["OPTIM_ARG"],
        base_seed=server_cfg["RANDOM_SEED"],
    )

    evaluate_config_fn = get_evaluate_config_fn(
        total_rounds=server_cfg["NUM_TRAIN_ROUND"],
        evaluate_bs=client_cfg["BATCH_SIZE"],
        criterion_str=client_cfg["CRITERION"],
    )

    # ------------------------------------------------------------------
    # Base kwargs shared by every strategy
    # ------------------------------------------------------------------
    base_kwargs = dict(
        local_models=local_models,
        model_as_fn=model_as_fn,
        run_devices=run_devices,
        fraction_fit=server_cfg["TRAINING_SAMPLE_FRACTION"],
        min_fit_clients=server_cfg["MIN_TRAINING_SAMPLE_SIZE"],
        fraction_evaluate=server_cfg["EVALUATE_SAMPLE_FRACTION"],
        min_evaluate_clients=server_cfg["MIN_EVALUATE_SAMPLE_SIZE"],
        min_available_clients=server_cfg["MIN_NUM_CLIENTS"],
        evaluate_fn=eval_fn,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    # Strategy-specific kwargs from config (may be empty)
    strategy_kwargs = server_cfg.get("AGGR_STRAT_ARGS", {})

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    strategy_name = server_cfg["AGGREGATE_STRAT"]

    if strategy_name == "FED-AVERAGE":
        from .aggregators.fedavg import FederatedAverage
        return FederatedAverage(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-BAYESIAN":
        from .aggregators.bayesian import FederatedBayesian
        return FederatedBayesian(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-MEDIAN":
        from .aggregators.median import FederatedMedian
        return FederatedMedian(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-GEOMED":
        from .aggregators.geomed import FederatedGeometricMedian
        return FederatedGeometricMedian(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-KRUM":
        _resolve_krum_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.krum import FederatedKrum
        return FederatedKrum(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-TRIMAVG":
        _resolve_trimavg_kwargs(strategy_kwargs, experiment_cfg)
        from .aggregators.trimmedavg import FederatedTrimmedAverage
        return FederatedTrimmedAverage(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-MIXING":
        _resolve_mixing_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.mixing import FederatedMixing
        return FederatedMixing(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-BULYAN":
        from .aggregators.bulyan import FederatedBulyan
        return FederatedBulyan(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-MDA":
        _resolve_mda_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.mda import FederatedMDA
        return FederatedMDA(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-CAF":
        _resolve_caf_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.caf import FederatedCAF
        return FederatedCAF(**base_kwargs, **strategy_kwargs)

    elif strategy_name == "FED-SMEA":
        _resolve_smea_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.smea import FederatedSMEA
        return FederatedSMEA(**base_kwargs, **strategy_kwargs)
    
    elif strategy_name == "FED-MEAMED":
        _resolve_meamed_kwargs(strategy_kwargs, experiment_cfg, server_cfg)
        from .aggregators.meamed import FederatedMeamed
        return FederatedMeamed(**base_kwargs, **strategy_kwargs)

    else:
        raise ValueError(f"Invalid aggregation strategy '{strategy_name}' requested.")


# ------------------------------------------------------------------
# Private helpers — resolve default strategy kwargs from experiment config
# ------------------------------------------------------------------

def _resolve_krum_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in Krum defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )
    if strategy_kwargs.get("num_clients_to_keep") is None:
        strategy_kwargs["num_clients_to_keep"] = (
            server_cfg["MIN_TRAINING_SAMPLE_SIZE"] - strategy_kwargs["num_malicious_clients"]
        )

def _resolve_trimavg_kwargs(strategy_kwargs: dict, experiment_cfg: dict):
    """Fill in TrimmedAverage defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("beta") is None:
        strategy_kwargs["beta"] = experiment_cfg["MAL_CLIENT_FRAC"]

def _resolve_mixing_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in Mixing defaults from experiment config if not explicitly set."""
    if not strategy_kwargs.get("aggregator_to_use"):
        strategy_kwargs["aggregator_to_use"] = "KRUM"
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )
    if (
        strategy_kwargs["aggregator_to_use"] == "KRUM"
        and strategy_kwargs.get("num_clients_to_keep") is None
    ):
        strategy_kwargs["num_clients_to_keep"] = (
            server_cfg["MIN_TRAINING_SAMPLE_SIZE"] - strategy_kwargs["num_malicious_clients"]
        )

def _resolve_caf_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in CAF defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )

def _resolve_meamed_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in MEAMED defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )

def _resolve_mda_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in MDA defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )

def _resolve_smea_kwargs(strategy_kwargs: dict, experiment_cfg: dict, server_cfg: dict):
    """Fill in SMEA defaults from experiment config if not explicitly set."""
    if strategy_kwargs.get("num_malicious_clients") is None:
        strategy_kwargs["num_malicious_clients"] = math.ceil(
            experiment_cfg["MAL_CLIENT_FRAC"] * server_cfg["MIN_TRAINING_SAMPLE_SIZE"]
        )
