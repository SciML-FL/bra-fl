"""Run one federated experiment on a CPU or CUDA workstation."""

import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from logging import DEBUG

import copy
import torch
import os
from os.path import join
import argparse
import ntpath

import fedml
from fedml.utils import log, ExperimentManager, setup_random_seeds

from fedml.client import create_client
from fedml.configs import parse_configs
from fedml.data import load_and_fetch_split, merge_splits
from fedml.models import load_model
from fedml.server import (
    create_server,
    get_client_manager
)
from fedml.strategy import get_strategy


def resolve_run_devices(requested, num_gpus, min_sample_size, server_type, user_configs):
    """Resolve hardware without changing the experimental configuration."""
    if min_sample_size < 1:
        raise ValueError("MIN_TRAINING_SAMPLE_SIZE must be positive")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus is not None and (num_gpus < 0 or num_gpus > available):
        raise ValueError(f"Requested {num_gpus} GPUs, but {available} are available")
    if requested == "auto":
        count = available if num_gpus is None else num_gpus
        if count == 0:
            return ["cpu"] * min_sample_size
        if (server_type == "FILTER"
                and user_configs["SERVER_CONFIGS"]["FILTER_CONFIGS"]["FILTER_TYPE"] == "GAN-FILTERING"
                and count > 1):
            return [f"cuda:{i % (count - 1) + 1}" for i in range(min_sample_size)]
        return [f"cuda:{i % count}" for i in range(min_sample_size)]
    device = torch.device(requested)
    if device.type == "cpu":
        return ["cpu"] * min_sample_size
    if device.type != "cuda":
        raise ValueError("Supported devices are cpu, auto, cuda, or cuda:N")
    index = 0 if device.index is None else device.index
    if index >= available:
        raise ValueError(f"CUDA device {index} is unavailable ({available} GPUs detected)")
    return [f"cuda:{index}"] * min_sample_size


def single_node_simulation(exp_name, user_configs, executor_type, num_gpus=None, model_as_fn=False, max_workers=1):

    # Extract required user configurations
    total_clients = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]
    min_sample_size = user_configs["SERVER_CONFIGS"]["MIN_TRAINING_SAMPLE_SIZE"]
    server_type = user_configs["SERVER_CONFIGS"]["SERVER_TYPE"]

    # Get run device information
    run_devices = resolve_run_devices(
        user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"], num_gpus,
        min_sample_size, server_type, user_configs,
    )

    log(DEBUG, f"Run device: {run_devices[0]}")

    # Load all dataset and make splits 
    (train_splits, split_labels), testset = load_and_fetch_split(n_clients=total_clients, dataset_conf=user_configs["DATASET_CONFIGS"])

    # Load appropriate number of local models
    model_fn = load_model(model_configs=user_configs["MODEL_CONFIGS"], as_fn=True)
    local_models = [(model_fn if model_as_fn else model_fn()) for i in range(min_sample_size)]
    # local_models = [(model_fn if model_as_fn else model_fn()) for i in range(total_clients)]

    # Load pre-trained weights if any are provided
    server_model = model_fn()
    if "WEIGHT_PATH" in user_configs["MODEL_CONFIGS"].keys() and user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"] is not None:
        server_model.load_state_dict(torch.load(user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"], weights_only=False))
        if not model_as_fn:
            for local_model in local_models:
                local_model.load_state_dict(torch.load(user_configs["MODEL_CONFIGS"]["WEIGHT_PATH"], weights_only=False))

    # Create client objects based on client types
    mal_client_type = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"]
    num_mal_clients = int(user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"] * total_clients)
    num_hon_clients = total_clients - num_mal_clients

    # Setup clients with different types
    log(DEBUG, f"Creating {num_hon_clients} honest clients and {num_mal_clients} malicious clients of type {mal_client_type}.")
    clients = [
        create_client(
            None, 
            id, 
            trainset=train_splits[id], 
            testset=testset, 
            run_device=run_devices[(id)%len(run_devices)],
            process=(executor_type=="ProcessPool"), 
            configs=user_configs["EXPERIMENT_CONFIGS"],
        ) 
        for id in range(num_hon_clients)
    ]
    
    if user_configs["EXPERIMENT_CONFIGS"]["MAL_SHARED_DATA"]:
        # Merge train_splits reserved for malicious clients
        merged_trainset = merge_splits(train_splits[num_hon_clients:])
        clients.extend([
            create_client(
                mal_client_type, 
                id+num_hon_clients, 
                trainset=copy.deepcopy(merged_trainset), 
                testset=testset, 
                run_device=run_devices[(id+num_hon_clients)%len(run_devices)],
                process=(executor_type=="ProcessPool"), 
                configs=user_configs["EXPERIMENT_CONFIGS"]
            ) 
            for id in range(num_mal_clients)
        ])
    else:
        clients.extend([
            create_client(
                mal_client_type, 
                id+num_hon_clients, 
                trainset=train_splits[id+num_hon_clients], 
                testset=testset, 
                run_device=run_devices[(id+num_hon_clients)%len(run_devices)],
                process=(executor_type=="ProcessPool"), 
                configs=user_configs["EXPERIMENT_CONFIGS"]
            ) 
            for id in range(num_mal_clients)
        ])


    ###########################################################
    ###########################################################
    # Setup a Federated Server instance
    ###########################################################
    ###########################################################

    # Fetch stats and store them locally?
    exp_manager = ExperimentManager(experiment_id=exp_name, hyperparameters=user_configs)

    # Create aggregation strategy
    agg_strat = get_strategy(user_configs=user_configs, local_models=local_models, model_as_fn=model_as_fn, run_devices=run_devices)

    # Create a client manager
    client_manager = get_client_manager(user_configs=user_configs)

    # Register all clients with client_manager
    for client in clients: 
        client_manager.register(client=client)
    log(
        DEBUG, 
        f"Successfully registered {client_manager.num_available()} clients..."
    )

    initial_parameters = server_model.get_weights()
    if executor_type=="ProcessPool": 
        initial_parameters = initial_parameters.cpu()

    # Create the server instance
    fedml_server = create_server(
        server_type=server_type,
        client_manager=client_manager,
        strategy=agg_strat,
        user_configs=user_configs,
        experiment_manager=exp_manager,
        initial_parameters=initial_parameters,
        executor_type=executor_type,
        max_workers=max_workers,
    )

    # Train the server for specified number of rounds
    history, runtime = fedml_server.fit(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"])

    # Save logging results to disk
    log(
        DEBUG, 
        "Saving logged results to disk ..."
    )
    exp_manager.save_to_disc(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exp_name)
    history.save_to_disc(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exp_name)

    # Save the final model state to disk
    log(
        DEBUG, 
        "Saving final model parameters ..."
    )
    torch.save(obj=fedml_server.parameters, f=join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"weights-{exp_name}.pt"))

    # if user_configs["OUTPUT_CONFIGS"]["WANDB_LOGGING"]:
    #     print("Logging results to WANDB service ...")
    #     log_to_wandb(user_configs=user_configs, experiment_manager=exp_manager, experiment_name=exp_config)

    log(
        DEBUG, 
        f"Finished federated experiment {exp_name}.yaml ...\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Run experiment for given configuration file.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Optional GPU count for auto device selection; zero selects CPU",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path (no default)",
    )
    parser.add_argument(
        "--executor-type",
        type=str,
        default="ProcessPool",
        choices=["ProcessPool", "ThreadPool", "NONE"],
        help="Client executor (default: ProcessPool)",
    )
    parser.add_argument("--device", default=None, help="Override config device: cpu, auto, cuda, or cuda:N")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum client processes/threads (default: 1)")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")

    user_configs = parse_configs(args.config_file)
    if args.device is not None:
        user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"] = args.device
        user_configs["SERVER_CONFIGS"]["RUN_DEVICE"] = args.device
    exp_name = ntpath.basename(args.config_file)[:-5]

    # Setup random seeds before anything else
    setup_random_seeds(seed_value=user_configs["SERVER_CONFIGS"]["RANDOM_SEED"])

    # Create stdout re-direction files
    os.makedirs(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exist_ok=True)
    logfile = open( join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}.log"), "w")
    fedml.utils.logger.update_console_handler(level=DEBUG, stream=logfile)

    # Setup default torch device
    default_device = "cpu"
    # if torch.cuda.is_available(): # and args.executor_type == "ThreadPool":
    #     default_device = f"cuda:{torch.cuda.current_device()}"
    torch.set_default_device(default_device)

    log(DEBUG, f"# of GPUs       : {args.num_gpus}")
    log(DEBUG, f"Config File     : {args.config_file}")
    log(DEBUG, f"Executor Type   : {args.executor_type}")

    single_node_simulation(exp_name=exp_name, user_configs=user_configs, num_gpus=args.num_gpus, executor_type=args.executor_type, max_workers=args.max_workers)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
