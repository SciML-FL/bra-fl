"""Offline synthetic smoke test for CPU training, aggregation, and failure reporting.

This deliberately tiny test verifies installation and execution, not paper results.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import torch

from fedml.configs import parse_configs
from fedml.data.split import CustomDataset
from fedml.models import load_model
from fedml.run_federated import resolve_run_devices, single_node_simulation
from fedml.utils.random import setup_random_seeds
from run_sweep import ROOT, verify_result

# Keep this synthetic test fast, including its spawned worker.
torch.set_num_threads(1)


def synthetic_data(features=2):
    values = torch.linspace(-1.0, 1.0, 16 * features).reshape(16, features)
    labels = (values[:, 0] > 0).long()
    train = CustomDataset(values, labels)
    test = CustomDataset(values.clone(), labels.clone())
    parts = [CustomDataset(values[i::2].clone(), labels[i::2].clone()) for i in range(2)]
    return train, test, parts


def run_synthetic(config, name, features=2):
    train, test, parts = synthetic_data(features)
    strategy_module = importlib.import_module("fedml.strategy.get_strategy")
    with patch("fedml.run_federated.load_and_fetch_split", return_value=((parts, None), test)), \
            patch.object(strategy_module, "load_data", return_value=(train, test)):
        setup_random_seeds(config["SERVER_CONFIGS"]["RANDOM_SEED"])
        single_node_simulation(name, config, "ProcessPool", num_gpus=0, max_workers=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="work/smoke-test", help="New or empty directory for synthetic test artifacts")
    args = parser.parse_args()
    output = (ROOT / args.output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = parse_configs(ROOT / "papers/p02_bayesian_aggregation/experiments/2026_bayesian/templates/cifar10.yaml")
    config["SERVER_CONFIGS"].update(
        RANDOM_SEED=333, SERVER_TYPE="NORMAL", NUM_TRAIN_ROUND=1,
        MIN_NUM_CLIENTS=2, MIN_TRAINING_SAMPLE_SIZE=2, TRAINING_SAMPLE_FRACTION=1.0,
        EVALUATE_SAMPLE_FRACTION=0.0, MIN_EVALUATE_SAMPLE_SIZE=0,
        CLIENTS_MANAGER="SIMPLE", EVALUATE_SERVER=True,
        AGGREGATE_STRAT="FED-BAYESIAN", AGGR_STRAT_ARGS={"variation": "v2"},
    )
    config["CLIENT_CONFIGS"].update(
        RUN_DEVICE="auto", LOCAL_EPCH=1, BATCH_SIZE=4, CRITERION="CROSSENTROPY",
        OPTIMIZER="SGD", INITIAL_LR=0.01, LEARN_RATE=0.01, WARMUP_RDS=0,
        LR_SCHEDULER="STATIC", SCHEDULER_ARGS={}, OPTIM_ARG={}, EVALUATE=False,
    )
    config["MODEL_CONFIGS"].update(MODEL_NAME="TEST-MLP", NUM_CLASSES=2, WEIGHT_PATH=None)
    config["DATASET_CONFIGS"]["DATASET_NAME"] = "SYNTHETIC-SMOKE"
    config["EXPERIMENT_CONFIGS"].update(MAL_CLIENT_FRAC=0.0, MAL_CLIENT_TYPE=None, MAL_SHARED_DATA=False, BATCH_TRAINING=False)
    config["OUTPUT_CONFIGS"]["WANDB_LOGGING"] = False
    assert resolve_run_devices("auto", 0, 2, "NORMAL", config) == ["cpu", "cpu"]
    setup_random_seeds(333)
    factory = load_model(config["MODEL_CONFIGS"], as_fn=True)
    # Mirror the original initialization order: two local models, then the server.
    factory()
    factory()
    initial = factory().get_weights()
    config["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"] = str(output)
    run_synthetic(config, "smoke")
    checks = verify_result(output / "smoke.npz", config, output / "weights-smoke.pt")
    final = torch.load(output / "weights-smoke.pt", map_location="cpu", weights_only=True)
    assert checks["rounds"] == 1 and checks["clients_per_round"] == 2
    assert checks["finite_centralized_losses"] and bool(torch.isfinite(final).all())
    assert not torch.equal(initial, final), "Training did not update the model"
    bad = copy.deepcopy(config)
    bad["SERVER_CONFIGS"]["EVALUATE_SERVER"] = False
    try:
        run_synthetic(bad, "deliberate_failure", features=3)
    except RuntimeError as exc:
        assert "refusing to record a partial-client experiment" in str(exc), str(exc)
    else:
        raise AssertionError("A worker failure was silently accepted")
    assert not (output / "deliberate_failure.npz").exists()
    print("PASS: CPU auto/zero-GPU resolution; 2 clients, 1 Bayesian round, finite updated weights; worker failures abort without success artifacts.")
    print(f"Synthetic test artifacts: {output}")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    raise SystemExit(main())
