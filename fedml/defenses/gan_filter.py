"""GAN-based generative defense filter."""

from collections import defaultdict
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from fedml.utils.typing import Parameters
from fedml.utils.logger import log
from fedml.models import load_model
from fedml.training import (
    evaluate_gan_classification,
    evaluate_gan_regression,
    get_criterion,
    get_optimizer,
    train_generator,
)
from fedml.utils.noise import get_noise_sampler

from fedml.defenses.base_filter import Filter


class GenerativeFilter(Filter):
    """Defense filter using a GAN to generate a validation dataset.

    Trains a generator adversarially against the current global model
    (used as discriminator) to produce synthetic validation data.
    Client updates are then evaluated on this data and filtered based
    on the configured filtration strategy.
    """

    def __init__(
        self,
        gen_configs: dict,
        dis_configs: dict,
        train_configs: dict,
        filter_configs: dict,
        skip_rounds: int = 1,
    ) -> None:
        self.gen_configs = gen_configs
        self.dis_configs = dis_configs
        self.train_configs = train_configs
        self.filter_configs = filter_configs
        self.skip_rounds = skip_rounds
        self.reinit = gen_configs.get("REINIT", False)

        self.gen_model = load_model(model_configs=gen_configs)
        self.dis_model = load_model(model_configs=dis_configs)
        self.criterion = get_criterion(criterion_str=train_configs["CRITERION"], **train_configs.get("CRITERION_ARG", {}))

        self.device = train_configs["DEVICE"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.noise_sampler = get_noise_sampler(
            distribution=gen_configs["NOISE_DIST"],
            noise_dim=gen_configs["NOISE_DIMS"],
            device=self.device,
        )

        if gen_configs["LABEL_DIST"] == "CLSLABEL":
            # For uniform label distribution, we need to specify the range
            self.label_sampler = get_noise_sampler(
                distribution=gen_configs["LABEL_DIST"],
                num_classes=gen_configs["LABEL_RANGE"][1],
                device=self.device,
            )
            self.classification = True
        else:
            self.label_sampler = get_noise_sampler(
                distribution=gen_configs["LABEL_DIST"],
                noise_dim=1,
                device=self.device,
                low=gen_configs["LABEL_RANGE"][0],
                high=gen_configs["LABEL_RANGE"][1],
            )
            self.classification = False

        # Pre-generate fixed noise inputs for consistent dataset generation
        self.input_znoises = self.noise_sampler.sample(
            batch_size=filter_configs["TOTAL_SAMPLES"]
        )
        self.input_classes = self.label_sampler.sample(
            batch_size=filter_configs["TOTAL_SAMPLES"]
        )

    @property
    def filter_type(self) -> str:
        return "GAN"

    def change_device(self, device: str) -> None:
        """Move models to specified device."""
        self.gen_model = self.gen_model.to(device)
        self.dis_model = self.dis_model.to(device)


    def server_tasks(self, global_weights: Parameters, server_round: int) -> None:
        """Train generator against current global model as discriminator."""
        if server_round <= self.skip_rounds:
            log(INFO, "filter_updates: Skipping generator training at server round %s", server_round)
            return None     # nothing to update

        self.dis_model.set_weights(weights=global_weights)

        # Re-initialize the generator at the start of each training round if specified in config
        if self.reinit:
            self.gen_model = load_model(model_configs=self.gen_configs) 

        # Stage models to correct device for training
        self.change_device(self.device)

        gen_optimizer = get_optimizer(
            optimizer_str=self.train_configs["OPTIMIZER"],
            local_model=self.gen_model,
            learning_rate=self.train_configs["LEARN_RATE"],
        )

        self.gen_model = train_generator(
            gen_model=self.gen_model,
            dis_model=self.dis_model,
            gen_optim=gen_optimizer,
            criterion=self.criterion,
            iterations=self.train_configs["ITERATION"],
            noise_dist=self.noise_sampler,
            label_dist=self.label_sampler,
            batch_size=self.train_configs["BATCH_SIZE"],
            patience=self.train_configs["PATIENCE"],
            delta=self.train_configs["DELTA"],
            check_every=self.train_configs["CHECK_EVERY"],
            device=self.device,
        )

        # Return only the trained weights, not the whole object
        trained_gen_weights = self.gen_model.get_weights().cpu()
        return trained_gen_weights

    def filter_updates(
        self,
        client_weights: List[Tuple[Parameters, int]],
        server_round: int,
    ) -> Tuple[List[int], Optional[Dict[str, List]]]:
        """Filter client updates using GAN-generated validation data."""
        if server_round <= self.skip_rounds:
            log(INFO, "filter_updates: Skipping filtration at server round %s", server_round)
            return list(range(len(client_weights))), None

        gen_dataset = _generate_dataset(
            gen_model=self.gen_model,
            input_znoises=self.input_znoises,
            input_classes=self.input_classes,
            device=self.device,
            batch_size=1024,
        )

        select_ids, client_stats = _perform_filtration(
            filter_configs=self.filter_configs,
            dis_model=self.dis_model,
            gen_dataset=gen_dataset,
            client_weights=client_weights,
            criterion=self.criterion,
            device=self.device,
            classification=self.classification
        )

        log(INFO, "filter_updates: (round=%s, total=%s, selected=%s)", server_round, len(client_weights), len(select_ids))
        # log(INFO, "filter_updates: (round=%s, total=%s, selected=%s)", server_round, len(client_weights), len(select_ids))
        log(DEBUG, "filter_updates: accuracies %s", client_stats["accu_all"])
        log(DEBUG, "filter_updates: avg_loss %s",
            [f"{i:.4f}" for i in client_stats["avg_loss"]])

        return select_ids, client_stats


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _perform_filtration(
    filter_configs: dict,
    dis_model: nn.Module,
    gen_dataset: TensorDataset,
    client_weights: List[Tuple[Parameters, int]],
    criterion,
    device: str,
    classification: bool = True,
) -> Tuple[List[int], Dict[str, List]]:
    """Evaluate each client update on GAN-generated data and apply selection rule."""
    dataloader = DataLoader(gen_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    threshold = filter_configs["BASELINE_OVERALL_MIN_ACC"]
    filtration_type = filter_configs["FILTERATION_TYPE"]

    select_ids = []
    client_stats = defaultdict(list)
    raw_outputs = []

    for index, (weights, _) in enumerate(client_weights):
        dis_model.set_weights(weights=weights)
        if classification:
            stats = evaluate_gan_classification(
                dis_model=dis_model,
                testloader=dataloader,
                device=device,
                criterion=criterion,
                raw_output=(filtration_type == "CLUSTER-RAW-OUTPUTS"),
            )
        else:
            stats = evaluate_gan_regression(
                dis_model=dis_model,
                testloader=dataloader,
                device=device,
                criterion=criterion,
            )

        client_stats["avg_loss"].append(stats[0])
        client_stats["accu_all"].append(stats[1])
        if stats[2] is not None:
            raw_outputs.append(np.vstack(stats[2]).ravel())

    # Selection rules
    if filtration_type == "BASELINE-OVERALL":
        select_ids = [i for i, a in enumerate(client_stats["accu_all"]) if a >= threshold]

    elif filtration_type == "MEAN-LOSS":
        tanh_losses = nn.functional.tanh(torch.tensor(client_stats["avg_loss"]))
        avg = tanh_losses.mean().item()
        select_ids = [i for i, l in enumerate(tanh_losses) if l <= avg]

    elif filtration_type == "MEDIAN-LOSS":
        med = np.median(client_stats["avg_loss"])
        select_ids = [i for i, l in enumerate(client_stats["avg_loss"]) if l <= med]

    elif filtration_type == "MIXED-LOSS":
        med = np.median(client_stats["avg_loss"])
        avg = np.mean(client_stats["avg_loss"])
        baseline = min(med, avg)
        select_ids = [i for i, l in enumerate(client_stats["avg_loss"]) if l <= baseline]

    elif filtration_type == "MIXED-2-LOSS":
        baseline = 0.5 * (np.median(client_stats["avg_loss"]) + np.mean(client_stats["avg_loss"]))
        select_ids = [i for i, l in enumerate(client_stats["avg_loss"]) if l <= baseline]

    elif filtration_type == "MEDIAN-ACCURACY":
        med = np.median(client_stats["accu_all"])
        select_ids = [i for i, a in enumerate(client_stats["accu_all"]) if a >= med]

    elif filtration_type == "MIXED-ACCURACY":
        med = np.median(client_stats["accu_all"])
        avg = np.mean(client_stats["accu_all"])
        baseline = max(med, avg)
        select_ids = [i for i, a in enumerate(client_stats["accu_all"]) if a >= baseline]

    elif filtration_type == "MIXED-2-ACCURACY":
        baseline = 0.5 * (np.median(client_stats["accu_all"]) + np.mean(client_stats["accu_all"]))
        select_ids = [i for i, a in enumerate(client_stats["accu_all"]) if a >= baseline]

    elif filtration_type == "CLUSTER-ACCURACY":
        accu_arr = np.array(client_stats["accu_all"]).reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=20).fit(accu_arr)
        clusters = km.predict(accu_arr)
        select_ids = np.where(clusters == clusters[accu_arr.argmax()])[0].tolist()

    elif filtration_type == "CLUSTER-LOSS":
        loss_arr = nn.functional.tanh(
            torch.tensor(client_stats["avg_loss"])
        ).detach().cpu().numpy().reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=20).fit(loss_arr)
        clusters = km.predict(loss_arr)
        select_ids = np.where(clusters == clusters[loss_arr.argmin()])[0].tolist()

    elif filtration_type == "CLUSTER-RAW-OUTPUTS":
        output_arr = np.array(raw_outputs)
        km = KMeans(n_clusters=2, n_init=20).fit(output_arr)
        clusters = km.predict(output_arr)
        c0 = np.where(clusters == 0)[0].tolist()
        c1 = np.where(clusters == 1)[0].tolist()
        select_ids = c0 if len(c0) > len(c1) else c1

    else:
        raise ValueError(f"Invalid filtration type '{filtration_type}' specified.")

    return select_ids, client_stats


def _generate_dataset(
    gen_model: nn.Module,
    input_znoises: torch.Tensor,
    input_classes: torch.Tensor,
    device: str,
    batch_size: int,
) -> TensorDataset:
    """Generate synthetic validation dataset using the trained generator."""
    if next(gen_model.parameters()).device != device:
        gen_model.to(device)

    data_X, data_Y = [], []
    gen_model.eval()

    with torch.no_grad():
        for start in range(0, input_classes.size(0), batch_size):
            z = input_znoises[start:start + batch_size].to(device)
            l = input_classes[start:start + batch_size].to(device)
            data_X.append(gen_model(z, l).cpu())
            data_Y.append(l.reshape(-1, input_classes.size(1)).cpu())

        return TensorDataset(torch.vstack(data_X), torch.vstack(data_Y))
