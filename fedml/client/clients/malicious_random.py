"""Implementation of Honest Client using FedML Framework"""

import copy
from functools import reduce
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fedml.common import (
    FitIns,
    FitRes,
)

from .honest_client import HonestClient


class RandomUpdateClient(HonestClient):
    """A malicious client submitting random updates (noise)."""

    def __init__(
        self,
        client_id: int,
        trainset: Dataset,
        testset: Dataset,
        process: bool = True,
        attack_config: Optional[Dict] = None,
    ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
        self.attack_config = copy.deepcopy(attack_config)

    @property
    def client_type(self):
        """Returns current client's type."""
        return "RANDOM"

    def post_training_callback(self, results, failures):
        my_fit_result = None
        all_attacking = []
        for _, res in results:
            if res.metrics["client_id"] == self.client_id: my_fit_result = res
            if res.metrics["attacking"]: all_attacking.append(res)

        # With all attacking results and current client's results
        # compute the collusion update only if the current client
        # was attacking in current round.
        if my_fit_result.metrics["attacking"]:
            num_examples_total = sum(res.num_examples for res in all_attacking)
            weighted_weights = [res.num_examples * res.parameters for res in all_attacking]
            weights_prime = reduce(torch.add, weighted_weights) / num_examples_total
            del my_fit_result.parameters
            my_fit_result.parameters = weights_prime

        return my_fit_result

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round
        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)

        # Get training config
        # local_epochs = int(ins.config["epochs"])
        # batch_size = int(ins.config["batch_size"])
        # num_examples = batch_size * (len(self._trainset) // batch_size) * local_epochs

        fit_results = super().fit(model, device, ins=ins)
        fit_results.metrics["attacking"] = True

        # Compute location and scale parameter of the update
        mean = (
            self.attack_config["RANDOM_CONFIG"]["LOCATION"]
            if "LOCATION" in self.attack_config["RANDOM_CONFIG"].keys()
            else 0
        )

        # Create random weights
        if self.attack_config["RANDOM_CONFIG"]["TYPE"] == "UNIFORM":
            random_noise = torch.rand(
                ins.parameters.size(),
                dtype=torch.float32,
                layout=ins.parameters.layout,
                device=device,
            )
            # Adjust the location of random update.
            random_noise -= 0.5 + mean
            # Normalize the update
            random_noise.mul_(
                torch.abs(fit_results.parameters) / torch.abs(random_noise)
            )
        elif self.attack_config["RANDOM_CONFIG"]["TYPE"] == "NORMAL":
            scale_factor = self.attack_config["RANDOM_CONFIG"]["NORM_SCALE"]
            std = torch.abs(fit_results.parameters)
            std.mul_(scale_factor)
            random_noise = torch.normal(mean=mean, std=std)
            del fit_results.parameters
            fit_results.parameters = random_noise
        else:
            raise ValueError(
                "Invalid noise type "
                + self.attack_config["RANDOM_CONFIG"]["TYPE"]
                + " specified."
            )

        return fit_results
