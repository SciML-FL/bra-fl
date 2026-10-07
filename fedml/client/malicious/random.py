"""Implementation of Honest Client using FedML Framework"""

import copy
from functools import reduce
from typing import Dict, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from fedml.utils.typing import (
    FitIns,
    FitRes,
)

from fedml.client import BaseClient
from fedml.utils.random import derive_seed


class RandomUpdateClient(BaseClient):
    """A malicious client submitting random updates (noise)."""

    def __init__(
        self,
        client_id: int,
        trainset: Dataset,
        testset: Dataset,
        batch_train: bool = False,
        process: bool = True,
        attack_config: Optional[Dict] = None,
        run_device: str = "cpu",
    ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            batch_train=batch_train,
            process=process,
            run_device=run_device,
        )
        self.attack_config = copy.deepcopy(attack_config)
        self.rng_generator = torch.Generator(device="cpu")

    @property
    def client_type(self):
        """Returns current client's type."""
        return "RANDOM"

    def post_training_callback(self, results, failures):
        my_fit_result = None
        all_attacking = []
        for _, res in results:
            if res.metrics["client_id"] == self.client_id: 
                my_fit_result = res
            if res.metrics["attacking"]: 
                all_attacking.append(res)

        # With all attacking results and current client's results
        # compute the collusion update only if the current client
        # was attacking in current round.
        if my_fit_result.metrics["attacking"]:
            num_examples_total = sum(res.num_examples for res in all_attacking)
            weighted_weights = [res.num_examples * res.parameters for res in all_attacking]
            weights_prime = reduce(torch.add, weighted_weights) / num_examples_total
            if self.process:
                del my_fit_result.parameters
            my_fit_result.parameters = weights_prime.detach().clone()

        return my_fit_result

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round
        server_round = int(fit_ins.config["server_round"])
        attack_rng = np.random.default_rng(
            derive_seed(int(fit_ins.config.get("base_seed", 0)), self.client_id, server_round, salt=1)
        )
        attack = attack_rng.random() < self.attack_config["ATTACK_RATIO"]

        # Compute the honest update and limit the
        # malicious update around the honest update
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return fit_results

        # Get training config
        # local_epochs = int(fit_ins.config["epochs"])
        # batch_size = int(fit_ins.config["batch_size"])
        # num_examples = batch_size * (len(self.trainset) // batch_size) * local_epochs

        # Compute location and scale parameter of the update
        mean = (
            self.attack_config["RANDOM_CONFIG"]["LOCATION"]
            if "LOCATION" in self.attack_config["RANDOM_CONFIG"].keys()
            else 0
        )

        # Create random weights
        if self.attack_config["RANDOM_CONFIG"]["TYPE"] == "UNIFORM":
            random_noise = rand_like(
                fit_ins.parameters,
                dtype=torch.float32,
                generator=self.rng_generator.manual_seed(server_round),

            )
            # Adjust the location of random update.
            random_noise -= 0.5 + mean
        elif self.attack_config["RANDOM_CONFIG"]["TYPE"] == "UNIFORM-2":
            random_noise = rand_like(
                fit_ins.parameters,
                dtype=torch.float32,
                generator=self.rng_generator.manual_seed(server_round),
            )
            # Adjust the location of random update.
            random_noise -= 0.5 + fit_results.parameters
        elif self.attack_config["RANDOM_CONFIG"]["TYPE"] == "NORMAL":
            # Generate random noise from Gaussian Distribution
            scale_factor = self.attack_config["RANDOM_CONFIG"]["NORM_SCALE"]
            std = torch.abs(fit_results.parameters) + torch.finfo(fit_results.parameters.dtype).eps
            std.mul_(scale_factor)
            random_noise = torch.normal(mean=mean, std=std, generator=self.rng_generator.manual_seed(server_round))
            # random_noise = normal(mean=mean, std=std) 

        else:
            raise ValueError(
                "Invalid noise type "
                + self.attack_config["RANDOM_CONFIG"]["TYPE"]
                + " specified."
            )

        del fit_results.parameters
        fit_results.parameters = random_noise.detach().clone()
        fit_results.metrics["attacking"] = True

        return fit_results

def normal(mean, std, generator=None, device=None, **kwargs):
    """
    Wrapper around torch.normal providing proper reproducibility.

    Generation is done on the given generator's device, then moved to the
    given ``device``.

    Args:
        size: tensor size
        generator (torch.Generator): RNG generator
        device (torch.device): Target device for the resulting tensor
    """
    rng_device = generator.device if generator is not None else device
    rnd_tensor = torch.normal(mean=mean, std=std, generator=generator, device=rng_device, **kwargs)
    rnd_tensor = rnd_tensor.to(device=device)
    return rnd_tensor

def rand(size, generator=None, device=None, **kwargs):
    """
    Wrapper around torch.rand providing proper reproducibility.

    Generation is done on the given generator's device, then moved to the
    given ``device``.

    Args:
        size: tensor size
        generator (torch.Generator): RNG generator
        device (torch.device): Target device for the resulting tensor
    """
    # FIXME: generator RNG device is ignored and needs to be passed to torch.randn (torch issue #62451)
    rng_device = generator.device if generator is not None else device
    rnd_tensor = torch.randn(size, generator=generator, device=rng_device, **kwargs)
    rnd_tensor = rnd_tensor.to(device=device)
    return rnd_tensor

def rand_like(tensor, generator=None, **kwargs):
    return rand(tensor.shape, layout=tensor.layout, generator=generator, device=tensor.device, **kwargs)