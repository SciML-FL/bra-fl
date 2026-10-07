"""Implementation of Honest Client using FedML Framework"""

import copy
from functools import reduce
import numpy as np

import torch
from torch.utils.data import Dataset

from logging import DEBUG
from typing import Optional, Dict, Callable
from fedml.utils.typing import (
    FitIns,
    FitRes,
)

from fedml.client import BaseClient
from fedml.utils.random import derive_seed


class SignFlipClient(BaseClient):
    """A malicious client submitting updates with flipped gradient signs.
    
    """
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
        self.scale_factor = self.attack_config["SIGNFLIP_CONFIG"]["SCALE_FACTOR"]

    @property
    def client_type(self):
        """Returns current client's type."""
        return "SIGNFLIP"

    # def post_training_callback(self, results, failures):
    #     my_fit_result = None
    #     all_attacking = []
    #     for _, res in results:
    #         if res.metrics["client_id"] == self.client_id: my_fit_result = res
    #         if res.metrics["attacking"]: all_attacking.append(res)

    #     # With all attacking results and current client's results
    #     # compute the collusion update only if the current client
    #     # was attacking in current round.
    #     if my_fit_result.metrics["attacking"]:
    #         num_examples_total = sum(res.num_examples for res in all_attacking)
    #         weighted_weights = [res.num_examples * res.parameters for res in all_attacking]
    #         weights_prime = reduce(torch.add, weighted_weights) / num_examples_total
    #         del my_fit_result.parameters
    #         my_fit_result.parameters = weights_prime

    #     return my_fit_result

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round, even
        # then perform with a specified probability.
        server_round = int(fit_ins.config["server_round"])
        attack_rng = np.random.default_rng(
            derive_seed(int(fit_ins.config.get("base_seed", 0)), self.client_id, server_round, salt=1)
        )
        attack = attack_rng.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)
        
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)
        fit_results.metrics["attacking"] = True

        # Flip Gradient Signs
        # Correct Update  : Update = New_Model - Old_Model
        # Flipped Update  : New_Model = Old_Model - Update
        update = fit_results.parameters - fit_ins.parameters
        fit_results.parameters = fit_ins.parameters - self.scale_factor * update

        return fit_results
