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


class ALIEClient(BaseClient):
    """A malicious client submitting random updates (noise)."""

    def __init__(
        self,
        client_id: int,
        trainset: Dataset,
        testset: Dataset,
        batch_train: bool = False,
        process: bool = True,
        attack_config: Optional[Dict] = None,
        run_device: Optional[str] = None,
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

    @property
    def client_type(self):
        """Returns current client's type."""
        return "ALIE"

    def post_training_callback(self, results, failures):
        my_fit_result = None
        all_attacking = []
        all_clean = []
        for _, res in results:
            if res.metrics["client_id"] == self.client_id: 
                my_fit_result = res
            if res.metrics["attacking"]: 
                all_attacking.append(res.parameters)
            else:
                all_clean.append(res.parameters)
        num_clients = len(results)
        num_malicious = len(all_attacking)

        # With all attacking results and current client's results
        # compute the collusion update only if the current client
        # was attacking in current round.
        if my_fit_result.metrics["attacking"] and len(all_clean) > 0:
            s = torch.floor_divide(num_clients, 2) + 1 - num_malicious
            cdf_value = (num_clients - num_malicious - s) / (num_clients - num_malicious)
            dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
            z_max = dist.icdf(cdf_value)

            stacked_models = torch.stack(all_clean, 1)
            mean = torch.mean(stacked_models, 1)
            std = torch.std(stacked_models, 1)

            # Compute malicious update
            # z_max = 1.0         ## Remove later, just testing for now
            # print(f"{self.client_id} -- {z_max}", flush=True)

            malicious_model = mean - (std * z_max)

            # Replace exisitng model with new malicious update
            if self.process:
                del my_fit_result.parameters
            my_fit_result.parameters = malicious_model.detach().clone()

        return my_fit_result

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: Optional[str] = None) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round
        server_round = int(fit_ins.config["server_round"])
        attack_rng = np.random.default_rng(
            derive_seed(int(fit_ins.config.get("base_seed", 0)), self.client_id, server_round, salt=1)
        )
        attack = attack_rng.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Even when attacking, send back clean model as
        # attack actually happens in post training callback
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)
        fit_results.metrics["attacking"] = True
        return fit_results     
