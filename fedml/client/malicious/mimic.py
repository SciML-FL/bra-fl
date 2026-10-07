"""Implementation of Honest Client using FedML Framework"""

import copy
from functools import reduce
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from fedml.utils.typing import (
    FitIns,
    FitRes,
)

from fedml.client import BaseClient
from fedml.utils.random import derive_seed


class MIMICClient(BaseClient):
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

    @property
    def client_type(self):
        """Returns current client's type."""
        return "MIMIC"

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

        # If attacking current round then copy the model (mimic)
        # of one specific honest client (first one).
        if my_fit_result.metrics["attacking"]:
            # Replace exisitng model with new malicious update
            if self.process:
                del my_fit_result.parameters
            my_fit_result.parameters = all_clean[0].detach().clone()

        return my_fit_result

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
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
