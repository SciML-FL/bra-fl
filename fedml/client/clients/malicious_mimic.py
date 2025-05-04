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


class MIMICClient(HonestClient):
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
            del my_fit_result.parameters
            my_fit_result.parameters = all_clean[0].detach().clone()

        return my_fit_result

    def fit(self, model, device, ins: FitIns) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round
        server_round = int(ins.config["server_round"])
        attack = np.random.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(model, device, ins=ins)

        # Even when attacking, send back clean model as
        # attack actually happens in post training callback
        fit_results = super().fit(model, device, ins=ins)
        fit_results.metrics["attacking"] = True
        return fit_results     
