"""Implementation of Honest Client using FedML Framework"""

import timeit

import copy
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Optional, Dict
from fedml.common import (
    Code,
    FitIns,
    FitRes,
    Status,
)

from .honest_client import HonestClient

class ModelReplacementClient(HonestClient):
    """A malicious client performing model replacement attack.
    
    """
    def __init__(
            self, 
            client_id: int,
            trainset: Dataset,
            testset: Dataset,
            process: bool = True,
            attack_config: Optional[Dict] = None,
            ) -> None:
        """Initializes a new honest client."""
        super().__init__(
            client_id=client_id,
            trainset=trainset,
            testset=testset,
            process=process,
        )
        self.attack_config = copy.deepcopy(attack_config)
        self.pretrained_weights = torch.load(self.attack_config["MPAF_CONFIG"]["TARGET_MODEL"])

    @property
    def client_type(self):
        """Returns current client's type."""
        return "MPAF"

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
        # learning_rate = 0.01 #float(config["learning_rate"])
        
        fit_begin = timeit.default_timer()
        
        # Set model parameters
        model.set_weights(ins.parameters, clone=(not self._process))

        # Return the refined weights and the number of examples used for training
        parameters_updated = self.pretrained_weights
        fit_duration = timeit.default_timer() - fit_begin

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = self.perform_evaluations(model, device, trainloader=None, testloader=None)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.attack_config["MPAF_CONFIG"]["SCALE_FACTOR"],
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": True,
                "client_type": self.client_type,
            },
        )
