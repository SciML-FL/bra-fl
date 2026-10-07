"""Implementation of Honest Client using FedML Framework"""

import copy
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


class IPMClient(BaseClient):
    """
    A malicious client submitting IPM-based update.
    We use an adaptive choice of the scaling factor (gamma) according to the paper:
    Shejwalkar, Virat, and Amir Houmansadr. "Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning." NDSS. 2021.
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
        self.gamma_init = attack_config["IPM_CONFIG"]["GAMMA_INIT"]
        self.version = attack_config["IPM_CONFIG"]["IPM_VERSION"]  # "minsum"

    @property
    def client_type(self):
        """Returns current client's type."""
        return "IPM"

    def _find_gamma_min_max(self, all_clean):
        """Find gamma using Min-Max rule from the paper"""
        # Precompute terms from condition (6) of the paper
        X = torch.stack(all_clean)
        mean = torch.mean(X, 0)
        # rhs
        X_norm_sq = (X**2).sum(dim=1)
        dist_sq = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * X @ X.T
        rhs = dist_sq.clamp_min_(0.0).sqrt().max()

        gamma = self.gamma_init
        best_gamma = self.gamma_init
        step = self.gamma_init / 2.0
        for _ in range(20):
            lhs = (X - (1 - gamma) * mean).norm(dim=1).max()
            if lhs <= rhs:
                best_gamma = gamma
                gamma += step / 2.0
            else:
                gamma -= step / 2.0
            step /= 2.0
            # Stopping criterion
            if np.abs(best_gamma - gamma) <= 0.01:
                break
        return (1 - best_gamma) * mean

    def _find_gamma_min_sum(self, all_clean):
        """Find gamma using Min-Sum rule from the paper"""
        # Precompute terms from condition (7) of the paper
        X = torch.stack(all_clean)
        mean = torch.mean(X, 0)
        mean_norm_sq = (mean**2).sum()
        # rhs
        X_norm_sq = (X**2).sum(dim=1)
        dist_sq = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * X @ X.T
        rhs = dist_sq.clamp_min_(0.0).sum(dim=1).max()
        X_norm_sq_sum = X_norm_sq.sum()
        X_dot_mean = (X @ mean).sum()

        gamma = self.gamma_init
        best_gamma = self.gamma_init
        step = self.gamma_init / 2.0
        for _ in range(20):
            c = 1 - gamma
            lhs = c**2 * X.size(0) * mean_norm_sq + X_norm_sq_sum - 2 * c * X_dot_mean
            if lhs <= rhs:
                best_gamma = gamma
                gamma += step / 2.0
            else:
                gamma -= step / 2.0
            step /= 2.0
            # Stopping criterion
            if np.abs(best_gamma - gamma) <= 0.01:
                break
        return (1 - best_gamma) * mean

    def post_training_callback(self, results, failures):
        my_fit_result = None
        all_clean = []
        for _, res in results:
            if res.metrics["client_id"] == self.client_id:
                my_fit_result = res
            if not res.metrics["attacking"]:
                all_clean.append(res.parameters)

        if my_fit_result.metrics["attacking"] and len(all_clean) > 0:
            # IPM with the optimal scaling factor gamma
            if self.version == "minmax":
                ipm_model = self._find_gamma_min_max(all_clean)
            elif self.version == "minsum":
                ipm_model = self._find_gamma_min_sum(all_clean)
            else:
                raise ValueError(
                    "IPM version must be `minmax` or `minsum`;"
                    + f" instead {self.version} was given"
                )

            # Replace existing model with new malicious update
            if self.process:
                del my_fit_result.parameters
            my_fit_result.parameters = ipm_model.detach().clone()

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
