"""Implementation of Honest Client using FedML Framework"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from typing import Optional, Dict, Callable
from fedml.utils.typing import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)
from fedml.data import CustomSubset, merge_splits
from fedml.client import BaseClient
from fedml.utils.random import derive_seed


class LabelFlippingClient(BaseClient):
    """A malicious client peforming targeted label flipping attack.
    
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
        self.scale_factor = self.attack_config["LABELFLIP_CONFIG"]["SCALE_FACTOR"]
        self.poisoned_trainset, self.poisoned_testset, self.concate_trainset, self.concate_testset = self.flip_labels()
        self.version = "v1"
        if "FLIP_VERSION" in self.attack_config["LABELFLIP_CONFIG"].keys():
            self.version = self.attack_config["LABELFLIP_CONFIG"]["FLIP_VERSION"] 

    @property
    def client_type(self):
        """Returns current client's type."""
        return "LABELFLIP"

    def flip_labels(self):
        """Perform some sort of data manipulation to create a specific target model."""
        temp_trainset = copy.deepcopy(self.trainset)
        temp_testset = copy.deepcopy(self.testset)

        train_indices, train_labels = [], []
        test_indices,  test_labels  = [], []

        for item in self.attack_config["LABELFLIP_CONFIG"]["TARGETS"]:
            source_label = item["SOURCE_LABEL"]
            target_label = item["TARGET_LABEL"]

            # Collect indices where oTargets matches the source label
            tr_indices = torch.where(self.trainset.oTargets == source_label)[0].tolist()
            train_indices.extend(tr_indices)
            train_labels.extend([target_label] * len(tr_indices))

            ts_indices = torch.where(self.testset.oTargets == source_label)[0].tolist()
            test_indices.extend(ts_indices)
            test_labels.extend([target_label] * len(ts_indices))

            # Mixed datasets: flip labels for matched samples
            temp_trainset.targets[temp_trainset.oTargets == source_label] = target_label
            temp_testset.targets[temp_testset.oTargets == source_label]   = target_label

        # CustomSubset handles both tensor-based (CIFAR) and lazy/path-based (ImageNet) datasets
        poisoned_trainset = CustomSubset(self.trainset, train_indices, labels=train_labels)
        poisoned_testset  = CustomSubset(self.testset,  test_indices,  labels=test_labels)

        return poisoned_trainset, poisoned_testset, temp_trainset, temp_testset

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
        # print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Only flip labels after specific round number
        server_round = int(fit_ins.config["server_round"])
        attack_rng = np.random.default_rng(
            derive_seed(int(fit_ins.config.get("base_seed", 0)), self.client_id, server_round, salt=1)
        )
        attack = attack_rng.random() < self.attack_config["ATTACK_RATIO"]

        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
            return super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Add malicious epoch count and learning rate
        if "LOCAL_EPOCHS" in self.attack_config["LABELFLIP_CONFIG"].keys() and self.attack_config["LABELFLIP_CONFIG"]["LOCAL_EPOCHS"] is not None:
            fit_ins.config["epochs"] = self.attack_config["LABELFLIP_CONFIG"]["LOCAL_EPOCHS"]
        if "LEARN_RATE" in self.attack_config["LABELFLIP_CONFIG"].keys() and self.attack_config["LABELFLIP_CONFIG"]["LEARN_RATE"] is not None:
            fit_ins.config["learning_rate"] = self.attack_config["LABELFLIP_CONFIG"]["LEARN_RATE"]

        # Perfrom attacked / malicious training
        if self.version == "v1":
            fit_results = self.attack_version_1(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)
        elif self.version == "v2":
            fit_results = self.attack_version_2(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Setup other metrics and perform scaling if required.
        fit_results.metrics["attacking"] = True

        # Scale the updated model if requested
        # Update          : New_Model = Old_Model ± Gradient
        # Gradients       : ± Gradient = New_Model - Old_Model
        if self.scale_factor != 1.0:
            gradients = fit_results.parameters - fit_ins.parameters
            del fit_results.parameters
            fit_results.parameters = fit_ins.parameters + (self.scale_factor * gradients)

        return fit_results

    def attack_version_1(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
        # Replace benign dataset with poisoned dataset (with labels flipped)
        org_trainset, org_testset = self.trainset, self.testset
        self.trainset, self.testset = self.concate_trainset, self.concate_testset

        # Train using the malicious dataset
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Revert the datasets back to original state
        self.trainset, self.testset = org_trainset, org_testset

        return fit_results

    def attack_version_2(self, fit_ins: FitIns, local_model, model_as_fn, override_device: str = None) -> FitRes:
        # Replace benign dataset with poisoned dataset (with labels flipped)
        org_trainset, org_testset = self.trainset, self.testset
        self.trainset, self.testset = self.poisoned_trainset, self.poisoned_testset

        # Perform training on malicious (label-flipped) dataset
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Perform re-calibration with mixed (clean + poisoned) dataset
        self.trainset, self.testset = self.concate_trainset, self.concate_testset
        fit_ins.parameters = fit_results.parameters
        fit_results = super().fit(fit_ins=fit_ins, local_model=local_model, model_as_fn=model_as_fn, override_device=override_device)

        # Revert the datasets back to original state
        self.trainset, self.testset = org_trainset, org_testset

        return fit_results

    def evaluate(self, eval_ins: EvaluateIns, local_model, model_as_fn, override_device: str = None) -> EvaluateRes:
        # Create fresh model instance and set weights to ensure clean state for attack
        device= override_device if override_device is not None else self.run_device
        model = local_model() if model_as_fn else local_model

        # Compute base class evaluations
        eval_results = super().evaluate(eval_ins=eval_ins, local_model=model, model_as_fn=False, override_device=device)
        # model = model.to(device)
        model.eval()

        if self.version == "v2":                # Since v1 is untargeted label flip
            # Compute the attack success rate
            # ORG - NEW - TOTAL - CLASSIFIED_ORG - CLASSIFIED_NEW
            tr_new_label = 0
            tr_total_samples = 0
            ts_new_label = 0
            ts_total_samples = 0

            #############################################################
            #############################################################
            # Evaluate Train Set for ASR
            #############################################################
            #############################################################
            tr_loader = DataLoader(self.poisoned_trainset, batch_size=256, shuffle=False)
            with torch.no_grad():
                for sample, target in tr_loader:
                    sample, target = sample.to(device), target.to(device)
                    outputs = model(sample)
                    _, predicted = torch.max(outputs.data, 1)  
                    tr_total_samples += target.size(0)
                    tr_new_label += (predicted == target).sum().item()

            #############################################################
            #############################################################
            # Evaluate Test Set for ASR
            #############################################################
            #############################################################
            ts_loader = DataLoader(self.poisoned_testset, batch_size=256, shuffle=False)
            with torch.no_grad():
                for sample, target in ts_loader:
                    sample, target = sample.to(device), target.to(device)
                    outputs = model(sample)
                    _, predicted = torch.max(outputs.data, 1)  
                    ts_total_samples += target.size(0)
                    ts_new_label += (predicted == target).sum().item()

            eval_results.metrics["train_samples"] = tr_total_samples
            eval_results.metrics["train_success"] =  tr_new_label
            eval_results.metrics["train_asr"] =  tr_new_label / tr_total_samples
            eval_results.metrics["test_samples"] = ts_total_samples
            eval_results.metrics["test_success"] =  ts_new_label
            eval_results.metrics["test_asr"] =  ts_new_label / ts_total_samples

        return eval_results
