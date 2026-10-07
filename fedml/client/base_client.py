"""Implementation of Honest Client using FedML Framework"""

import gc
import gc
from typing import Optional, Callable

import itertools
import timeit

import json
import torch
from torch.utils.data import Dataset


from fedml.utils.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
)
from fedml.training import get_criterion, get_optimizer, train, evaluate, train_batch
from fedml.utils.random import derive_seed, setup_random_seeds


import os
# num_workers = min(os.cpu_count(), 8)  # cap at 8, use all cores if fewer
num_workers = 0

class BaseClient:
    """Base class for all FL clients.

    Implements standard federated training, evaluation, and communication
    routines shared across honest and malicious client types.

    Subclass patterns:
        - Honest clients: inherit as-is, just define client_type.
        - Malicious clients (parameter modification): call super().fit(),
          then modify fit_res.parameters and set fit_res.metrics["attacking"].
        - Malicious clients (custom training): override fit() entirely,
          setting metrics["attacking"] directly in their own FitRes.
    """
    def __init__(
        self, 
        client_id: int,
        trainset: Dataset,
        testset: Dataset,
        batch_train: bool = False,
        process: bool = True,
        run_device: str = "cpu",
    ) -> None:
        """Initializes a new honest client."""
        self.client_id = client_id
        self.trainset = trainset
        self.testset = testset
        self.process = process
        self.batch_train = batch_train
        self.run_device = run_device
        # self.trainloader, self.train_iterator = None, None

        # if self.default_device is not None:
        #     self.trainset.to_device(device=self.default_device)
        #     self.testset.to_device(device=self.default_device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client_id(self):
        """Returns current client's id."""
        return self._client_id
    
    @client_id.setter
    def client_id(self, client_id: int):
        self._client_id = client_id

    @property
    def cid(self):
        return self.client_id

    @property
    def client_type(self) -> str:
        """Returns this client's type string. Subclasses should override."""
        raise NotImplementedError("Subclasses must define client_type.")

    @property
    def model_fn(self) -> Optional[Callable]:
        """Zero-argument callable that constructs a fresh model instance."""
        return self._model_fn

    @model_fn.setter
    def model_fn(self, fn: Callable):
        self._model_fn = fn


    # ------------------------------------------------------------------
    # Core FL methods
    # ------------------------------------------------------------------

    def fit(self, fit_ins: FitIns, local_model, model_as_fn, override_device: Optional[str] = None) -> FitRes:
        config = fit_ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        server_round = int(fit_ins.config["server_round"])
        # total_rounds = int(fit_ins.config["total_rounds"])
        local_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        learning_rate = float(config["learning_rate"])
        optimizer_str = config["optimizer"]
        criterion_str = config["criterion"]
        optim_kwargs = dict(json.loads(config["optim_kwargs"]))
        perform_evals = config["perform_evals"]
        device = override_device if override_device is not None else self.run_device

        # Deterministic per-(client, round) seeding so this client's local
        # training is reproducible regardless of which worker process runs it
        # (spawn-ed ProcessPool workers never inherit the parent's RNG state).
        seed = derive_seed(int(config.get("base_seed", 0)), self.client_id, server_round)
        setup_random_seeds(seed)
        loader_generator = torch.Generator().manual_seed(seed)

        # Set model parameters
        model = local_model() if model_as_fn else local_model
        model.set_weights(fit_ins.parameters, clone=(not self.process))
        model.to(device)

        # Create loss function and optimizer
        criterion = get_criterion(
            criterion_str=criterion_str
        )
        optimizer = get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=model,
            learning_rate=learning_rate,
            **optim_kwargs,
        )

        # Stage dataset to GPU
        # original_device = self.trainset.data.device
        original_device = self.trainset.get_device()
        self.trainset.to_device(device=device)

        # Train model with fresh dataloader
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=num_workers, generator=loader_generator,
        )

        if not self.batch_train:
            num_examples, tr_loss, tr_accuracy = train(
                model=model, 
                trainloader=trainloader, 
                epochs=local_epochs, 
                learning_rate=learning_rate,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            del trainloader
        else:
            # Create a cyclic iterator over the trainloader
            train_iterator = itertools.cycle(trainloader)

            # Consume iterator for set number of batches. We assume
            # local rounds are same between communication rounds.
            for _ in range(local_epochs*server_round): next(train_iterator)

            num_examples, tr_loss, tr_accuracy = train_batch(
                model=model, 
                train_iterator=train_iterator,
                num_rounds=local_epochs,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

        # Perform necessary evaluations
        ts_loss, ts_accuracy = (None, None)
        if perform_evals:
            tr_loss, tr_accuracy, ts_loss, ts_accuracy = self.perform_evaluations(model, device, criterion)

        # Peforming cleanups
        self.trainset.to_device(device=original_device)

        del optimizer
        model.zero_grad(set_to_none=True)

        # Extract updated parameters (move to CPU if running as process)
        if self.process:
            model.to("cpu")

        # Extract updated parameters (move to CPU if running as process)   
        parameters_updated = model.get_weights().cpu()  # always move to CPU — safe even if already there

        if model_as_fn: 
            del model  # if model_as_fn, this is a temporary instance we created just to get the updated weights. Delete it to free memory.

        import gc
        gc.collect()               # collect Python-side refs before clearing cache
        torch.cuda.empty_cache()

        fit_duration = timeit.default_timer() - fit_begin

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_examples,
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": False,                 # malicious clients override this in their own FitRes
                "client_type": self.client_type,
            },
        )

    def perform_evaluations(self, model, device, criterion, trainloader=None, testloader=None):
        
        # Check if data loaders need to be created
        if trainloader is None:
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=1024, shuffle=False, num_workers=num_workers)

        if testloader is None:
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=1024, shuffle=False, num_workers=num_workers)

        # Perform necessary evaluations
        tr_loss, tr_accuracy, _ = evaluate(model, trainloader, device=device, criterion=criterion)
        ts_loss, ts_accuracy, _ = evaluate(model, testloader, device=device, criterion=criterion)

        # Performing cleanups
        del trainloader, testloader
        
        # Return evaluation stats
        return tr_loss, tr_accuracy, ts_loss, ts_accuracy

    def evaluate(self, eval_ins: EvaluateIns, local_model, model_as_fn, override_device: Optional[str] = None) -> EvaluateRes:
        config = eval_ins.config

        # Get training config
        server_round = int(eval_ins.config["server_round"])
        total_rounds = int(eval_ins.config["total_rounds"])
        batch_size = int(eval_ins.config["batch_size"])
        criterion_str = eval_ins.config["criterion"]
        device = override_device if override_device is not None else self.run_device

        # Use provided weights to update the local model\
        model = local_model() if model_as_fn else local_model
        model.set_weights(eval_ins.parameters, clone=(not self.process))
        model.to(device)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        criterion = get_criterion(
            criterion_str=criterion_str
        )

        # Collect return results
        loss, accuracy, num_examples = evaluate(model, testloader, device=device, criterion=criterion)

        # Performing cleanups
        del testloader
 
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples,
            metrics={
                "client_id": int(self.client_id),
                "accuracy": float(accuracy),
                "loss": float(loss),
            },
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses for partial fit modifications
    # ------------------------------------------------------------------

    def post_training_callback(self, results, failures):
        """Find and return this client's result from a round's results list.

        Useful for collusion attacks and attacks requiring access to the
        full population's updated parameters.
        """
        for _, res in results:
            if res.metrics["client_id"] == self.client_id:
                return res
        raise ValueError(f"Client {self.client_id} did not find its result in post_training_callback.")