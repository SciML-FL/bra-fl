"""Abstract base strategy with shared default implementations."""

from logging import DEBUG

from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Dict, Tuple

from fedml.utils.typing import (
    Parameters, 
    Scalar,
    EvaluateIns,
    FitIns,
)

from fedml.utils.logger import log


class Strategy(ABC):
    """Abstract base class for FL server strategies.

    Provides default implementations for configure_fit and configure_evaluate
    since these are identical across all strategies. Subclasses must implement
    aggregate_fit, aggregate_evaluate, evaluate, and initialize_parameters.
    """

    def __init__(
        self,
        *,
        local_models: List,
        model_as_fn: bool,
        run_devices: List[str],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable] =None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        **kwargs, # Accept extra kwargs for flexibility, e.g. strategy-specific args from config
    ) -> None:
        # Imported lazily to avoid a strategy/server import cycle when an
        # aggregation helper is tested in isolation.
        from fedml.server.criterion import MaliciousSampling

        self.local_models = local_models
        self.model_as_fn = model_as_fn
        self.run_devices = run_devices
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.evaluate_fn = evaluate_fn

        # if len(self.local_models) != len(self.run_devices) or len(self.local_models) != self.min_fit_clients:
        #     raise ValueError(
        #         "Number of local models must match number of run devices."
        #     )

        self.fit_criterion = None
        self.evaluate_criterion = MaliciousSampling()

        log(DEBUG, f"Initialized {self} Aggregation Strategy.")

    # ------------------------------------------------------------------
    # Concrete shared implementations
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Release from memory after use
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, _ = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            criterion=self.fit_criterion,
            eval_round=False,
        )

        return [(client, fit_ins, self.local_models[i], self.model_as_fn, self.run_devices[i]) for i, client in enumerate(clients)]
        # return [(client, fit_ins, self.local_models[client.client_id], self.model_as_fn, self.run_devices[i]) for i, client in enumerate(clients)]


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, _ = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            criterion=self.evaluate_criterion,
            eval_round=True,
        )

        return [(client, evaluate_ins, self.local_models[i], self.model_as_fn, self.run_devices[i]) for i, client in enumerate(clients)]
        # return [(client, evaluate_ins, self.local_models[client.client_id], self.model_as_fn, self.run_devices[i]) for i, client in enumerate(clients)]


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
        eval_res = self.evaluate_fn(server_round, parameters, {})
        if eval_res is None:
            return None
        return eval_res

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected=None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
