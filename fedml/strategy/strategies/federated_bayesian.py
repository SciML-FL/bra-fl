"""Implementation of Federated Average (FedAvg) strategy."""

from logging import WARNING, DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union

from fedml.common import (
    MetricsAggregationFn,
    Parameters,
    Scalar,
    log
)
from .federated_average import FederatedAverage
from .aggregate import aggregate_bayesian


class FederatedBayesian(FederatedAverage):
    def __init__(
        self,
        *,
        local_models: List[any], 
        run_devices: List[str],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, Parameters, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        variation: str = "v1",
        **kwargs,
    ) -> None:
        super().__init__(
            local_models=local_models,
            run_devices=run_devices,
            fraction_fit = fraction_fit,
            fraction_evaluate = fraction_evaluate,
            min_fit_clients = min_fit_clients,
            min_evaluate_clients = min_evaluate_clients,
            min_available_clients = min_available_clients,
            evaluate_fn = evaluate_fn,
            on_fit_config_fn = on_fit_config_fn,
            on_evaluate_config_fn = on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters = initial_parameters,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn,         
        )
        self.robust_aggregation_variation = variation
        log(
            DEBUG,
            f"Buidling Robust Aggregation Strategy with Variation {self.robust_aggregation_variation}"
        )

    def __repr__(self) -> str:
        return "FederatedBayesian"

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected: Optional[List[int]] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []
        for indx, (_, fit_res) in enumerate(results):
            if selected is None:
                weights_results.append((fit_res.parameters, fit_res.num_examples))
            elif indx in selected:
                weights_results.append((fit_res.parameters, fit_res.num_examples))   
        # weights_results = [(fit_res.parameters, fit_res.num_examples) for _, fit_res in results]

        parameters_aggregated, weight_pi, _ = aggregate_bayesian(weights_results, version=self.robust_aggregation_variation)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                fit_metrics=fit_metrics, 
                selected=selected,
                weight_pi=weight_pi.detach().cpu().numpy(),
                update_norms=self.compute_benign_norms(
                    results=results,
                    aggregated_parameters=parameters_aggregated
                )
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
