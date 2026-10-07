"""Implementation of Mean Around Median (Meamed) strategy."""

from logging import DEBUG, WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple

from fedml.utils.logger import log
from fedml.utils.typing import Parameters, Scalar
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_meamed

class FederatedMeamed(FederatedAverage):
    """Mean Around Median (Meamed) strategy.

    Implementation based on Xie et al., "Generalized Byzantine-tolerant SGD", 2018.
    """

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        log(DEBUG, f"Building {self} Aggregation Strategy with num_malicious_clients: {num_malicious_clients}")

    def __repr__(self) -> str:
        return "FederatedMeamed"

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected: Optional[List[int]] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Mean Around Median."""
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

        parameters_aggregated = (
            aggregate_meamed(weights_results, self.num_malicious_clients)
            if len(weights_results) > 0
            else None
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                fit_metrics=fit_metrics,
                selected=selected,
                update_norms=self.compute_benign_norms(
                    results=results, aggregated_parameters=parameters_aggregated
                ),
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated