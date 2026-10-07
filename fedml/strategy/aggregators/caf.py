"""Implementation of Covariance-bound Agnostic Filter (CAF) strategy."""

from logging import DEBUG, WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple

from fedml.utils.logger import log
from fedml.utils.typing import Parameters, Scalar
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_caf

class FederatedCAF(FederatedAverage):
    """Covariance-bound Agnostic Filter (CAF) strategy.

    Implementation based on Allouah et al.,
    "Towards Trustworthy Federated Learning with Untrusted Participants", 2025.
    """

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        max_iters: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.max_iters = max_iters
        log(DEBUG, f"Building {self} Aggregation Strategy with num_malicious_clients: {num_malicious_clients}, max_iters: {max_iters}")

    def __repr__(self) -> str:
        return "FederatedCAF"

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected: Optional[List[int]] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using CAF."""
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
            aggregate_caf(
                weights_results,
                num_malicious=self.num_malicious_clients,
                max_iters=self.max_iters,
            )
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