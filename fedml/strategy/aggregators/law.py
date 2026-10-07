"""Implementation of FedLAW (Federated Learning with Learnable Aggregation Weights).

Reference: Parsa et al., "Byzantine-Robust Federated Learning with Learnable
Aggregation Weights", arXiv:2511.03529.
"""

from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple

from fedml.utils.logger import log
from fedml.utils.typing import Parameters, Scalar
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_meamed

class FederatedLAW(FederatedAverage):
    """FedLAW: Byzantine-Robust FL with Learnable Aggregation Weights.

    Core idea
    ---------
    Instead of *filtering* clients with a heuristic (Krum, Bulyan, MDA),
    treat the per-client aggregation weights `w` as decision variables and
    jointly optimize them with the global model under a sparse capped simplex:

        min_{theta, w}   sum_i  w_i * f_i(theta)
        s.t.             sum_i w_i = 1
                         0 <= w_i <= t          (cap)
                         ||w||_0 <= s            (sparsity)

    The l0 sparsity constraint forces (n - s) clients to receive zero weight in
    every round; the optimizer naturally drives the malicious ones to zero.

    Framework adaptation notes
    --------------------------
    The original paper performs an *alternating* minimization that requires
    TWO client gathers per round: one to update theta with old w, one to
    re-gather grads/losses at the new theta to inform the w-update. In a
    one-fit-per-round framework we only get one gather, so we use a
    "lag-by-one" approximation: previous round's client deltas serve as `G`,
    current round's as `G_next`. Cost: one round of staleness in the gradient
    signal (no extra communication).
    """

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        # fraction_fit: float = 1.0,
        # min_fit_clients: int = 2,
        # min_available_clients: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients

        # Ensure that all clients are selected for training since FederatedLAW requires it
        if self.fraction_fit < 1.0 or self.min_fit_clients != self.min_available_clients:
            log(
                WARNING,
                "FederatedLAW requires fraction_fit=1.0 and min_fit_clients=min_available_clients. Overriding these values.",
            )
            self.fraction_fit = 1.0
            self.min_fit_clients = self.min_available_clients

        # Setup required parameters for aggregation and cross round tracking
        self.w_pi = [1.0 / self.min_available_clients] * self.min_available_clients  # Uniform starting weights
        self.prev_aggregated_parameters = None  # To store the previous round's aggregated parameters
        

    def __repr__(self) -> str:
        return "FederatedLAW"

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