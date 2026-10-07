"""Federated Bayesian aggregation strategy."""

from logging import DEBUG

from fedml.utils.logger import log
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_bayesian


class FederatedBayesian(FederatedAverage):

    def __init__(self, *, variation: str = "v1", **kwargs) -> None:
        super().__init__(**kwargs)
        self.robust_aggregation_variation = variation
        log(DEBUG, f"Building {self} Aggregation Strategy with variation: {variation}")

    def __repr__(self) -> str:
        return "FederatedBayesian"

    def _aggregate_weights(self, weights_results, **kwargs):
        parameters_aggregated, weight_pi, _, rho_hat = aggregate_bayesian(
            weights_results, version=self.robust_aggregation_variation
        )
        return parameters_aggregated, weight_pi, rho_hat

    def aggregate_fit(self, server_round, results, failures, selected=None):
        """Override to pass weight_pi as extra metric kwarg."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = self._filter_results(results, selected)
        parameters_aggregated, weight_pi, rho_hat = (
            self._aggregate_weights(weights_results) if weights_results
            else (None, None, None)
        )

        metrics_aggregated = self._build_metrics(
            server_round, results, selected, parameters_aggregated,
            weight_pi=weight_pi.detach().cpu().numpy() if weight_pi is not None else None,
            rho_hat=float(rho_hat) if rho_hat is not None else None,
        )

        return parameters_aggregated, metrics_aggregated