"""Federated Geometric Median aggregation strategy."""

from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_geometric_median


class FederatedGeometricMedian(FederatedAverage):

    def __repr__(self) -> str:
        return "FederatedGeometricMedian"

    def _aggregate_weights(self, weights_results, **kwargs):
        parameters_aggregated, geo_weights = aggregate_geometric_median(weights_results)
        return parameters_aggregated, geo_weights

    def aggregate_fit(self, server_round, results, failures, selected=None):
        """Override to pass geo_weights as extra metric kwarg."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = self._filter_results(results, selected)
        parameters_aggregated, geo_weights = (
            self._aggregate_weights(weights_results) if weights_results
            else (None, None)
        )

        metrics_aggregated = self._build_metrics(
            server_round, results, selected, parameters_aggregated,
            weight_pi=geo_weights.cpu().detach().numpy() if geo_weights is not None else None,
        )

        return parameters_aggregated, metrics_aggregated