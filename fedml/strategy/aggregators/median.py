"""Federated Median aggregation strategy."""

from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_median



class FederatedMedian(FederatedAverage):

    def __repr__(self) -> str:
        return "FederatedMedian"

    def _aggregate_weights(self, weights_results, **kwargs):
        return aggregate_median(weights_results)