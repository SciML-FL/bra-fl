"""Federated Trimmed Average aggregation strategy."""

from logging import DEBUG

from fedml.utils.logger import log
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_trimmed_average


class FederatedTrimmedAverage(FederatedAverage):

    def __init__(self, *, beta: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        log(DEBUG, f"Building {self} Aggregation Strategy with beta: {beta}")

    def __repr__(self) -> str:
        return "FederatedTrimmedAverage"

    def _aggregate_weights(self, weights_results, **kwargs):
        return aggregate_trimmed_average(weights_results, self.beta)