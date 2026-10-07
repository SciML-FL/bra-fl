"""Federated Bulyan aggregation strategy."""

from typing import Callable

from logging import DEBUG

from fedml.utils.logger import log
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_krum, aggregate_bulyan


class FederatedBulyan(FederatedAverage):

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        first_aggregation_rule: Callable = aggregate_krum,
        **kwargs,
    ) -> None:
        # Separate out aggregation_rule_kwargs from base kwargs
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.first_aggregation_rule = first_aggregation_rule
        log(DEBUG, f"Building {self} Aggregation Strategy with num_malicious_clients: {num_malicious_clients}, first_aggregation_rule: {first_aggregation_rule.__name__}")

    def __repr__(self) -> str:
        return "FederatedBulyan"

    def _aggregate_weights(self, weights_results, **kwargs):
        return aggregate_bulyan(
            weights_results,
            self.num_malicious_clients,
            self.first_aggregation_rule,
        )