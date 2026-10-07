"""Federated Mixing aggregation strategy."""

from logging import DEBUG

from fedml.utils.logger import log
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_mixing


class FederatedMixing(FederatedAverage):

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        num_clients_to_keep: int = 0,
        aggregator_to_use: str = "KRUM",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        self.aggregator_to_use = aggregator_to_use
        log(DEBUG, f"Building {self} Aggregation Strategy with num_malicious_clients: {num_malicious_clients}, num_clients_to_keep: {num_clients_to_keep}, aggregator_to_use: {aggregator_to_use}")

    def __repr__(self) -> str:
        return "FederatedMixing"  # fixed from "FederatedNNM"

    def _aggregate_weights(self, weights_results, **kwargs):
        return aggregate_mixing(
            results=weights_results,
            aggregator=self.aggregator_to_use,
            num_malicious=self.num_malicious_clients,
            to_keep=self.num_clients_to_keep,
        )