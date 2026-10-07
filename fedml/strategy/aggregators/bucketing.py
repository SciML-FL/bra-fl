"""Federated Bucketing aggregation strategy."""

from logging import DEBUG

from fedml.utils.logger import log
from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_bucketing


class FederatedBucketing(FederatedAverage):

    def __init__(
        self,
        *,
        num_malicious_clients: int = 0,
        num_clients_to_keep: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep
        log(DEBUG, f"Building {self} Aggregation Strategy with num_malicious_clients: {num_malicious_clients}, num_clients_to_keep: {num_clients_to_keep}")

    def __repr__(self) -> str:
        return "FederatedBucketing"  # fixed from "FederatedKrum"

    def _aggregate_weights(self, weights_results, **kwargs):
        return aggregate_bucketing(
            weights_results,
            self.num_malicious_clients,
            self.num_clients_to_keep,
        )