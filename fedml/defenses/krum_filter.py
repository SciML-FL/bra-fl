"""Krum-based defense filter."""

from typing import Dict, List, Optional, Tuple

import torch

from fedml.utils.typing import Parameters
from fedml.strategy.aggregators.aggregate import _compute_distances

from fedml.defenses.base_filter import Filter


class KrumFilter(Filter):
    """Selects client updates using the Krum or Multi-Krum scoring rule.

    Computes pairwise distances between updates and selects the client(s)
    whose updates are closest to the majority — robust to Byzantine clients.
    """

    def __init__(self, num_malicious_clients: int, num_clients_to_keep: int, **kwargs) -> None:
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep

    @property
    def filter_type(self) -> str:
        return "KRUM"

    def server_tasks(self, global_weights: Parameters, server_round: int) -> None:
        """No server-side preparation needed for Krum."""
        return

    def filter_updates(
        self,
        client_weights: List[Tuple[Parameters, int]],
        server_round: int,
    ) -> Tuple[List[int], Optional[Dict]]:
        """Select updates using Krum or Multi-Krum scoring."""
        weights_list = [w for w, _ in client_weights]
        distance_matrix = _compute_distances(weights_list)

        num_closest = max(1, len(weights_list) - self.num_malicious_clients - 2)
        sorted_indices = torch.argsort(distance_matrix, dim=1)
        scores = torch.sum(
            distance_matrix.gather(1, sorted_indices[:, 1:(num_closest + 1)]),
            dim=1,
        )

        if self.num_clients_to_keep > 0:
            # Multi-Krum: return top-k clients by score
            best_indices = torch.argsort(scores, descending=False)[:self.num_clients_to_keep]
        else:
            # Krum: return single best client
            best_indices = [torch.argmin(scores).item()]

        stats = {"distances": distance_matrix}
        return best_indices, stats