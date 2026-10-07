"""Mean-distance defense filter."""

from typing import List, Tuple

import torch

from fedml.utils.typing import Parameters
from fedml.strategy.aggregators.aggregate import _flatten_weights

from fedml.defenses.base_filter import Filter


class MeanFilter(Filter):
    """Selects client updates within a norm distance threshold from the mean.

    Computes the mean of all updates and rejects clients whose update
    deviates beyond a configured norm limit.
    """

    def __init__(self, filter_configs: dict) -> None:
        self.filter_configs = filter_configs

    @property
    def filter_type(self) -> str:
        return "MEAN"

    def server_tasks(self, global_weights: Parameters, server_round: int) -> None:
        """No server-side preparation needed for mean filtering."""
        return

    def filter_updates(
        self,
        client_weights: List[Tuple[Parameters, int]],
        server_round: int,
    ) -> Tuple[List[int], None]:
        """Select updates within norm_limit distance from the mean update."""
        weights_list = [w for w, _ in client_weights]
        flat_weights = _flatten_weights(weights_list)

        weights_mean = torch.mean(flat_weights, dim=0)
        norm_distances = torch.linalg.norm(flat_weights - weights_mean, ord=2, dim=1)

        selected = torch.argwhere(
            norm_distances < self.filter_configs["NORM_LIMIT"]
        ).squeeze()

        return selected, None