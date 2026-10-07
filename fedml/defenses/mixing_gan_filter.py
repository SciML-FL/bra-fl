"""Mixing GAN defense filter — combines nearest-neighbor mixing with GAN filtration."""

from logging import INFO
from typing import Dict, List, Optional, Tuple

import torch

from fedml.utils.typing import Parameters
from fedml.utils.logger import log
from fedml.strategy.aggregators.aggregate import _compute_distances, aggregate

from fedml.defenses.gan_filter import GenerativeFilter


class MixingGenerativeFilter(GenerativeFilter):
    """GAN filter with nearest-neighbor mixing pre-processing.

    Before GAN-based evaluation, each client's update is replaced by the
    average of its nearest neighbors — smoothing out outlier updates before
    the GAN discriminator evaluates them.
    """

    @property
    def filter_type(self) -> str:
        return "MixingGAN"

    def filter_updates(
        self,
        client_weights: List[Tuple[Parameters, int]],
        server_round: int,
    ) -> Tuple[List[int], Optional[Dict[str, List]]]:
        """Apply nearest-neighbor mixing then delegate to GAN filter."""
        if server_round <= self.skip_rounds:
            log(INFO, "filter_updates: Skipping filtration at server round %s", server_round)
            return list(range(len(client_weights))), None

        weights = [w for w, _ in client_weights]
        distance_matrix = _compute_distances(weights)

        nf = max(1, len(weights) // 2)
        sorted_indices = torch.argsort(distance_matrix, dim=1)

        # Replace each client's update with the mean of its nf nearest neighbors
        weights_mixed = [
            (aggregate([client_weights[i] for i in row[:nf]]), 1)
            for row in sorted_indices
        ]

        return super().filter_updates(
            client_weights=weights_mixed,
            server_round=server_round,
        )