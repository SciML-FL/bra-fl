"""Abstract base class for all FL defense filters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from fedml.utils.typing import Parameters


class Filter(ABC):
    """Base class for all client update filters.

    All filters must implement filter_updates() and server_tasks().
    Filters that don't require server-side preparation should implement
    server_tasks() as a no-op.
    """

    @property
    def filter_type(self) -> str:
        """Type string for this filter. Subclasses should override."""
        return "BASE"

    @abstractmethod
    def filter_updates(
        self,
        client_weights: List[Tuple[Parameters, int]],
        server_round: int,
    ) -> Tuple[List[int], Optional[Dict]]:
        """Select which client updates to include in aggregation.

        Parameters
        ----------
        client_weights:
            List of (parameters, num_examples) tuples from each client.
        server_round:
            Current federated learning round.

        Returns
        -------
        selected_indices:
            Indices of clients whose updates should be aggregated.
        client_stats:
            Optional dict of per-client diagnostics for logging/plotting.
        """

    @abstractmethod
    def server_tasks(
        self,
        global_weights: Parameters,
        server_round: int,
    ) -> None:
        """Perform server-side tasks that can run in parallel to client training.

        Useful for GAN-based filters that need to train a generator
        concurrently with client training. No-op for simpler filters.
        """