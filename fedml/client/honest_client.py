"""Honest FL client — standard training with no parameter modifications."""

from typing import Optional, Callable
from torch.utils.data import Dataset

from fedml.client import BaseClient


class HonestClient(BaseClient):
    """An honest federated learning client.

    Performs standard local training and returns updated parameters
    without any modification.
    """

    @property
    def client_type(self):
        """Returns current client's type."""
        return "HONEST"