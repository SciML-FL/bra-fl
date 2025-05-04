"""ClientManager."""


import math
import random
import threading
from abc import ABC, abstractmethod
from logging import INFO, DEBUG
from typing import Optional

from fedml.common import log
from .criterion import Criterion

class ClientManager(ABC):
    """Abstract base class for managing FedML clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """

    @abstractmethod
    def register(self, client) -> bool:
        """Register FedML Client instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if Client is
            already registered or can not be registered for any reason.
        """

    @abstractmethod
    def unregister(self, client) -> None:
        """Unregister FedML Client instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client
        """

    @abstractmethod
    def all(self):
        """Return all available clients."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients = {}

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def register(self, client) -> bool:
        """Register FedML Client instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if Client is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client

        return True

    def unregister(self, client) -> None:
        """Unregister FedML Client instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.Client
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

    def all(self):
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]


class DynamicAdversarialManager(SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(self, max_malicious_ratio=0.0, min_malicious_ratio = 0.0) -> None:
        super().__init__()
        self.min_malicious_ratio = min_malicious_ratio
        self.max_malicious_ratio = max_malicious_ratio

    def sample(self, num_clients, criterion = None, eval_round = False):
        clients = super().sample(num_clients, criterion, eval_round)
        if not eval_round:
            min_malicious: int = int(self.min_malicious_ratio*num_clients) #  0
            max_malicious: int = int(self.max_malicious_ratio*num_clients) # 10
            num_malicious: int = random.randint(min_malicious, max_malicious) # 7

            for client in clients:
                if client.client_type == "HONEST": continue
                if (num_malicious > 0):
                    client.attack_config["ATTACK_RATIO"] = 1.0
                    num_malicious -= 1
                else:
                    client.attack_config["ATTACK_RATIO"] = 0.0       
        return clients


class AdversarialClientManager(SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(self, malicious_ratio, strict=False) -> None:
        super().__init__()
        self.malicious_ratio = malicious_ratio
        self.strict = strict

    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""
        if eval_round:
            return super().sample(num_clients=num_clients, criterion=criterion, eval_round=eval_round)
        else:
            # Sample clients which meet the criterion
            available_cids = list(self.clients)
            if criterion is not None:
                available_cids = [
                    cid for cid in available_cids if criterion.select(self.clients[cid])
                ]

            if num_clients > len(available_cids):
                log(
                    INFO,
                    "Sampling failed: number of available clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(available_cids),
                    num_clients,
                )
                return []

            # Separate selected clients in honest vs. malicious set
            honest_cids, malicious_cids = [], []
            for cid in available_cids:
                if self.clients[cid].client_type == "HONEST":
                    honest_cids.append(cid)
                else:
                    malicious_cids.append(cid)

            num_malicious = math.ceil(self.malicious_ratio * num_clients)
            if num_malicious > len(malicious_cids) and self.strict:
                log(
                    INFO,
                    "Sampling failed: number of available honest clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(malicious_cids),
                    num_malicious,
                )
                return []
            elif num_malicious > len(malicious_cids):
                num_malicious = len(malicious_cids)

            num_honest = num_clients - num_malicious
            if num_honest > len(honest_cids):
                log(
                    INFO,
                    "Sampling failed: number of available honest clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(honest_cids),
                    num_honest,
                )
                return []

            sampled_cids = random.sample(honest_cids, num_honest) + random.sample(malicious_cids, num_malicious)
            return [self.clients[cid] for cid in sampled_cids]


class HonestClientManager(SimpleClientManager):
    """Provides a pool of available clients."""

    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""
        if eval_round:
            return super().sample(num_clients=num_clients, criterion=criterion, eval_round=eval_round)
        else:
            # Sample clients which meet the criterion
            available_cids = list(self.clients)
            if criterion is not None:
                available_cids = [
                    cid for cid in available_cids if criterion.select(self.clients[cid])
                ]

            # Extract only honest clients from available clients
            honest_cids = []
            for cid in available_cids:
                if self.clients[cid].client_type == "HONEST":
                    honest_cids.append(cid)

            if num_clients > len(honest_cids):
                log(
                    INFO,
                    "Sampling failed: number of available honest clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(honest_cids),
                    num_clients,
                )
                return []

            sampled_cids = random.sample(honest_cids, num_clients)
            return [self.clients[cid] for cid in sampled_cids]


class UpperboundAdversarial(SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(self, upperbound, strict=False) -> None:
        super().__init__()
        self.upperbound = upperbound
        self.strict = strict

    def sample(
        self,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        eval_round: bool = False,
    ):
        """Sample a number of FedML Client instances."""
        if eval_round:
            return super().sample(num_clients=num_clients, criterion=criterion, eval_round=eval_round)
        else:
            # Sample clients which meet the criterion
            available_cids = list(self.clients)
            if criterion is not None:
                available_cids = [
                    cid for cid in available_cids if criterion.select(self.clients[cid])
                ]

            if num_clients > len(available_cids):
                log(
                    INFO,
                    "Sampling failed: number of available clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(available_cids),
                    num_clients,
                )
                return []

            sampled_cids = random.sample(available_cids, num_clients)
            
            # Separate selected clients into Malicious vs. Honest clients
            honest_indices, malicious_indices = [], []
            for index, cid in enumerate(sampled_cids):
                if self.clients[cid].client_type == "HONEST":
                    honest_indices.append(index)
                else:
                    malicious_indices.append(index)

            # Check the upperbound criteria
            if len(honest_indices) <= len(malicious_indices):
                # Compute additional number of honest clients
                # needed to fullfill the upperbound criteria
                additional_honest = len(malicious_indices) - len(honest_indices)
                if additional_honest == 0: additional_honest += 1

                # Separate selected clients in honest vs. malicious set
                honest_cids, malicious_cids = [], []
                for cid in available_cids:
                    if self.clients[cid].client_type == "HONEST":
                        honest_cids.append(cid)
                    else:
                        malicious_cids.append(cid)
                
                # Sample additional honest clients
                sampled_honest_cids = random.sample(honest_cids, additional_honest)

                # Randomly sample some of the malicious clients to replace
                to_replace_indices = random.sample(malicious_indices, additional_honest)

                # Make the replacement
                for i, index in enumerate(to_replace_indices):
                    sampled_cids[index] = sampled_honest_cids[i]

            return [self.clients[cid] for cid in sampled_cids]


def get_client_manager(user_configs: dict):
    server_configs = user_configs["SERVER_CONFIGS"]
    if server_configs["CLIENTS_MANAGER"] == "ADVERSARIAL":
        return AdversarialClientManager(malicious_ratio=user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"])
    elif server_configs["CLIENTS_MANAGER"] == "SIMPLE":
        return SimpleClientManager()
    elif server_configs["CLIENTS_MANAGER"] == "HONEST":
        return HonestClientManager()
    elif server_configs["CLIENTS_MANAGER"] == "UPPERBOUND":
        return UpperboundAdversarial(upperbound=0.5)
    elif server_configs["CLIENTS_MANAGER"] == "DYNAMIC-ADVERSARIAL":
        return DynamicAdversarialManager(max_malicious_ratio=user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"])
    else:
        raise ValueError(f"Undefined client manager type {server_configs['CLIENTS_MANAGER']}")