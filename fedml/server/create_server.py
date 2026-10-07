"""Factory function for creating FL server instances."""

from typing import Callable


def create_server(
        server_type: str,
        client_manager,
        strategy: Callable,
        user_configs: dict,
        executor_type: str,
        initial_parameters=None,
        experiment_manager=None,
        max_workers: int | None = None,
    ):
    """Create and return the appropriate FL server instance."""

    if server_type == "NORMAL":
        from .base_server import BaseServer
        return BaseServer(
            client_manager=client_manager,
            strategy=strategy,
            experiment_manager=experiment_manager,
            initial_parameters=initial_parameters,
            executor_type=executor_type,
            max_workers=max_workers,
        )
    elif server_type == "FILTER":
        from .filtered_server import FilteredServer
        return FilteredServer(
            client_manager=client_manager,
            strategy=strategy,
            experiment_manager=experiment_manager,
            initial_parameters=initial_parameters,
            user_configs=user_configs,
            executor_type=executor_type,
            max_workers=max_workers,
        )
    else:
        raise ValueError(f"Invalid server type '{server_type}' requested.")
