"""Base FL server — round orchestration, evaluation, and history tracking."""

import concurrent.futures
import timeit
from logging import INFO, WARNING
from typing import Dict, Optional

from fedml.utils.typing import Scalar, Parameters
from fedml.utils.history import History
from fedml.utils.logger import log

from fedml.server.execution import fit_clients, evaluate_clients, post_training


class BaseServer:
    """Base federated learning server.

    Orchestrates the federated training loop, delegating client selection
    and aggregation to the strategy, and client execution to the executor.

    Subclasses can override fit_round() to add pre/post-aggregation logic
    such as filtering (see FilteredServer).
    """

    def __init__(
        self,
        *,
        client_manager,
        experiment_manager=None,
        strategy=None,
        user_configs: Optional[Dict] = None,
        initial_parameters=None,
        executor_type: Optional[str] = "ThreadPool",
        max_workers: int = 1,
    ) -> None:
        self.experiment_manager = experiment_manager
        self.user_configs = user_configs
        self.client_manager = client_manager
        self.strategy = strategy
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

        self.set_initial_parameters(initial_parameters=initial_parameters)

        self.executor = None
        if executor_type == "ThreadPool":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif executor_type == "ProcessPool":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

    def __del__(self):
        if self.executor:
            self.executor.shutdown()

    def set_initial_parameters(self, initial_parameters):
        """Set initial global model parameters."""
        if initial_parameters is not None:
            self.parameters = initial_parameters
        else:
            raise NotImplementedError("Parameter initialization not implemented.")

    # ------------------------------------------------------------------
    # Round logic
    # ------------------------------------------------------------------

    def fit_round(self, server_round: int):
        """Perform a single round of federated training."""
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )

        if not client_instructions:
            raise RuntimeError(f"Round {server_round}: no clients selected")

        log(INFO, "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions), self.client_manager.num_available())

        results, failures = fit_clients(
            executor=self.executor,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            group_id=server_round,
        )

        self._require_complete_results(server_round, "fit", client_instructions, results, failures)

        log(INFO, "aggregate_fit: received %s results and %s failures",
            len(results), len(failures))

        results, failures = post_training(
            executor=self.executor,
            client_instructions=client_instructions,
            results=results,
            failures=failures,
        )

        self._require_complete_results(server_round, "post-training", client_instructions, results, failures)

        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    @staticmethod
    def _require_complete_results(server_round, stage, instructions, results, failures):
        if failures or len(results) != len(instructions):
            raise RuntimeError(
                f"Round {server_round} {stage}: received {len(results)} of "
                f"{len(instructions)} results and {len(failures)} failures; "
                "refusing to record a partial-client experiment as complete"
            )

    def evaluate_round(self, server_round: int):
        """Validate current global model on a sample of clients."""
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )

        if not client_instructions:
            log(WARNING, "configure_evaluate: no clients selected, skipping evaluation")
            return None

        log(INFO, "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions), self.client_manager.num_available())

        results, failures = evaluate_clients(
            executor=self.executor,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            group_id=server_round,
        )

        self._require_complete_results(server_round, "evaluate", client_instructions, results, failures)

        log(INFO, "aggregate_evaluate: received %s results and %s failures",
            len(results), len(failures))

        aggregated_result: tuple[
            Optional[float],
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        log(INFO, "eval stats: (%s, %s)",
            metrics_aggregated["train_asr"], metrics_aggregated["test_asr"])

        return loss_aggregated, metrics_aggregated, (results, failures)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self, num_rounds: int):
        """Run federated training for num_rounds rounds."""
        history = History()
        start_time = timeit.default_timer()

        # Initial centralized evaluation
        log(INFO, "[INIT]")
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        for current_round in range(1, num_rounds + 1):

            # Federated training round
            res_fit = self.fit_round(current_round)
            if res_fit is None or res_fit[0] is None:
                raise RuntimeError(f"Round {current_round}: aggregation did not produce parameters")
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime is not None:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                # log(INFO, "fit metrics: (%s, %s)", current_round, fit_metrics)

                if self.experiment_manager is not None:
                    self.experiment_manager.log(fit_metrics, nested=True)

            # Centralized evaluation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(INFO, "fit progress: (%s, %s, %s, %s)",
                    current_round, loss_cen, metrics_cen,
                    timeit.default_timer() - start_time)
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)
                if self.experiment_manager is not None:
                    self.experiment_manager.log({
                        "centralized_loss": loss_cen,
                        "centralized_accu": metrics_cen["accuracy"],
                    }, nested=True)

            # Distributed evaluation
            res_fed = self.evaluate_round(server_round=current_round)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                # log(INFO, "eval metrics: (%s, %s)", current_round, evaluate_metrics_fed)

                if self.experiment_manager is not None:
                    self.experiment_manager.log(evaluate_metrics_fed, nested=False)

        elapsed = timeit.default_timer() - start_time
        return history, elapsed