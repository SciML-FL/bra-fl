"""Filtered FL server — extends BaseServer with defense filter applied before aggregation."""

import concurrent.futures
from logging import ERROR, INFO, WARNING, DEBUG
from typing import Dict, Optional
from unittest import result

from fedml.utils.typing import Parameters, Scalar
from fedml.utils.logger import log
from fedml.defenses import create_filter

from fedml.server import BaseServer
from fedml.server.execution import fit_clients, post_training, _handle_failed_future


class FilteredServer(BaseServer):
    """Federated learning server with a pluggable defense filter.

    Extends BaseServer by injecting a filter step between client training
    and aggregation. The filter selects which client updates to include
    in aggregation based on its detection logic (e.g. GAN, Krum, mean).
    """

    def __init__(
        self,
        *,
        client_manager,
        experiment_manager=None,
        strategy=None,
        user_configs: Optional[Dict] = None,
        initial_parameters=None,
        executor_type: str = "ThreadPool",
        max_workers: Optional[int] = None,
    ) -> None:
        super().__init__(
            client_manager=client_manager,
            experiment_manager=experiment_manager,
            strategy=strategy,
            user_configs=user_configs,
            initial_parameters=initial_parameters,
            executor_type=executor_type,
            max_workers=max_workers,
        )
        self.filter = create_filter(user_configs=user_configs)

    def fit_round(self, server_round: int):
        """Perform a single round of federated training with filtering."""
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self.client_manager,
        )

        if not client_instructions:
            log(WARNING, "configure_fit: no clients selected, cancel")
            return None

        log(INFO, "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions), self.client_manager.num_available())

        # Launch server-side filter tasks concurrently with client training
        log(DEBUG, f"parameters device: {self.parameters.device}")
        filter_future = self.executor.submit(
            self.filter.server_tasks, self.parameters, server_round
        )

        # Client training
        results, failures = fit_clients(
            executor=self.executor,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            group_id=server_round,
        )

        self._require_complete_results(server_round, "fit", client_instructions, results, failures)
        log(INFO, "aggregate_fit: received %s results and %s failures", len(results), len(failures))

        # Post-training callbacks
        log(DEBUG, f"Post-training callbacks: launching with {len(client_instructions)} client instructions ...")
        results, failures = post_training(
            executor=self.executor,
            client_instructions=client_instructions,
            results=results,
            failures=failures,
        )
        self._require_complete_results(server_round, "post-training", client_instructions, results, failures)
        log(DEBUG, f"Post-training callbacks: completed with {len(results)} results and {len(failures)} failures.")

        # Wait for filter server tasks to complete before filtering
        finished_fs, _ = concurrent.futures.wait(fs={filter_future}, timeout=None)

        # If filter server tasks failed, skip filtering and aggregation for this round
        for future in finished_fs:
            if _handle_failed_future(future=future) is not None:
                log(ERROR, f"Filter server tasks failed for round {server_round}, skipping aggregation.")
                return None
            else:
                log(DEBUG, f"Filter server tasks completed successfully for round {server_round}. Proceeding with filtering and aggregation.")
                trained_weights = future.result()
                if trained_weights is not None:
                    self.filter.gen_model.set_weights(trained_weights)

        # Filter client updates
        selected_indexes, client_stats = self.filter.filter_updates(
            client_weights=[
                (fit_res.parameters, fit_res.num_examples)
                for _, fit_res in results
            ],
            server_round=server_round,
        )

        # Compute TPR TNR for diagnostics
        tpr, tnr = 0, 0
        num_malicious = sum(1 for _, fit_res in results if fit_res.metrics.get("attacking", False))
        num_benign = len(results) - num_malicious
        if num_malicious > 0:
            tpr = sum(
                1 for index, (_, fit_res) in enumerate(results) 
                if fit_res.metrics.get("attacking", False) 
                and index not in selected_indexes) / num_malicious
        if num_benign > 0:
            tnr = sum(
                1 for index, (_, fit_res) in enumerate(results) 
                if not fit_res.metrics.get("attacking", False) 
                and index in selected_indexes) / num_benign

        client_ids = [fit_res.metrics.get("client_id", "NONE") for (_, fit_res) in results]
        attack_status = [fit_res.metrics.get("attacking", False) for (_, fit_res) in results]
        select_status = [index in selected_indexes for index in range(len(results))]
        log(INFO, f"Filter diagnostics (client_ids): {client_ids}")
        log(INFO, f"Filter diagnostics (attacking): {attack_status}")
        log(INFO, f"Filter diagnostics (selected?): {select_status}")
        log(INFO, f"Filter diagnostics: TPR={tpr:.2f}, TNR={tnr:.2f} (malicious={num_malicious}, benign={num_benign})")

        # Aggregate filtered results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
            selected=selected_indexes,
        )
        parameters_aggregated, metrics_aggregated = aggregated_result

        # Attach per-client filter diagnostics to metrics
        self._log_filter_stats(results, metrics_aggregated, client_stats)

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def _log_filter_stats(self, results, metrics_aggregated, client_stats) -> None:
        """Attach per-client filter diagnostics to metrics_aggregated in place."""
        client_ids = [res.metrics["client_id"] for _, res in results]

        if self.filter.filter_type == "GAN" and client_stats is not None:
            metrics_aggregated["filter_loss"] = {}
            metrics_aggregated["filter_accu_all"] = {}
            for index, cid in enumerate(client_ids):
                metrics_aggregated["filter_loss"][f"client_{cid}"] = client_stats["avg_loss"][index]
                metrics_aggregated["filter_accu_all"][f"client_{cid}"] = client_stats["accu_all"][index]

        if self.filter.filter_type == "KRUM" and client_stats is not None:
            metrics_aggregated["distances"] = {}
            for index, cid in enumerate(client_ids):
                metrics_aggregated["distances"][f"client_{cid}"] = client_stats["distances"][index, :]