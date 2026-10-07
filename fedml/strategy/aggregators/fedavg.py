"""Federated Averaging (FedAvg) strategy."""

from logging import WARNING
from typing import Dict, List, Optional, Tuple

import torch

from fedml.utils.logger import log
from fedml.utils.typing import Parameters, Scalar

from fedml.strategy.base_strategy import Strategy
from fedml.strategy.aggregators.aggregate import aggregate, weighted_loss_avg


class FederatedAverage(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    All other strategies inherit from this class and override
    _aggregate_weights() with their specific aggregation logic.
    """

    def __repr__(self) -> str:
        return "FederatedAverage"

    # ------------------------------------------------------------------
    # Template method — subclasses override this only
    # ------------------------------------------------------------------

    def _aggregate_weights(self, weights_results, **kwargs):
        """Aggregate filtered weight results into new global parameters.

        Subclasses override this method to swap in a different aggregation
        algorithm. weights_results is a list of (parameters, num_examples).
        """
        return aggregate(weights_results)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _filter_results(self, results, selected: Optional[List[int]]):
        """Extract (parameters, num_examples) pairs, applying selection mask."""
        return [
            (fit_res.parameters, fit_res.num_examples)
            for indx, (_, fit_res) in enumerate(results)
            if selected is None or indx in selected
        ]

    def _build_metrics(
        self,
        server_round: int,
        results,
        selected,
        parameters_aggregated,
        **extra_metric_kwargs,
    ) -> Dict[str, Scalar]:
        """Build aggregated metrics dict, delegating to aggregation fn if set."""
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            return self.fit_metrics_aggregation_fn(
                fit_metrics=fit_metrics,
                selected=selected,
                update_norms=self.compute_benign_norms(
                    results=results,
                    aggregated_parameters=parameters_aggregated,
                ),
                **extra_metric_kwargs,
            )
        if server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        return {}

    # ------------------------------------------------------------------
    # aggregate_fit orchestrator
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        selected: Optional[List[int]] = None,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using the strategy's aggregation rule."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = self._filter_results(results, selected)
        parameters_aggregated = (
            self._aggregate_weights(weights_results) if weights_results else None
        )

        metrics_aggregated = self._build_metrics(
            server_round, results, selected, parameters_aggregated
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss)
             for _, evaluate_res in results]
        )

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def compute_benign_norms(self, results, aggregated_parameters) -> Dict:
        """Compute per-class norms for benign vs malicious updates.

        Currently disabled — returns early with empty dict.
        Retained for future use.
        """
        return {}

        # NOTE: Code below is retained for future use but currently disabled.
        if aggregated_parameters is None:
            return {}
        norm_dict = {}

        with torch.no_grad():
            benign_weights = []
            advers_weights = []
            for _, fit_res in results:
                if fit_res.metrics["attacking"]:
                    advers_weights.append((fit_res.parameters, fit_res.num_examples))
                else:
                    benign_weights.append((fit_res.parameters, fit_res.num_examples))

            param_list = [aggregated_parameters]

            if benign_weights:
                benign_aggregate = aggregate(benign_weights)
                param_list.append(benign_aggregate)
                flattened_benign_weights = torch.stack(
                    [w for (w, _) in benign_weights], dim=0
                )
            if advers_weights:
                advers_aggregate = aggregate(advers_weights)
                param_list.append(advers_aggregate)
                flattened_malicious_weights = torch.stack(
                    [w for (w, _) in advers_weights], dim=0
                )

            flattened_weights = torch.stack(param_list, dim=0)
            norm_dict["norm_aggregated_update"] = torch.linalg.vector_norm(
                flattened_weights[0]
            ).item()

            if benign_weights:
                norm_dict["num_benign_updates"] = len(benign_weights)
                norm_dict["norm_benign_updates"] = torch.linalg.vector_norm(
                    flattened_weights[1]
                ).item()
                diff = flattened_weights[1] - flattened_weights[0]
                norm_dict["norm_diff_agg_benign_aggregated"] = (
                    torch.linalg.vector_norm(diff).item()
                )
                norm_dict["norm_diff_all_benign_aggregated"] = (
                    torch.linalg.vector_norm(
                        flattened_benign_weights - flattened_weights[0], dim=1
                    ).cpu().detach().numpy()
                )
            else:
                norm_dict.update({
                    "num_benign_updates": 0,
                    "norm_benign_updates": None,
                    "norm_diff_agg_benign_aggregated": None,
                    "norm_diff_all_benign_aggregated": None,
                })

            if advers_weights:
                norm_dict["num_malicious_updates"] = len(advers_weights)
                norm_dict["norm_malicious_updates"] = torch.linalg.vector_norm(
                    flattened_weights[2]
                ).item()
                diff = flattened_weights[2] - flattened_weights[0]
                norm_dict["norm_diff_agg_malicious_aggregated"] = (
                    torch.linalg.vector_norm(diff).item()
                )
                norm_dict["norm_diff_all_malicious_aggregated"] = (
                    torch.linalg.vector_norm(
                        flattened_malicious_weights - flattened_weights[0], dim=1
                    ).cpu().detach().numpy()
                )
            else:
                norm_dict.update({
                    "num_malicious_updates": 0,
                    "norm_malicious_updates": None,
                    "norm_diff_agg_malicious_aggregated": None,
                    "norm_diff_all_malicious_aggregated": None,
                })

        return norm_dict