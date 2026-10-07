"""Federated Bayesian aggregation strategy."""

from collections.abc import Iterable
from logging import DEBUG
from math import isfinite

import torch

from fedml.strategy.aggregators import FederatedAverage
from fedml.strategy.aggregators.aggregate import aggregate_bayesian
from fedml.utils.logger import log


def _validate_scale(scale: float) -> float:
    """Return a finite, strictly positive aggregation input scale."""
    scale = float(scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "aggregation input scales must be finite and strictly positive; "
            f"got {scale!r}"
        )
    return scale


def _scale_label(scale: float) -> str:
    """Format a scale for stable metric names."""
    return format(scale, ".12g").replace("-", "m").replace(".", "p").replace("+", "")


def aggregate_bayesian_scaled(weights_results, *, variation: str, scale: float):
    """Aggregate scaled model weights and map the aggregate back.

    The existing Bayesian aggregator receives ``scale * weights``. Only the
    aggregate is divided by ``scale``; all remaining return values are those
    produced by the existing implementation in the scaled coordinates.
    """
    scale = _validate_scale(scale)
    scaled_results = weights_results
    if scale != 1.0:
        scaled_results = [
            (parameters * scale, num_examples)
            for parameters, num_examples in weights_results
        ]
    parameters, weight_pi, sigma2, rho_hat = aggregate_bayesian(
        scaled_results, version=variation
    )
    return parameters / scale, weight_pi, sigma2, rho_hat


class FederatedBayesian(FederatedAverage):

    def __init__(
        self,
        *,
        variation: str = "v1",
        aggregation_input_scale: float = 1.0,
        diagnostic_scales: Iterable[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.robust_aggregation_variation = variation
        self.aggregation_input_scale = _validate_scale(aggregation_input_scale)
        self.diagnostic_scales = tuple(
            dict.fromkeys(_validate_scale(scale) for scale in (diagnostic_scales or ()))
        )
        log(
            DEBUG,
            "Building %s Aggregation Strategy with variation=%s, "
            "aggregation_input_scale=%s, diagnostic_scales=%s",
            self,
            variation,
            self.aggregation_input_scale,
            self.diagnostic_scales,
        )

    def __repr__(self) -> str:
        return "FederatedBayesian"

    def _aggregate_weights(self, weights_results, **kwargs):
        scales = [self.aggregation_input_scale]
        if self.diagnostic_scales:
            scales.extend((1.0, *self.diagnostic_scales))

        outputs = {}
        for scale in dict.fromkeys(scales):
            outputs[scale] = aggregate_bayesian_scaled(
                weights_results,
                variation=self.robust_aggregation_variation,
                scale=scale,
            )

        parameters_aggregated, weight_pi, _, rho_hat = outputs[
            self.aggregation_input_scale
        ]
        diagnostics = self._build_scale_diagnostics(weights_results, outputs)
        return parameters_aggregated, weight_pi, rho_hat, diagnostics

    def _build_scale_diagnostics(self, weights_results, outputs):
        """Compare matched scaled aggregations on the same submitted weights."""
        if not self.diagnostic_scales:
            return {}

        reference_parameters, reference_weights, _, _ = outputs[1.0]
        with torch.no_grad():
            first_model = weights_results[0][0]
            center = torch.zeros_like(first_model)
            for parameters, _ in weights_results:
                center.add_(parameters)
            center.div_(len(weights_results))

            dispersion2 = torch.zeros(
                (), dtype=first_model.dtype, device=first_model.device
            )
            for parameters, _ in weights_results:
                dispersion2.add_(torch.sum((parameters - center) ** 2))
            dispersion = torch.sqrt(dispersion2 / len(weights_results))
            dispersion = dispersion.clamp_min(torch.finfo(first_model.dtype).eps)

        diagnostics = {}
        for scale in self.diagnostic_scales:
            parameters, weights, sigma2, rho_hat = outputs[scale]
            prefix = f"scale_diag_s{_scale_label(scale)}"
            aggregate_difference = torch.linalg.vector_norm(
                parameters - reference_parameters
            )
            weight_difference = torch.mean(torch.abs(weights - reference_weights))
            floor_active = sigma2 <= torch.finfo(sigma2.dtype).eps

            diagnostics[f"{prefix}_aggregate_dispersion_ratio"] = float(
                (aggregate_difference / dispersion).detach().cpu()
            )
            diagnostics[f"{prefix}_weight_mean_abs_diff"] = float(
                weight_difference.detach().cpu()
            )
            diagnostics[f"{prefix}_rho_hat"] = float(rho_hat.detach().cpu())
            diagnostics[f"{prefix}_sigma2_scaled"] = float(sigma2.detach().cpu())
            diagnostics[f"{prefix}_sigma2_native"] = float(
                (sigma2 / (scale * scale)).detach().cpu()
            )
            diagnostics[f"{prefix}_variance_floor_active"] = bool(
                floor_active.detach().cpu()
            )

        return diagnostics

    def aggregate_fit(self, server_round, results, failures, selected=None):
        """Override to pass weight_pi as extra metric kwarg."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = self._filter_results(results, selected)
        parameters_aggregated, weight_pi, rho_hat, scale_diagnostics = (
            self._aggregate_weights(weights_results) if weights_results
            else (None, None, None, {})
        )

        metrics_aggregated = self._build_metrics(
            server_round, results, selected, parameters_aggregated,
            weight_pi=weight_pi.detach().cpu().numpy() if weight_pi is not None else None,
            rho_hat=float(rho_hat) if rho_hat is not None else None,
            scale_diagnostics=scale_diagnostics,
        )

        return parameters_aggregated, metrics_aggregated
