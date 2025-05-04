"""Aggregation functions for strategy implementations."""

import math
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from fedml.common import Parameters


def aggregate(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [weights * num_examples for weights, num_examples in results]

    # Compute average weights of each layer
    weights_prime = reduce(torch.add, weighted_weights) / num_examples_total

    return weights_prime


def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_median(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w = (
        torch.stack(weights, dim=0)
        .float()
        .quantile(q=0.5, dim=0, interpolation="midpoint")
    )

    return median_w


def aggregate_geometric_median(results: list[tuple[Parameters, int]]) -> Parameters:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    models = [models for (models, _) in results]
    alphas = torch.tensor(
        [num_examples for (_, num_examples) in results],
        dtype=torch.float32,
        device=models[0][0].device,
    )
    alphas /= alphas.sum()

    geomedian, geo_weights = _compute_geometric_median(
        flat_models=torch.stack(models, dim=0), alphas=alphas
    )
    geo_weights /= geo_weights.max()

    return geomedian, geo_weights


def aggregate_krum(
    results: list[tuple[Parameters, int]], num_malicious: int, to_keep: int
) -> Parameters:
    """Compute krum or multi-krum."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    sorted_indices = torch.argsort(distance_matrix, dim=1)

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = torch.sum(
        distance_matrix.gather(1, sorted_indices[:, 1 : (num_closest + 1)]), dim=1
    )

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = torch.argsort(scores, descending=False)[:to_keep]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[torch.argmin(scores).item()]


def aggregate_mixing(
    results: list[tuple[Parameters, int]], aggregator: str, num_malicious: int, to_keep: int
) -> Parameters:
    """Compute fixing by mixing based aggregation."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f closest parameters vectors
    nf = len(weights) - num_malicious
    num_closest = max(1, nf)
    sorted_indices = torch.argsort(distance_matrix, dim=1)

    # Compute average of n-f nearest neighbors of each model
    # and create a results tuple with new weights
    weights_mixed = [
        (aggregate([results[i] for i in row[:nf]]), 1) for row in sorted_indices
    ]

    if aggregator == "KRUM":
        return aggregate_krum(results=weights_mixed, num_malicious=num_malicious, to_keep=to_keep)
    elif aggregator == "ROBUSTAVG":
        aggregated_parameters, _, _ = aggregate_bayesian(results=weights_mixed)
        return aggregated_parameters
    elif aggregator == "TRIMAVG":
        return aggregate_trimmed_average(results=weights_mixed, proportiontocut=(num_malicious/len(results)))
    elif aggregator == "GEOMED":
        aggregated_parameters, _ = aggregate_geometric_median(results=weights_mixed)
        return aggregated_parameters
    elif aggregator == "MEDIAN":
        return aggregate_median(results=weights_mixed)

    else:
        raise ValueError(f"Invalid aggregator {aggregator} specified.")


def aggregate_trimmed_average(
    results: list[tuple[Parameters, int]], proportiontocut: float
) -> Parameters:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    trimmed_w: Parameters = _trim_mean(
        torch.stack(weights, dim=0).float(), proportiontocut=proportiontocut
    )
    return trimmed_w


def aggregate_bulyan(
    results: list[tuple[Parameters, int]],
    num_malicious: int,
    aggregation_rule: Callable,  # type: ignore
    **aggregation_rule_kwargs: Dict[str, Any],
) -> Parameters:
    """Perform Bulyan aggregation.

    Parameters
    ----------
    results: list[tuple[Parameters, int]]
        Weights and number of samples for each of the client.
    num_malicious: int
        The maximum number of malicious clients.
    aggregation_rule: Callable
        Byzantine resilient aggregation rule used as the first step of the Bulyan
    aggregation_rule_kwargs: Any
        The arguments to the aggregation rule.

    Returns
    -------
    aggregated_parameters: Parameters
        Aggregated parameters according to the Bulyan strategy.
    """
    byzantine_resilient_single_ret_model_aggregation = [aggregate_krum]
    # also GeoMed (but not implemented yet)
    byzantine_resilient_many_return_models_aggregation = []  # type: ignore
    # Brute, Medoid (but not implemented yet)

    num_clients = len(results)
    if num_clients < 4 * num_malicious + 3:
        raise ValueError(
            "The Bulyan aggregation requires then number of clients to be greater or "
            "equal to the 4 * num_malicious + 3. This is the assumption of this method."
            "It is needed to ensure that the method reduces the attacker's leeway to "
            "the one proved in the paper."
        )
    selected_models_set: list[tuple[Parameters, int]] = []

    theta = num_clients - 2 * num_malicious
    beta = theta - 2 * num_malicious

    for _ in range(theta):
        best_model = aggregation_rule(
            results=results, num_malicious=num_malicious, **aggregation_rule_kwargs
        )
        list_of_weights = [weights for weights, num_samples in results]
        # This group gives exact result
        if aggregation_rule in byzantine_resilient_single_ret_model_aggregation:
            best_idx = _find_reference_weights(best_model, list_of_weights)
        # This group requires finding the closest model to the returned one
        # (weights distance wise)
        elif aggregation_rule in byzantine_resilient_many_return_models_aggregation:
            # when different aggregation strategies available
            # write a function to find the closest model
            raise NotImplementedError(
                "aggregate_bulyan currently does not support the aggregation rules that"
                " return many models as results. "
                "Such aggregation rules are currently not available in FedML."
            )
        else:
            raise ValueError(
                "The given aggregation rule is not added as Byzantine resilient. "
                "Please choose from Byzantine resilient rules."
            )

        selected_models_set.append(results[best_idx])

        # remove idx from tracker and weights_results
        results.pop(best_idx)

    # Compute median parameter vector across selected_models_set
    median_vect = aggregate_median(selected_models_set)

    # Take the averaged beta parameters of the closest distance to the median
    # (coordinate-wise)
    parameters_aggregated = _aggregate_n_closest_weights(
        median_vect, selected_models_set, beta_closest=beta
    )
    return parameters_aggregated


def update_weights(losses, tol=1e-3, maxiter=100):
    """Optimize Bernoulli probabilities"""
    weights = 0.95 * torch.ones_like(losses)
    new_weights = weights.detach().clone()
    for _ in range(maxiter):
        eps = 1 - torch.mean(weights)
        ratio = eps / (1 - eps)
        new_weights = torch.exp(-losses) / (ratio + torch.exp(-losses))
        error = torch.linalg.norm(new_weights - weights)
        weights = new_weights.detach().clone()
        if error < tol:
            break
    return new_weights


def mean(sample, maxiter=100, tol=1e-3):
    weights = torch.ones(sample.size(0), device=sample.device)
    theta = weights @ sample / torch.sum(weights)
    residuals = torch.linalg.norm(theta - sample, dim=1) ** 2
    sigma2 = weights @ residuals / torch.sum(weights)
    losses = 0.5 * residuals / sigma2

    for _ in range(maxiter):
        weights = update_weights(losses)
        prev_theta = theta.detach().clone()
        theta = weights @ sample / torch.sum(weights)
        residuals = torch.linalg.norm(theta - sample, dim=1) ** 2
        sigma2 = weights @ residuals / torch.sum(weights)
        losses = 0.5 * residuals / sigma2

        discrepancy = torch.linalg.norm(theta - prev_theta) / torch.linalg.norm(
            prev_theta
        )
        if discrepancy <= tol:
            break

    return theta, weights


def aggregate_bayesian(
    results: List[Tuple[Parameters, int]], version="v1"
) -> Parameters:
    """Compute weighted average using bayesian robust aggregation mechanism."""

    # Create a list of parameters (nn weights or gradients) and client sizes
    models = [model for (model, _) in results]

    with torch.no_grad():
        client_sizes = torch.tensor(
            [num_examples for (_, num_examples) in results],
            device=models[0][0].device,
            dtype=torch.float,
        )
        client_sizes /= client_sizes.mean()

        compute_bayesian = {
            "v1": _compute_bayesian_mean,
            "v2": _compute_bayesian_scale_invariant,
        }[version]

        # Compute Weights and normalize them for better tracking / readability
        avg_model_prime, scale, weight_pi = compute_bayesian(models, client_sizes)

        if len(weight_pi.size()) > 1:
            weight_pi = weight_pi.mean(dim=0)
        weight_pi /= weight_pi.max()

    return avg_model_prime, weight_pi, scale


def _trim_mean(array: torch.Tensor, proportiontocut: float) -> torch.Tensor:
    """Compute trimmed mean along axis=0."""
    axis = 0
    nobs = array.size(axis)
    todrop = int(proportiontocut * nobs)
    result = torch.mean(
        torch.topk(
            torch.topk(array, k=nobs - todrop, dim=0, largest=True)[0],
            k=nobs - (2 * todrop),
            dim=0,
            largest=False,
        )[0],
        dim=0,
    )

    return result


def _compute_geometric_median(
    flat_models: torch.Tensor, alphas: torch.Tensor, maxiter=100, tol=1e-20, eps=1e-8
) -> Tuple[Parameters, torch.Tensor]:
    """Compute the geometric median weights using Weiszfeld algorithm.

    :param models: An list of model weights
    :param maxiter: Maximum number of iterations to run
    :param tol: Tolerance threshold
    :param eps: Minimum threshold (to avoid division by zero)
    :returns: An array of geometric median weights
    """

    # Compute geometric median using Weiszfeld algorithm
    with torch.no_grad():
        # Find initial estimate using the initial guess
        geomedian = alphas @ flat_models / alphas.sum()

        for _ in range(maxiter):
            prev_geomedian = geomedian  # .detach().clone()
            dis = torch.linalg.vector_norm(flat_models - geomedian, dim=1)
            weights = alphas / torch.clamp(dis, min=eps)
            geomedian = weights @ flat_models / weights.sum()

            if torch.linalg.norm(prev_geomedian - geomedian) <= tol * torch.linalg.norm(
                geomedian
            ):
                break

    return geomedian, weights


def _compute_bayesian_mean(
    models: list[Parameters],
    max_eps: float = None,
    maxiter: int = 100,
    tol: float = 1e-3,
) -> Tuple[Parameters, float, list[float]]:
    """Compute the weights using bayesian method.

    :param models: An list of model weights
    :param max_eps: Maximum proportion of malicious models; must be in [0; 1]
    :param maxiter: Maximum number of iterations to run
    :param tol: Tolerance threshold
    :returns: (tuple): a tuple containing:
        - avg_model  - Aggregated model
        - total_size - The total size as computed by the algorithm
        - weights    - The weights assigned to each model
    """

    # Convert ndarray list to flat array list
    flat_models = torch.stack(models, dim=0)

    def update_params(w):
        mean = w @ flat_models / torch.sum(w)
        distances2 = torch.linalg.norm(mean - flat_models, dim=1) ** 2
        sigma2 = w @ distances2 / torch.sum(w)
        resids = 0.5 * (distances2 / sigma2 + torch.log(2 * torch.pi * sigma2))
        return mean, sigma2, resids

    weights = torch.ones(flat_models.size(0), device=flat_models.device)
    avg_model, sigma2, residuals = update_params(weights)
    prev_avg_model = None

    for _ in range(maxiter):
        weights = update_weights(residuals)
        if prev_avg_model is not None:
            del prev_avg_model
        prev_avg_model = avg_model  # .detach().clone()
        avg_model, sigma2, residuals = update_params(weights)
        discrepancy = torch.linalg.norm(avg_model - prev_avg_model) / torch.linalg.norm(
            prev_avg_model
        )
        if discrepancy <= tol:
            break

    return avg_model, sigma2, weights


def _compute_bayesian_scale_invariant(
    models: list[Parameters],
    max_eps: float = None,
    maxiter: int = 100,
    tol: float = 1e-3,
) -> Tuple[Parameters, float, list[float]]:
    """Compute the weights using bayesian method
    that is invariant to the scale of model weights.

    :param models: An list of model weights
    :param max_eps: Maximum proportion of malicious models; must be in [0; 1]
    :param maxiter: Maximum number of iterations to run
    :param tol: Tolerance threshold
    :returns: (tuple): a tuple containing:
        - avg_model  - Aggregated model
        - total_size - The total size as computed by the algorithm
        - weights    - The weights assigned to each model
    """

    # Convert ndarray list to flat array list
    flat_models = torch.stack(models, dim=0)

    def update_params(w):
        mean = w @ flat_models / torch.sum(w)
        distances2 = torch.linalg.norm(mean - flat_models, dim=1) ** 2
        sigma2 = w @ distances2 / torch.sum(w)
        resids = 0.5 * (distances2 / sigma2 + math.log(2 * math.pi))
        return mean, sigma2, resids

    weights = torch.ones(flat_models.size(0), device=flat_models.device)
    avg_model, sigma2, residuals = update_params(weights)
    prev_avg_model = None

    for _ in range(maxiter):
        weights = update_weights(residuals)
        if prev_avg_model is not None:
            del prev_avg_model
        prev_avg_model = avg_model  # .detach().clone()
        avg_model, sigma2, residuals = update_params(weights)
        discrepancy = torch.linalg.norm(avg_model - prev_avg_model) / torch.linalg.norm(
            prev_avg_model
        )
        if discrepancy <= tol:
            break

    return avg_model, sigma2, weights


def _compute_distances(weights: list[Parameters]) -> torch.Tensor:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = torch.stack(weights, dim=0)

    distance_matrix = torch.zeros((len(weights), len(weights)))
    for i, flat_w_i in enumerate(flat_w):
        for j, flat_w_j in enumerate(flat_w):
            delta = flat_w_i - flat_w_j
            norm = torch.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix


def _check_weights_equality(weights1: Parameters, weights2: Parameters) -> bool:
    """Check if weights are the same."""
    # if len(weights1) != len(weights2):
    #     return False
    return all(
        torch.equal(layer_weights1, layer_weights2)
        for layer_weights1, layer_weights2 in zip(weights1, weights2)
    )


def _find_reference_weights(
    reference_weights: Parameters, list_of_weights: list[Parameters]
) -> int:
    """Find the reference weights by looping through the `list_of_weights`.

    Raise Error if the reference weights is not found.

    Parameters
    ----------
    reference_weights: Parameters
        Weights that will be searched for.
    list_of_weights: List[Parameters]
        List of weights that will be searched through.

    Returns
    -------
    index: int
        The index of `reference_weights` in the `list_of_weights`.

    Raises
    ------
    ValueError
        If `reference_weights` is not found in `list_of_weights`.
    """
    for idx, weights in enumerate(list_of_weights):
        if _check_weights_equality(reference_weights, weights):
            return idx
    raise ValueError("The reference weights not found in list_of_weights.")


def _aggregate_n_closest_weights(
    reference_weights: Parameters,
    results: list[tuple[Parameters, int]],
    beta_closest: int,
) -> Parameters:
    """Calculate element-wise mean of the `N` closest values.

    Note, each i-th coordinate of the result weight is the average of the beta_closest
    -ith coordinates to the reference weights


    Parameters
    ----------
    reference_weights: Parameters
        The weights from which the distances will be computed
    results: list[tuple[Parameters, int]]
        The weights from models
    beta_closest: int
        The number of the closest distance weights that will be averaged

    Returns
    -------
    aggregated_weights: Parameters
        Averaged (element-wise) beta weights that have the closest distance to
         reference weights
    """
    list_of_weights = [weights for weights, _ in results]
    aggregated_weights = []

    for layer_id, layer_weights in enumerate(reference_weights):
        other_weights_layer_list = []
        for other_w in list_of_weights:
            other_weights_layer = other_w[layer_id]
            other_weights_layer_list.append(other_weights_layer)
        other_weights_layer_np = np.array(other_weights_layer_list)
        diff_np = np.abs(layer_weights - other_weights_layer_np)
        # Create indices of the smallest differences
        # We do not need the exact order but just the beta closest weights
        # therefore np.argpartition is used instead of np.argsort
        indices = np.argpartition(diff_np, kth=beta_closest - 1, axis=0)
        # Take the weights (coordinate-wise) corresponding to the beta of the
        # closest distances
        beta_closest_weights = np.take_along_axis(
            other_weights_layer_np, indices=indices, axis=0
        )[:beta_closest]
        aggregated_weights.append(np.mean(beta_closest_weights, axis=0))
    return aggregated_weights
