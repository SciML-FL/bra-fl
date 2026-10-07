"""Aggregation functions for strategy implementations."""

import itertools
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fedml.utils import Parameters

# ------------------------------------------------------------------
# Core aggregation functions
# ------------------------------------------------------------------


def aggregate(results: List[Tuple[Parameters, int]]) -> Parameters:
    """Compute weighted average of parameters."""
    num_examples_total = sum(num_examples for (_, num_examples) in results)
    weighted_weights = [weights * num_examples for weights, num_examples in results]
    return reduce(torch.add, weighted_weights) / num_examples_total


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Compute weighted average of evaluation losses."""
    num_total = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total


def aggregate_median(results: List[Tuple[Parameters, int]]) -> Parameters:
    """Compute coordinate-wise median."""
    weights = [w for w, _ in results]
    return (
        torch.stack(weights, dim=0)
        .float()
        .quantile(q=0.5, dim=0, interpolation="midpoint")
    )


def aggregate_geometric_median(
    results: List[Tuple[Parameters, int]],
) -> Tuple[Parameters, torch.Tensor]:
    """Compute geometric median using the Weiszfeld algorithm."""
    models = [m for (m, _) in results]
    alphas = torch.tensor(
        [n for (_, n) in results],
        dtype=torch.float32,
        device=models[0].device,
    )
    alphas /= alphas.sum()

    geomedian, geo_weights = _compute_geometric_median(
        flat_models=torch.stack(models, dim=0), alphas=alphas
    )
    geo_weights /= geo_weights.max()
    return geomedian, geo_weights


def aggregate_krum(
    results: List[Tuple[Parameters, int]],
    num_malicious: int,
    to_keep: int,
) -> Parameters:
    """Compute Krum or Multi-Krum aggregation."""
    weights = [w for w, _ in results]
    distance_matrix = _compute_distances(weights)

    num_closest = max(1, len(weights) - num_malicious - 2)
    sorted_indices = torch.argsort(distance_matrix, dim=1)
    scores = torch.sum(
        distance_matrix.gather(1, sorted_indices[:, 1 : (num_closest + 1)]), dim=1
    )

    if to_keep > 0:
        # Multi-Krum: average the to_keep best clients
        best_indices = torch.argsort(scores, descending=False)[:to_keep]
        return aggregate([results[i] for i in best_indices])

    # Krum: return single best model
    return weights[torch.argmin(scores).item()]


def aggregate_trimmed_average(
    results: List[Tuple[Parameters, int]],
    proportiontocut: float,
) -> Parameters:
    """Compute trimmed mean, dropping proportiontocut from each tail."""
    weights = [w for w, _ in results]
    return _trim_mean(
        torch.stack(weights, dim=0).float(), proportiontocut=proportiontocut
    )


def aggregate_bulyan(
    results: List[Tuple[Parameters, int]],
    num_malicious: int,
    aggregation_rule: Callable,
    **aggregation_rule_kwargs: Dict[str, Any],
) -> Parameters:
    """Perform Bulyan aggregation.

    Parameters
    ----------
    results:
        Weights and number of samples per client.
    num_malicious:
        Maximum number of malicious clients.
    aggregation_rule:
        Byzantine-resilient rule used in the first step (e.g. aggregate_krum).
    aggregation_rule_kwargs:
        Additional arguments forwarded to the aggregation rule.
    """
    byzantine_resilient_single_ret = [aggregate_krum]

    num_clients = len(results)
    if num_clients < 4 * num_malicious + 3:
        raise ValueError(
            f"Bulyan requires num_clients >= 4 * num_malicious + 3. "
            f"Got num_clients={num_clients}, num_malicious={num_malicious}."
        )

    theta = num_clients - 2 * num_malicious
    beta = theta - 2 * num_malicious
    selected: List[Tuple[Parameters, int]] = []

    for _ in range(theta):
        best_model = aggregation_rule(
            results=results, num_malicious=num_malicious, **aggregation_rule_kwargs
        )
        list_of_weights = [w for w, _ in results]

        if aggregation_rule in byzantine_resilient_single_ret:
            best_idx = _find_reference_weights(best_model, list_of_weights)
        else:
            raise NotImplementedError(
                "aggregate_bulyan does not yet support aggregation rules that "
                "return multiple models."
            )

        selected.append(results[best_idx])
        results.pop(best_idx)

    median_vect = aggregate_median(selected)
    return _aggregate_n_closest_weights(median_vect, selected, beta_closest=beta)


def aggregate_meamed(
    results: list[tuple[Parameters, int]], num_malicious: int
) -> Parameters:
    """Compute mean around median (Meamed)."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    n = len(weights)
    k = n - num_malicious

    if num_malicious * 2 >= n:
        raise ValueError(f"Cannot tolerate 2f >= n. Got f={num_malicious}, n={n}")

    stacked_weights = torch.stack(weights, dim=0).float()
    median_w = stacked_weights.quantile(q=0.5, dim=0, interpolation="midpoint")
    abs_diff = torch.abs(stacked_weights - median_w)

    nearest_indices = torch.topk(abs_diff, k=k, dim=0, largest=False)[1]
    nearest_weights = torch.gather(stacked_weights, dim=0, index=nearest_indices)
    return nearest_weights.mean(dim=0)


def aggregate_mda(
    results: list[tuple[Parameters, int]], num_malicious: int
) -> Parameters:
    """Compute Minimum Diameter Averaging (MDA)."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    n = len(weights)
    k = n - num_malicious

    if num_malicious * 2 >= n:
        raise ValueError(f"Cannot tolerate 2f >= n. Got f={num_malicious}, n={n}")

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    min_diameter = float("inf")
    best_indices = list(range(k))

    for subset in itertools.combinations(range(n), k):
        # Find the diameter of the current subset
        if len(subset) < 2:
            diameter = 0.0
        else:
            subset_tensor = torch.tensor(subset, device=distance_matrix.device)
            diameter = torch.max(
                distance_matrix[subset_tensor][:, subset_tensor]
            ).item()

        if diameter < min_diameter:
            min_diameter = diameter
            best_indices = subset

    # Return the average of the selected subset
    best_results = [results[i] for i in best_indices]
    return aggregate(best_results)


def aggregate_caf(
    results: list[tuple[Parameters, int]], num_malicious: int, max_iters: int
) -> Parameters:
    """Compute Covariance-bound Agnostic Filter (CAF)."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    n = len(weights)

    if num_malicious * 2 >= n:
        raise ValueError(f"Cannot tolerate 2f >= n. Got f={num_malicious}, n={n}")

    vectors = torch.stack(weights, dim=0).float()
    num_vectors, dimension = vectors.shape
    c = torch.ones(num_vectors, dtype=vectors.dtype, device=vectors.device)
    c_sum = torch.sum(c)
    best_eigenvalue = float("inf")
    best_mu_c = None

    while c_sum > n - 2 * num_malicious:
        weighted_sum = torch.sum(c[:, None] * vectors, dim=0)
        current_mu_c = weighted_sum / c_sum
        diffs = vectors - current_mu_c

        max_eigenvalue, max_eigenvector = _compute_dominant_eigenvector(
            diffs=diffs, weights=c, dimension=dimension, max_iters=max_iters
        )

        if max_eigenvalue < best_eigenvalue:
            best_eigenvalue = max_eigenvalue
            best_mu_c = current_mu_c

        tau = torch.matmul(diffs, max_eigenvector) ** 2
        tau_max = torch.max(tau)
        if tau_max <= 0:
            break

        c = c * (1 - tau / tau_max)
        c_sum = torch.sum(c)

    if best_mu_c is None:
        return torch.mean(vectors, dim=0)

    return best_mu_c


def aggregate_smea(
    results: list[tuple[Parameters, int]], num_malicious: int, max_iters: int = 50
) -> Parameters:
    """Compute Smallest Maximum Eigenvalue Averaging (SMEA)."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]
    n = len(weights)
    k = n - num_malicious

    if num_malicious * 2 >= n:
        raise ValueError(
            f"Too many Byzantine clients (2f >= n). Got f={num_malicious}, n={n}"
        )

    vectors = torch.stack(weights, dim=0).float()
    _, dimension = vectors.shape
    min_eigenvalue = float("inf")
    best_indices = list(range(k))

    for subset in itertools.combinations(range(n), k):
        subset_tensor = torch.tensor(subset, device=vectors.device)
        subset_vectors = vectors[subset_tensor]
        avg = torch.mean(subset_vectors, dim=0)
        diffs = subset_vectors - avg
        max_eigenvalue, _ = _compute_dominant_eigenvector_unweighted(
            diffs=diffs, dimension=dimension, max_iters=max_iters
        )

        if max_eigenvalue < min_eigenvalue:
            min_eigenvalue = max_eigenvalue
            best_indices = subset

    best_tensor = torch.tensor(best_indices, device=vectors.device)
    return torch.mean(vectors[best_tensor], dim=0)


def aggregate_bayesian(
    results: List[Tuple[Parameters, int]], version="v1"
) -> tuple[Parameters, Parameters, Parameters]:
    """Compute weighted average using bayesian robust aggregation mechanism."""

    # Create a list of parameters (nn weights or gradients)
    models = [model for (model, _) in results]

    with torch.no_grad():
        compute_bayesian = {
            "v1": lambda models: _compute_bayesian_mean(models),
            "v2": lambda models: _compute_bayesian_mean(models, rho_min=None),
        }[version]

        avg_model_prime, sigma2, weight_pi = compute_bayesian(models)
        
        # Compute rho_hat (rho_hat = mean of raw posteriors)
        rho_hat = weight_pi.mean()

    return avg_model_prime, weight_pi, sigma2, rho_hat


def aggregate_mixing(
    results: List[Tuple[Parameters, int]],
    aggregator: str,
    num_malicious: int,
    to_keep: int,
) -> Parameters:
    """Nearest-neighbor mixing aggregation."""
    weights = [w for w, _ in results]
    distance_matrix = _compute_distances(weights)

    nf = max(1, len(weights) - num_malicious)
    sorted_indices = torch.argsort(distance_matrix, dim=1)

    # Build mixed weights: each client's model is replaced by the
    # average of its nf nearest neighbors
    weights_mixed = [
        (aggregate([results[i] for i in row[:nf]]), 1) for row in sorted_indices
    ]

    if aggregator == "KRUM":
        return aggregate_krum(
            results=weights_mixed, num_malicious=num_malicious, to_keep=to_keep
        )
    elif aggregator == "ROBUSTAVG":
        aggregated_parameters, _, _ = aggregate_bayesian(results=weights_mixed)
        return aggregated_parameters
    elif aggregator == "TRIMAVG":
        return aggregate_trimmed_average(
            results=weights_mixed,
            proportiontocut=(num_malicious / len(results)),
        )
    elif aggregator == "GEOMED":
        aggregated_parameters, _ = aggregate_geometric_median(results=weights_mixed)
        return aggregated_parameters
    elif aggregator == "MEDIAN":
        return aggregate_median(results=weights_mixed)
    else:
        raise ValueError(f"Invalid aggregator '{aggregator}' specified.")


def aggregate_bucketing(
    results: List[Tuple[Parameters, int]],
    num_malicious: int,
    to_keep: int,
) -> Parameters:
    """Bucketing aggregation.

    TODO: Not yet implemented.

    Parameters
    ----------
    results:
        Weights and number of samples per client.
    num_malicious:
        Maximum number of malicious clients.
    to_keep:
        Number of clients to keep after bucketing.
    """
    raise NotImplementedError(
        "aggregate_bucketing is not yet implemented. "
        "Implement the bucketing logic here and remove this error."
    )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _compute_distances(weights: List[Parameters]) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between weight vectors.

    Vectorized implementation: O(n) stacking + O(1) broadcasting,
    replacing the previous O(n²) nested Python loop.
    """
    # flat_w: (n, d)
    flat_w = torch.stack(weights, dim=0).float()

    # ||a - b||² = ||a||² + ||b||² - 2 * a·b
    sq_norms = (flat_w * flat_w).sum(dim=1)  # (n,)
    distance_matrix = (
        sq_norms.unsqueeze(1)  # (n, 1)
        + sq_norms.unsqueeze(0)  # (1, n)
        - 2.0 * (flat_w @ flat_w.T)  # (n, n)
    )
    # Clamp to avoid tiny negative values from floating point errors
    return distance_matrix.clamp(min=0.0)


def _trim_mean(array: torch.Tensor, proportiontocut: float) -> torch.Tensor:
    """Compute trimmed mean along dim=0, dropping proportiontocut from each tail."""
    nobs = array.size(0)
    todrop = int(proportiontocut * nobs)
    return torch.mean(
        torch.topk(
            torch.topk(array, k=nobs - todrop, dim=0, largest=True)[0],
            k=nobs - (2 * todrop),
            dim=0,
            largest=False,
        )[0],
        dim=0,
    )


def _compute_geometric_median(
    flat_models: torch.Tensor,
    alphas: torch.Tensor,
    maxiter: int = 100,
    tol: float = 1e-20,
    eps: float = 1e-8,
) -> Tuple[Parameters, torch.Tensor]:
    """Compute geometric median using the Weiszfeld algorithm."""
    with torch.no_grad():
        geomedian = alphas @ flat_models / alphas.sum()
        for _ in range(maxiter):
            prev = geomedian
            dis = torch.linalg.vector_norm(flat_models - geomedian, dim=1)
            weights = alphas / torch.clamp(dis, min=eps)
            geomedian = weights @ flat_models / weights.sum()
            if torch.linalg.norm(prev - geomedian) <= tol * torch.linalg.norm(
                geomedian
            ):
                break
    return geomedian, weights


def _compute_dominant_eigenvector(
    diffs: torch.Tensor,
    weights: torch.Tensor,
    dimension: int,
    max_iters: int = 50,
    eps: float = 1e-12,
) -> Tuple[float, torch.Tensor]:
    """Compute dominant covariance eigenvector using power iteration."""
    vector = torch.randn(dimension, dtype=diffs.dtype, device=diffs.device)
    vector_norm = torch.linalg.norm(vector).clamp_min(eps)
    vector = vector / vector_norm

    eigenvalue = torch.tensor(float("inf"), dtype=diffs.dtype, device=diffs.device)
    for _ in range(max_iters):
        dot_products = torch.matmul(diffs, vector)
        weighted_sum = torch.sum(
            weights[:, None] * diffs * dot_products[:, None], dim=0
        ) / torch.sum(weights).clamp_min(eps)
        weighted_sum_norm = torch.linalg.norm(weighted_sum)
        if weighted_sum_norm <= eps:
            return 0.0, vector

        vector = weighted_sum / weighted_sum_norm
        eigenvalue = torch.dot(vector, weighted_sum)

    return eigenvalue.item(), vector


def _compute_dominant_eigenvector_unweighted(
    diffs: torch.Tensor,
    dimension: int,
    max_iters: int = 50,
    eps: float = 1e-12,
) -> Tuple[float, torch.Tensor]:
    """Compute dominant covariance eigenvector using unweighted power iteration."""
    vector = torch.randn(dimension, dtype=diffs.dtype, device=diffs.device)
    vector_norm = torch.linalg.norm(vector).clamp_min(eps)
    vector = vector / vector_norm

    eigenvalue = torch.tensor(float("inf"), dtype=diffs.dtype, device=diffs.device)
    for _ in range(max_iters):
        dot_products = torch.matmul(diffs, vector)
        weighted_sum = torch.sum(diffs * dot_products[:, None], dim=0)
        weighted_sum_norm = torch.linalg.norm(weighted_sum)
        if weighted_sum_norm <= eps:
            return 0.0, vector

        vector = weighted_sum / weighted_sum_norm
        eigenvalue = torch.dot(vector, weighted_sum)

    return eigenvalue.item(), vector


def _compute_bayesian_mean(
    models: list[Parameters],
    maxiter: int = 100,
    tol: float = 1e-3,
    rho_init: float = 0.95,
    rho_min: float | None = 0.5,
):
    """Compute robust aggregate using EM-type updates.

    :param models: List of model weights.
    :param maxiter: Maximum number of outer iterations.
    :param tol: Tolerance threshold for the aggregate update.
    :param rho_init: Initial value of the benign fraction rho.
    :param rho_min: Lower bound on rho.
                Use 0.5 for benign-majority assumption.
    :returns: tuple containing:
        - avg_model: Aggregated model.
        - sigma2: Estimated scale parameter.
        - weights: Relative aggregation weights normalized so that max(weights)=1.
    """

    flat_models = torch.stack(models, dim=0)

    device = flat_models.device
    dtype = flat_models.dtype
    eps_num = torch.finfo(dtype).eps

    rho_lower = eps_num if rho_min is None else float(rho_min)
    rho_upper = 1.0 - eps_num
    log_n = torch.log(
        torch.tensor(float(flat_models.size(0)), device=device, dtype=dtype)
    )
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype))

    def update_params(weights):
        denom = torch.sum(weights)

        mean = weights @ flat_models / denom
        distances2 = torch.sum((flat_models - mean) ** 2, dim=1)

        sigma2 = weights @ distances2 / denom
        sigma2 = torch.clamp(sigma2, min=eps_num)

        losses = 0.5 * (distances2 / sigma2 + log_2pi + torch.log(sigma2))

        return mean, sigma2, losses

    def update_weights(losses, rho):
        rho = torch.clamp(rho, rho_lower, rho_upper)

        logit_rho = torch.log(rho) - torch.log1p(-rho)
        log_weights = F.logsigmoid(logit_rho - losses)

        rho_new = torch.exp(torch.logsumexp(log_weights, dim=0) - log_n)
        rho_new = torch.clamp(rho_new, rho_lower, rho_upper)

        weights = torch.exp(log_weights - torch.max(log_weights))

        return rho_new, weights

    # Initialization: equal relative weights
    weights = torch.ones(flat_models.size(0), device=device, dtype=dtype)
    rho = torch.tensor(rho_init, device=device, dtype=dtype)
    rho = torch.clamp(rho, rho_lower, rho_upper)

    avg_model, sigma2, losses = update_params(weights)

    for _ in range(maxiter):
        prev_avg_model = avg_model

        rho, weights = update_weights(losses, rho)
        avg_model, sigma2, losses = update_params(weights)

        denom = torch.linalg.norm(prev_avg_model).clamp_min(eps_num)
        discrepancy = torch.linalg.norm(avg_model - prev_avg_model) / denom

        if discrepancy <= tol:
            break

    return avg_model, sigma2, weights


def _check_weights_equality(w1: Parameters, w2: Parameters) -> bool:
    """Check if two parameter vectors are identical."""
    return all(torch.equal(a, b) for a, b in zip(w1, w2))


def _find_reference_weights(
    reference: Parameters,
    candidates: List[Parameters],
) -> int:
    """Find index of reference weights in candidates list."""
    for idx, w in enumerate(candidates):
        if _check_weights_equality(reference, w):
            return idx
    raise ValueError("Reference weights not found in candidates list.")


def _aggregate_n_closest_weights(
    reference: Parameters,
    results: List[Tuple[Parameters, int]],
    beta_closest: int,
) -> Parameters:
    """Element-wise mean of the beta_closest values to reference (used by Bulyan)."""
    list_of_weights = [w for w, _ in results]
    aggregated = []

    for layer_id, layer_ref in enumerate(reference):
        layer_others = np.array([w[layer_id] for w in list_of_weights])
        diff = np.abs(layer_ref - layer_others)
        indices = np.argpartition(diff, kth=beta_closest - 1, axis=0)
        beta_weights = np.take_along_axis(layer_others, indices, axis=0)[:beta_closest]
        aggregated.append(np.mean(beta_weights, axis=0))

    return aggregated


def _flatten_weights(weights_list: List[Parameters]) -> torch.Tensor:
    """Stack a list of parameter vectors into a 2D tensor of shape (n_clients, n_params).

    Used by filters and strategies that need to operate on the full
    weight matrix rather than individual parameter vectors.
    """
    return torch.stack(weights_list, dim=0).float()
