"""Metrics aggregation functions for fit and evaluate rounds."""

from typing import Dict, List, Optional, Tuple
import torch
from fedml.utils import Metrics


def aggregate_fit_metrics(
    fit_metrics: List[Tuple[int, Metrics]],
    selected: Optional[List[int]] = None,
    weight_pi=None,
    rho_hat=None,
    update_norms: Optional[Dict] = None,
    scale_diagnostics: Optional[Dict] = None,
) -> Metrics:
    """Aggregate per-client fit metrics into a single metrics dict."""
    aggregated = {
        "sampled": [],
        "train_accu": {},
        "train_loss": {},
        "test_accu": {},
        "test_loss": {},
        "attacking": {},
        "client_type": {},
        "fit_duration": {},
        "num_examples": {},
    }

    if selected is not None:
        aggregated["selected"] = {}

    if weight_pi is not None:
        aggregated["weight_pi"] = {}
        if torch.is_tensor(weight_pi):
            weight_pi = weight_pi.cpu().detach().numpy()

    if rho_hat is not None:
        if torch.is_tensor(rho_hat):
            rho_hat = float(rho_hat.cpu().detach())
        aggregated["rho_hat"] = float(rho_hat)

    if scale_diagnostics:
        aggregated.update(scale_diagnostics)

    if update_norms:
        for key, value in update_norms.items():
            aggregated[key] = value

    for indx, (num_examples, client_dict) in enumerate(fit_metrics):
        cid = f"client_{client_dict['client_id']}"
        aggregated["sampled"].append(client_dict["client_id"])
        aggregated["train_accu"][cid] = client_dict["train_accu"]
        aggregated["train_loss"][cid] = client_dict["train_loss"]
        aggregated["test_accu"][cid] = client_dict["test_accu"]
        aggregated["test_loss"][cid] = client_dict["test_loss"]
        aggregated["attacking"][cid] = client_dict["attacking"]
        aggregated["client_type"][cid] = client_dict["client_type"]
        aggregated["fit_duration"][cid] = client_dict["fit_duration"]
        aggregated["num_examples"][cid] = num_examples
        if selected is not None:
            aggregated["selected"][cid] = indx in selected
        if weight_pi is not None:
            aggregated["weight_pi"][cid] = weight_pi[indx]

    return aggregated


def aggregate_evaluate_metrics(
    eval_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """Aggregate per-client evaluation metrics into a single metrics dict."""
    aggregated = {
        "train_samples": [],
        "train_success": [],
        "train_asr": [],
        "test_samples": [],
        "test_success": [],
        "test_asr": [],
    }

    for _, client_dict in eval_metrics:
        if "train_samples" in client_dict:
            aggregated["train_samples"].append(client_dict["train_samples"])
            aggregated["train_success"].append(client_dict["train_success"])
            aggregated["train_asr"].append(client_dict["train_asr"])
            aggregated["test_samples"].append(client_dict["test_samples"])
            aggregated["test_success"].append(client_dict["test_success"])
            aggregated["test_asr"].append(client_dict["test_asr"])

    return aggregated
