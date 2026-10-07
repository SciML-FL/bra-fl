"""Evaluation function to test the model performance."""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn


def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
    """Validate the model on the entire test set.
    
    :param model: The local model that needs to be evaluated.
    :param testloader: The dataloader of the dataset to use for evaluation.
    :param device: The device to evaluate the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model evaluation.
    :returns: Evaluation loss and accuracy of the model.
    """
    correct = 0
    total = 0
    loss = 0.0

    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss += criterion(outputs, target).item() * target.size(0)
            _, predicted = torch.max(outputs.data, 1)  
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    loss /= total
    return loss, accuracy, total


def evaluate_gan_classification(
        dis_model: nn.Module,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion: Optional[nn.Module],
        raw_output: bool = False
    ) -> Tuple[float, float, List[float]]:
    """Validate the model on the entire test set.
    
    :param dis_model: The local model that needs to be evaluated.
    :param testloader: The dataloader of the dataset to use for evaluation.
    :param device: The device to evaluate the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model evaluation.
    :returns: Evaluation loss and accuracy of the model.
    """
    # --- Multi-GPU: wrap with DataParallel if available ---
    if torch.cuda.device_count() > 1:
        if not isinstance(dis_model, nn.DataParallel):
            dis_model = nn.DataParallel(dis_model)

    # Stage the discriminator model to the run device
    if next(dis_model.parameters()).device != device:
        dis_model.to(device)

    dis_model.eval()
    criterion.eval()  # Set the criterion to evaluation mode if it has such a mode

    # Variables to hold evaluation stats
    total_samples = 0
    total_loss = 0.0
    # accu_cls = [0 for _ in range(num_classes)]
    total_correct = 0

    raw_outputs = [] if raw_output else None

    dis_model.eval()
    with torch.no_grad(): 
        # Generate samples with given z and l values
        # and use them to evaluate current model
        for samples, labels in testloader:
            samples, labels = samples.to(device), labels.to(device)

            dis_predict = dis_model(samples)
            _, preds = torch.max(dis_predict.data, dim=1)

            # Compute Loss
            g_loss = criterion(dis_predict, labels)
            total_loss += g_loss.item() * len(labels)
            total_samples += len(labels)

            # Compute total correct predictions for accuracy calculation
            total_correct += (preds == labels.argmax(dim=1)).float().sum().item()

            if raw_output:
                # Store raw outputs for further analysis
                raw_outputs.append(dis_predict.cpu().numpy())

    # Updates statistics
    avg_loss = total_loss / total_samples
    avg_accu = total_correct / total_samples

    # Unwrap before returning so callers get a clean model
    if isinstance(dis_model, nn.DataParallel):
        dis_model = dis_model.module

    return avg_loss, avg_accu, raw_outputs

def evaluate_gan_regression(
        dis_model: nn.Module,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion: Optional[nn.Module],
        raw_output: bool = False,
        aggregation: str = "mean"  # "mean" or "median"
    ) -> Tuple[float, float, List[float]]:
    """Validate a regression model on the synthetic validation set."""

    if torch.cuda.device_count() > 1:
        if not isinstance(dis_model, nn.DataParallel):
            dis_model = nn.DataParallel(dis_model)

    if next(dis_model.parameters()).device != device:
        dis_model.to(device)

    dis_model.eval()
    # criterion.eval()

    all_sample_losses = []  # Collect per-sample losses
    raw_outputs = [] if raw_output else None

    with torch.no_grad():
        for samples, labels in testloader:
            samples, labels = samples.to(device), labels.to(device)

            dis_predict = dis_model(samples)

            # Per-sample MAE (not MSE — linear, not quadratic)
            # per_sample = torch.abs(
            #     dis_predict.squeeze() - labels.squeeze()
            # )
            per_sample = criterion(dis_predict, labels) #.cpu()  # [batch_size]
            all_sample_losses.append(per_sample)

            if raw_output:
                raw_outputs.append(dis_predict.cpu().numpy())

    # Concatenate all per-sample losses across batches
    # all_sample_losses = torch.cat(all_sample_losses, dim=0)
    all_sample_losses = torch.tensor(all_sample_losses)

    # Aggregate
    if aggregation == "median":
        avg_loss = all_sample_losses.median().item()
    else:
        avg_loss = all_sample_losses.mean().item()

    # Regression "accuracy" — use something meaningful
    # e.g., fraction of predictions within 10% of target range
    # or just return 0.0 as placeholder
    avg_accu = 0.0

    if isinstance(dis_model, nn.DataParallel):
        dis_model = dis_model.module

    return avg_loss, avg_accu, raw_outputs
