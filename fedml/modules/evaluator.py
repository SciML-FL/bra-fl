"""Evaluation function to test the model performance."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

def evaluate(
        model,
        testloader: torch.utils.data.DataLoader,
        device: str,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[float, float]:
    
    """Validate the model on the entire test set.
    
    :param model: The local model that needs to be evaluated.
    :param testloader: The dataloader of the dataset to use for evaluation.
    :param device: The device to evaluate the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model evaluation.
    :returns: Evaluation loss and accuracy of the model.
    """
    if criterion is None: criterion = nn.CrossEntropyLoss()
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