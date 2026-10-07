"""Factory function for loss/criterion functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_criterion(criterion_str: str, **kwargs) -> nn.Module:
    """Return the requested loss function."""
    if criterion_str == "CROSSENTROPY":
        return nn.CrossEntropyLoss()
    elif criterion_str == "NLLL":
        return nn.NLLLoss()
    elif criterion_str == "MSELOSS":
        return nn.MSELoss()
    elif criterion_str == "CLASSIFICATION-GENLOSS":
        return ClassificationGenLoss(**kwargs)
    elif criterion_str == "REGRESSION-GENLOSS":
        return RegressionGenLoss(**kwargs)
    else:
        raise ValueError(f"Invalid criterion '{criterion_str}' requested.")

class ClassificationGenLoss(nn.Module):
    """Combined KL-divergence + cross-entropy loss for the generator."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self._train_mode = True
    
    def eval(self):
        self._train_mode = False   

    def train(self):
        self._train_mode = True   

    def forward(self, discriminator_output, target_labels):
        ce_loss = F.cross_entropy(discriminator_output, target_labels.argmax(dim=1))
        if self._train_mode:
            kl_loss = F.kl_div(
                F.log_softmax(discriminator_output, dim=1),
                target_labels,
                reduction="batchmean",
            )
            total_loss = self.alpha * ce_loss + self.beta * kl_loss
        else:
            total_loss = ce_loss
        return total_loss

class RegressionGenLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, bandwidth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # MMD weight
        self.bandwidth = bandwidth
        self._train_mode = True
    
    def eval(self):
        self._train_mode = False   

    def train(self):
        self._train_mode = True   

    def mse_loss(self, predicted, target):
        """Regression equivalent of cross-entropy."""
        return F.mse_loss(predicted.squeeze(), target.squeeze())

    def mmd_loss(self, predicted_batch, target_batch):
        """
        Maximum Mean Discrepancy — regression analogue of KL divergence.
        Measures whether the distribution of predicted UPDRS scores over
        the batch matches the distribution of conditioning target values.
        Uses an RBF kernel.
        """
        def rbf_kernel(a, b, bandwidth):
            diff = a.unsqueeze(1) - b.unsqueeze(0)  # [N, M, 1]
            return torch.exp(-diff.pow(2).sum(-1) / (2 * bandwidth ** 2))

        k_pp = rbf_kernel(predicted_batch, predicted_batch, self.bandwidth).mean()
        k_tt = rbf_kernel(target_batch, target_batch, self.bandwidth).mean()
        k_pt = rbf_kernel(predicted_batch, target_batch, self.bandwidth).mean()
        return k_pp + k_tt - 2 * k_pt

    def forward(self, predicted_values, target_values):
        mse_loss = self.mse_loss(predicted_values, target_values)

        if self._train_mode:
            mmd_loss = self.mmd_loss(predicted_values, target_values)
            total_loss = self.alpha * mse_loss + self.beta * mmd_loss
        else:
            total_loss = mse_loss

        return total_loss
