"""Training function to train the model for given number of epochs."""

from typing import Callable, Iterator
from logging import INFO

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedml.utils.logger import log
from fedml.utils.noise import NoiseDistribution
from fedml.training.lr_scheduler import get_lr_scheduler

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, current_loss: float):
        if self.best_loss is None or current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True

def train(
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
    ) -> None:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :returns: None.
    """
    total_examples = 0
    final_loss = 0.0
    final_accuracy = 0.0

    # Train the model
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0
        epoch_examples = 0

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero out the parameter gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Backward Pass and Optimization
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute statistics
            running_loss += loss.item() * labels.size(0)
            epoch_examples  += labels.size(0)
            running_corrects += torch.sum(preds == labels).item()

            # print(f"\rEpoch: {epoch+1}/{epochs}, Iteration: {i+1}/{len(trainloader)}, Loss: {loss.item() * inputs.size(0)}.", end="", flush=True)
            
        final_loss = running_loss / epoch_examples
        final_accuracy = running_corrects / epoch_examples
        total_examples += epoch_examples

    return total_examples, final_loss, final_accuracy

def train_batch(
        model: nn.Module,
        train_iterator: Iterator,
        num_rounds: int,
        device: str,  # pylint: disable=no-member
        criterion,
        optimizer,
    ) -> None:
    """Helper function to train the model using a single batch.

    :param model: The local model that needs to be trained.
    :param train_iterator: An iterator over the trainloader used to sample training batches.
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :returns: None.
    """
    model.train()
    num_examples = 0
    total_loss = 0.0
    total_corrects = 0

    for i in range(num_rounds):
        # Sample a batch
        inputs, targets = next(train_iterator)

        # Perform training step
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        num_examples += targets.size(0)
        total_loss += loss.item() * targets.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total_corrects += (predicted == targets).sum().item()

    return num_examples, total_loss / num_examples, total_corrects / num_examples

def train_generator(
        gen_model: nn.Module,
        dis_model: nn.Module,
        gen_optim,
        criterion,
        batch_size,
        iterations,
        noise_dist: NoiseDistribution,
        label_dist: NoiseDistribution,
        device,
        patience: int = 5,
        delta: float = 0.0,
        check_every: int = 100,
    ) -> nn.Module:
    """Train the generator."""
    """Helper function to train the model.

    :param gen_model: The generator part of the GAN model.
    :param dis_model: The discriminator part of the GAN model.
    :param gen_optim: The optimizer to use for model training.
    :param criterion: The loss function to use for model training.
    :param batch_size: The batchsize to use for generating random input data.
    :param epochs: Number of training rounds / epochs
    :param iterations_per_epoch: Number of iterations to run per epoch.
    :param device: The device to train the model on i.e. cpu or cuda. 
    :returns: The trained generator model is returned back.
    """

    # Wrap with DataParallel if multiple GPUs available
    # and we're not already wrapped
    if torch.cuda.device_count() > 1:
        if not isinstance(gen_model, nn.DataParallel):
            gen_model = nn.DataParallel(gen_model)
        if not isinstance(dis_model, nn.DataParallel):
            dis_model = nn.DataParallel(dis_model)

    if next(gen_model.parameters()).device != device:
        gen_model.to(device)

    if next(dis_model.parameters()).device != device:
        dis_model.to(device)
    
    lr_scheduler = get_lr_scheduler(
        optimizer=gen_optim, 
        total_epochs=iterations, 
        method="CUSTOM", 
        milestones=[x for x in range(0, iterations+1, 1000) if x != 0],
        gamma=0.99,
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    # Keep track of total loss
    total_loss = 0.0

    # Create learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=gen_optim, step_size=(iterations//2), gamma=0.1)
    gen_model.train()
    dis_model.eval()
    criterion.train()

    log(INFO, "\n"+"+"*50+"\n| Training Generator Model\n"+"+"*50)
    for iteration in range(1, iterations+1):
        gen_optim.zero_grad()

        # Generate a batch of random samples
        input_z = noise_dist.sample(batch_size)
        soft_targets, hard_labels_onehot = label_dist.sample(batch_size, return_condition=True)

        # Generate a batch of images using the generator model
        gen_images = gen_model(input_z, hard_labels_onehot)
        dis_predict = dis_model(gen_images)

        # Compute loss and perform optimization step
        g_loss = criterion(dis_predict, soft_targets).to(device)
        g_loss.backward()
        gen_optim.step()

        # Update total loss
        total_loss += g_loss.item()

        # Update learning rate
        lr_scheduler.step()
       
        # Checking for early stopping
        if iteration % check_every == 0:
            early_stopping.check_early_stop(total_loss/check_every)

            # Print early stopping status
            if early_stopping.stop_training:
                log(INFO, "-"*50)
                log(INFO, f"Early stopping at iteration {iteration} with loss {total_loss/check_every:.4f}")
                break

            total_loss = 0.0  # Reset total loss for next period

        # Print progress
        if iteration % 500 == 0:
            log(INFO, f"| Iteration {iteration:5d}    / {iterations:5d}: Loss = {g_loss.item():2.4f}")
    log(INFO, "+"*50)

    # Unwrap before returning so callers get a clean model
    if isinstance(gen_model, nn.DataParallel):
        return gen_model.module
    
    return gen_model


def backdoor_train(
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        trigger_func: Callable,
        target_label: int,
        poison_ratio: float,
        # batch_size: int,
    ) -> None:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :returns: None.
    """
    num_examples = 0

    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Add trigger to fraction of images
            poison_indices = random.sample(
                population=list(range(data[0].size(dim=0))), 
                k = int(poison_ratio*data[0].size(dim=0))
            )

            data[0][poison_indices] = trigger_func(data[0][poison_indices])
            data[1][poison_indices] = target_label

            # Perform training with poisoned images
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_examples += labels.size(0)

    return num_examples

def backdoor_train_2(
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: str,  # pylint: disable=no-member
        learning_rate: float,
        criterion,
        optimizer,
        base_model: nn.Module,
        eta: float,
        # batch_size: int,
    ) -> None:
    """Helper function to train the model.

    :param model: The local model that needs to be trained.
    :param trainloader: The dataloader of the dataset to use for training.
    :param epochs: Number of training rounds / epochs
    :param device: The device to train the model on i.e. cpu or cuda. 
    :param learning_rate: The initial learning rate the optimizer is using.
    :param criterion: The loss function to use for model training.
    :param optimizer: The optimizer to use for model training.
    :returns: None.
    """
    num_examples = 0

    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Perform training with poisoned images
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            current = model.get_weights()
            loss = (1.0 - eta) * loss + eta * torch.linalg.norm(base_model - current)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_examples += labels.size(0)

    return num_examples