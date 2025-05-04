"""Training function to train the model for given number of epochs."""

from typing import Callable
from logging import INFO

import random
import torch
import torch.nn as nn

from fedml.common import log

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
    # Define loss and optimizer
    # log(
    #     INFO,
    #     f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each"
    # )

    num_examples = 0

    model.train()
    # Train the model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
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