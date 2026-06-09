"""
Module for pytorch training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.MSELoss, optimizer: optim.Optimizer, device: str) -> float:
    """
    Function to train the model on a single epoch. Returns the average training loss.
    """
    
    # Set the model on train mode
    model.train()
    running_loss = 0.0

    # Iterate within dataloader
    for tensors, labels in dataloader:
        # Move the tensors and labels to device
        tensors, labels = tensors.to(device), labels.to(device)
        
        # Core 5-step optimization
        optimizer.zero_grad(set_to_none=True) # set_to_none=True to optimize memory allocation
        outputs = model(tensors)
        loss = criterion(outputs, tensors)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tensors.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.MSELoss, device: str) -> float:
    """
    Function to evaluate the model. Returns the average validation loss.
    """

    # Set the model on eval mode
    model.eval()
    running_loss = 0.0

    # Deactivate the gradients for memory optimization
    # NOTE: inference_mode eliminates gradient overhead, making it faster than no_grad but we'll need the tensors for the future so, no_grad is safer to use
    with torch.no_grad():
        # Iterate within dataloader
        for tensors, labels in dataloader:
            # Move the tensors and labels to device
            tensors, labels = tensors.to(device), labels.to(device)

            outputs = model(tensors)
            loss = criterion(outputs, tensors)

            running_loss += loss.item() * tensors.size(0)

    return running_loss / len(dataloader.dataset)