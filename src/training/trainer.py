"""
Module for pytorch training and validation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.training.losses import MemAELoss, PredictionLoss


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.MSELoss, optimizer: optim.Optimizer, device: str) -> float:
    """
    Function to train the vanilla autoencoder model on a single epoch. Returns the average training loss.
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
    Function to evaluate the vanilla autoencoder model. Returns the average validation loss.
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


def train_one_epoch_memae(model: nn.Module, dataloader: DataLoader, criterion: MemAELoss, optimizer: optim.Optimizer, device: str) -> tuple:
    """
    Function to train the model on a single epoch for memory augmented autoencoder model. Returns the average training loss.
    """
    
    # Set the model on train mode
    model.train()
    running_total = 0.0
    running_recon = 0.0
    running_entropy = 0.0

    # Iterate within dataloader
    for tensors, labels in dataloader:
        # Move the tensors and labels to device
        tensors, labels = tensors.to(device), labels.to(device)
        
        # Core 5-step optimization
        optimizer.zero_grad(set_to_none=True) # set_to_none=True to optimize memory allocation
        recon, attn = model(tensors)
        loss, (recon_loss, entropy) = criterion(recon, tensors, attn)
        loss.backward()
        optimizer.step()

        # Scale using batch size
        bs = tensors.size(0)
        running_total += loss.item() * bs
        running_recon += recon_loss.item() * bs
        running_entropy += entropy.item() * bs

    n = len(dataloader.dataset)
    return running_total / n, running_recon / n, running_entropy / n


def validate_memae(model: nn.Module, dataloader: DataLoader, criterion: MemAELoss, device: str) -> tuple:
    """
    Function to evaluate the memory augmented autoencoder model. Returns the average validation loss.
    """

    # Set the model on eval mode
    model.eval()
    running_total = 0.0
    running_recon = 0.0
    running_entropy = 0.0

    # Deactivate the gradients for memory optimization
    # NOTE: inference_mode eliminates gradient overhead, making it faster than no_grad but we'll need the tensors for the future so, no_grad is safer to use
    with torch.no_grad():
        # Iterate within dataloader
        for tensors, labels in dataloader:
            # Move the tensors and labels to device
            tensors, labels = tensors.to(device), labels.to(device)

            recon, attn = model(tensors)
            loss, (recon_loss, entropy) = criterion(recon, tensors, attn)

            # Scale using batch size
            bs = tensors.size(0)
            running_total += loss.item() * bs
            running_recon += recon_loss.item() * bs
            running_entropy += entropy.item() * bs

    n = len(dataloader.dataset)
    return running_total / n, running_recon / n, running_entropy / n


def train_one_epoch_pred(model: nn.Module, dataloader: DataLoader, criterion: PredictionLoss, optimizer: optim.Optimizer, device: str) -> tuple:
    """
    Function to train our prediction model.
    """
    
    model.train()
    running_total = 0.0
    running_intensity = 0.0
    running_gradient = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)                                   # (B,1,H,W), single tensor
        loss, (intensity, gradient) = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        running_total += loss.item() * bs
        running_intensity += intensity.item() * bs
        running_gradient += gradient.item() * bs

    n = len(dataloader.dataset)
    return running_total / n, running_intensity / n, running_gradient / n


def validate_pred(model: nn.Module, dataloader: DataLoader, criterion: PredictionLoss, device: str) -> tuple:
    """
    Function to evaluate the prediction model. Returns the average validation, intensity and gradient losses.
    """

    # Set the model on eval mode
    model.eval()
    running_total = 0.0
    running_intensity = 0.0
    running_gradient = 0.0

    with torch.no_grad():
        # Iterate within dataloader
        for inputs, targets in dataloader:
            # Move the tensors and labels to device
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss, (intensity, gradient) = criterion(preds, targets)

            # Scale using batch size
            bs = inputs.size(0)
            running_total += loss.item() * bs
            running_intensity += intensity.item() * bs
            running_gradient += gradient.item() * bs

    n = len(dataloader.dataset)
    return running_total / n, running_intensity / n, running_gradient / n