"""
Module to train the vanilla autoencoder model.
"""

import math
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from src.models.autoencoder import AutoEncoder
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.training.trainer import train_one_epoch, validate


if __name__ == "__main__":
    # Create checkpoints dir to keep working
    Path("checkpoints").mkdir(exist_ok=True)

    # Take the device and additional hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    epochs = 100
    batch_size = 4
    num_workers = 2
    patience = 15
    epochs_without_improvement = 0

    # Initialize the wandb for logging
    wandb.init(
        project="video-anomaly-detection",
        name="m1-vanilla-ae-ped2",          # run name
        config={                            # hyperparameters
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": "vanilla-3d-ae",
            "bottleneck": "16:1",
            "subset": "ped2",
        },
    )

    # Components to complete training
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Split the clips within train data to train and val
    train_clips = list(range(13))   # [0..12]
    val_clips   = [13, 14, 15]

    # Datasets based on splits
    train_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="train",
                        clip_indices=train_clips, transform=transform)
    val_ds   = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="train",
                        clip_indices=val_clips, transform=transform)

    # Dataloaders based on splitted datasets
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Training loop
    best_val_loss = math.inf
    for epoch in range(epochs):
        # Calculate the average losses
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        avg_val_loss = validate(model, val_loader, criterion, device)

        # Print out the results of corresponding epoch
        print(f"Epoch: {epoch}")
        print("="*30)
        print(f"Average Train Loss: {avg_train_loss}")
        print(f"Average Val Loss: {avg_val_loss}")
        print("="*30)

        # Log the results to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],   # to track the LR in order to ensure schedling works appropriately
            "epoch": epoch,
        })

        # Save better model checkpoints and update the best_val_loss, otherwise stop the training after some patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the model
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, "checkpoints/ae_best.pt")
            print(f"Better model saved (val_loss={avg_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Schedule the LR
        scheduler.step()

    # Finish the run
    wandb.finish()