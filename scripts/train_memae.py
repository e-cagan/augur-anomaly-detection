"""
Module to train the memory augmented autoencoder model.
"""

import math
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from src.models.memory_ae import MemoryAE
from src.training.losses import MemAELoss
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.training.trainer import train_one_epoch_memae, validate_memae


if __name__ == "__main__":
    # Create checkpoints dir to keep working
    Path("checkpoints").mkdir(exist_ok=True)

    # Take the device and additional hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    alpha = 2e-4
    epochs = 100
    n_slots = 500
    batch_size = 4
    num_workers = 2
    patience = 15
    epochs_without_improvement = 0

    # Initialize the wandb for logging
    wandb.init(
        project="video-anomaly-detection",
        name="m2-memae-ped2",               # run name
        config={                            # hyperparameters
            "lr": lr,
            "alpha": alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": "3d-memae",
            "bottleneck": "16:1",
            "subset": "ped2",
        },
    )

    # Components to complete training
    model = MemoryAE(n_slots=n_slots).to(device)
    criterion = MemAELoss(entropy_weight=alpha)
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
    best_val_recon = math.inf
    for epoch in range(epochs):
        # Calculate the average losses
        train_total, train_recon, train_entropy = train_one_epoch_memae(model, train_loader, criterion, optimizer, device)
        val_total, val_recon, val_entropy = validate_memae(model, val_loader, criterion, device)

        # Print out the results of corresponding epoch
        print(f"Epoch: {epoch}")
        print("="*30)
        print(f"Average Train Loss: {train_total}")
        print(f"Average Val Loss: {val_total}")
        print("="*30)

        # Log the results to wandb
        wandb.log({
            "train_total": train_total,
            "train_recon": train_recon,
            "train_entropy": train_entropy,
            "val_total": val_total,
            "val_recon": val_recon,
            "val_entropy": val_entropy,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        # Save better model checkpoints and update the best_val_loss, otherwise stop the training after some patience
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            epochs_without_improvement = 0
            # Save the model
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_total,
            }, "checkpoints/memae_best.pt")
            print(f"Better model saved (val_loss={val_total:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Schedule the LR
        scheduler.step()

    # Finish the run
    wandb.finish()