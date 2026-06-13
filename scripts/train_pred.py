"""
Module to train the U-Net future-frame-prediction model (M3).
"""

import math
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from src.models.predictor import UNetPredictor
from src.training.losses import PredictionLoss
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.training.trainer import train_one_epoch_pred, validate_pred


if __name__ == "__main__":
    # Create checkpoints dir to keep working
    Path("checkpoints").mkdir(exist_ok=True)

    # Device and hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    grad_weight = 1.0
    epochs = 100
    batch_size = 4
    num_workers = 2
    patience = 15
    epochs_without_improvement = 0

    # wandb logging
    wandb.init(
        project="video-anomaly-detection",
        name="m3-ffp-ped2",                 # run name
        config={
            "lr": lr,
            "grad_weight": grad_weight,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": "unet-ffp",
            "input_frames": 15,
            "subset": "ped2",
        },
    )

    # Components
    model = UNetPredictor().to(device)
    criterion = PredictionLoss(grad_weight=grad_weight)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Clip-level train/val split
    train_clips = list(range(13))   # [0..12]
    val_clips   = [13, 14, 15]

    # Datasets -- mode="prediction" is REQUIRED (returns (input_15, target_1))
    train_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="train",
                           clip_indices=train_clips, transform=transform,
                           mode="prediction")
    val_ds   = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="train",
                           clip_indices=val_clips, transform=transform,
                           mode="prediction")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Training loop
    best_val_intensity = math.inf
    for epoch in range(epochs):
        train_total, train_intensity, train_grad = train_one_epoch_pred(model, train_loader, criterion, optimizer, device)
        val_total, val_intensity, val_grad = validate_pred(model, val_loader, criterion, device)

        print(f"Epoch: {epoch}")
        print("=" * 30)
        print(f"Train total: {train_total:.6f} | intensity: {train_intensity:.6f} | grad: {train_grad:.6f}")
        print(f"Val   total: {val_total:.6f} | intensity: {val_intensity:.6f} | grad: {val_grad:.6f}")
        print("=" * 30)

        # Log (before break, so the early-stop epoch is logged too)
        wandb.log({
            "train_total": train_total,
            "train_intensity": train_intensity,
            "train_grad": train_grad,
            "val_total": val_total,
            "val_intensity": val_intensity,
            "val_grad": val_grad,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        # Scheduler step (before break, stays consistent)
        scheduler.step()

        # Checkpoint on best val INTENSITY (prediction quality); grad is a regularizer
        if val_intensity < best_val_intensity:
            best_val_intensity = val_intensity
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_intensity": val_intensity,
                "val_total": val_total,
            }, "checkpoints/pred_best.pt")
            print(f"Better model saved (val_intensity={val_intensity:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    wandb.finish()