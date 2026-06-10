"""
Module to evaluate implemented functions, models etc.
"""

import torch
from src.models.autoencoder import AutoEncoder
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.inference.scoring import compute_frame_errors, aggregate_all
from src.eval.visualization import plot_error_distribution
from src.eval.metrics import compute_auc, compute_eer


if __name__ == "__main__":
    # Device and model checkpoints to test on test set
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder().to(device)
    ckpt = torch.load("checkpoints/ae_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    # Test set
    test_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="test", transform=transform)
    results = compute_frame_errors(model, test_ds, device)
    all_scores, all_labels = aggregate_all(results)

    # Plot the error distribution
    plot_error_distribution(all_scores, all_labels)
    
    # AUC and EER computation
    auc = compute_auc(all_scores, all_labels)
    eer = compute_eer(all_scores, all_labels)
    print(f"Frame-level AUC: {auc:.4f}")
    print(f"EER: {eer:.4f}")