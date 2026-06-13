"""
Evaluate the U-Net future-frame-prediction model (M3) on UCSD Ped2 test split.
Frame-level AUC + EER + error histogram.
"""

import torch
from src.models.predictor import UNetPredictor
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.inference.scoring import compute_prediction_errors, aggregate_all
from src.eval.visualization import plot_error_distribution
from src.eval.metrics import compute_auc, compute_eer


if __name__ == "__main__":
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model: UNetPredictor + prediction checkpoint
    model = UNetPredictor().to(device)
    ckpt = torch.load("checkpoints/pred_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Test data (all 12 clips, split=test, prediction mode)
    test_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="test",
                          transform=transform, mode="prediction")

    # Scoring
    results = compute_prediction_errors(model, test_ds, device)
    all_scores, all_labels = aggregate_all(results)

    # Histogram
    plot_error_distribution(all_scores, all_labels, save_path="docs/error_dist_pred.png")

    # AUC + EER
    auc = compute_auc(all_scores, all_labels)
    eer = compute_eer(all_scores, all_labels)
    print(f"FFP Frame-level AUC: {auc:.4f}")
    print(f"FFP EER: {eer:.4f}")