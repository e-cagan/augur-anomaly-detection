"""
Module for per-frame anomaly scoring on UCSD test split.

Pipeline: model reconstruction -> per-frame error -> overlapping-window
averaging -> per-clip frame-aligned anomaly scores.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from src.data.ucsd_loader import UCSDDataset
from src.models.autoencoder import AutoEncoder
from src.data.video_transforms import transform


def smooth_scores(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Temporal Gaussian smoothing on a single clip's per-frame scores."""
    return gaussian_filter1d(scores, sigma=sigma)


def compute_frame_errors(model: nn.Module, dataset: UCSDDataset, device: str) -> dict:
    """
    Compute per-frame reconstruction error for every clip in the test set,
    averaging across overlapping windows.

    Returns:
        dict mapping clip_idx -> (scores, labels)
        - scores: np.ndarray shape (n_frames,), avg reconstruction error per frame
        - labels: np.ndarray shape (n_frames,), 0/1 ground truth per frame
    """
    model.eval()

    # Prepare an accumulator for every clip
    error_sum = {}
    count = {}
    for clip_idx in range(len(dataset.clips)):
        n_frames = len(dataset.clips[clip_idx])
        error_sum[clip_idx] = np.zeros(n_frames, dtype=np.float64)
        count[clip_idx] = np.zeros(n_frames, dtype=np.float64)

    # Give out the windows towards the model
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (window, _labels) in enumerate(loader):
            # Window shape: (1, T, C, H, W)  -- batch size 1
            window = window.to(device)

            # Reconstruction
            recon = model(window)

            # Calculate per-frame error with taking the mean based on (C,H,W) channels
            per_frame_err = torch.mean(((window - recon)**2), dim=(0, 2, 3, 4)).cpu().numpy()   # shape: (T,)

            # Take a particular window from a particular clip
            clip_idx, start_frame = dataset.windows[idx]

            # For every t, global frame = start_frame + t
            error_sum[clip_idx][start_frame : start_frame + dataset.window_size] += per_frame_err
            count[clip_idx][start_frame : start_frame + dataset.window_size] += 1

    # Ortalama al + ground truth'u hizala
    results = {}
    for clip_idx in error_sum:
        # Counts and errors
        counts = count[clip_idx]
        errs = error_sum[clip_idx]

        # Log the number of frames that aren't valid
        print(f"clip {clip_idx}: {(counts==0).sum()} frames with no window coverage")

        # Valid frame filter
        valid = counts > 0
        
        # Take out the average which gives the result of average error
        scores = errs[valid] / counts[valid]          # Only valid frames
        scores = smooth_scores(scores, sigma=1.0)     # Clip based smoothing
        labels = dataset.labels[clip_idx][valid]      # Apply same mask

        results[clip_idx] = (scores, labels)

    return results


def aggregate_all(results: dict) -> tuple:
    """
    Flatten per-clip results into two 1D arrays for global AUC.

    Returns:
        all_scores: np.ndarray (total_frames,)
        all_labels: np.ndarray (total_frames,)
    """
    scores_list = []
    labels_list = []

    # Append corresponding clip's (scores, labels) by order
    for clip_idx in results:
        scores, labels = results[clip_idx]
        scores_list.append(scores)
        labels_list.append(labels)

    # Concatenate the results on 1D numpy array
    all_scores = np.concatenate(scores_list)
    all_labels = np.concatenate(labels_list)

    return all_scores, all_labels


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Modeli yükle (eğittiğin best checkpoint)
    model = AutoEncoder().to(device)
    ckpt = torch.load("checkpoints/ae_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Test dataset — clip_indices YOK (tüm 12 clip), split="test"
    test_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="test", transform=transform)

    results = compute_frame_errors(model, test_ds, device)
    all_scores, all_labels = aggregate_all(results)

    # Sanity check
    print(f"shape: {all_scores.shape}, {all_labels.shape}")          # same, 1D
    print(f"anomaly frames: {all_labels.sum()}/{len(all_labels)}")

    normal_mean  = all_scores[all_labels == 0].mean()
    anomaly_mean = all_scores[all_labels == 1].mean()
    print(f"normal mean error:  {normal_mean:.6f}")
    print(f"anomaly mean error: {anomaly_mean:.6f}")