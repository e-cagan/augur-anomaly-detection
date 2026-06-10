"""
Module for visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_error_distribution(all_scores, all_labels, save_path="docs/error_dist.png"):
    """
    Function that plots the error distribution between normal and anomaly scores using label masking.
    """
    
    # Mask the labels
    normal_scores  = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1]

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.hist(normal_scores,  bins=50, alpha=0.6, label="Normal",  density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", density=True)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.title("Per-frame reconstruction error: normal vs anomaly")
    plt.legend()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")