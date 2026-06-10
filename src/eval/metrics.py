"""
Module for measuring metrics.
"""

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def compute_auc(all_scores, all_labels):
    """Frame-level ROC-AUC."""
    return roc_auc_score(all_labels, all_scores)


def compute_eer(all_scores, all_labels):
    """Equal Error Rate: The error at point FPR == FNR"""
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    
    # FPR and FNR's nearest index
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer