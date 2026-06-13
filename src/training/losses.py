"""
Module for custom loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemAELoss(nn.Module):
    """MSE reconstruction + entropy regularization on attention weights."""

    def __init__(self, entropy_weight: float = 0.0002):
        super().__init__()
        self.entropy_weight = entropy_weight   # alpha
        self.mse = nn.MSELoss()

    def forward(self, recon, target, attn):
        """
        Args:
            recon:  (B, T, C, H, W) reconstruction
            target: (B, T, C, H, W) input
            attn:   (B, n_queries, N) attention weights
        Returns:
            total_loss, (recon_loss, entropy_loss)  # ayrı logla
        """
        eps = 1e-12

        # Reconstruction
        recon_loss = self.mse(recon, target)

        # Entropy: per-query entropy, then mean
        # E = mean( -sum_i ( w_i * log(w_i + eps) ) )
        entropy = (-(attn * torch.log(attn + eps)).sum(dim=-1)).mean()

        # Sum
        total = recon_loss + self.entropy_weight * entropy

        return total, (recon_loss, entropy)
    

class PredictionLoss(nn.Module):
    """Intensity (L2) + gradient loss for future frame prediction."""

    def __init__(self, grad_weight: float = 1.0):
        super().__init__()
        self.grad_weight = grad_weight   # lambda_grad

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 1, H, W) predicted frame
            target: (B, 1, H, W) ground truth frame
        Returns:
            total, (intensity, gradient)
        """
        # Intensity (L2)
        intensity = F.mse_loss(pred, target)

        # Gradient loss
        ## Horizontal (x) gradient (last axis = W)
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        ## Vertical (y) gradient (the axis before last = H)
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        ## loss: gradient differences
        gradient = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Total
        total = intensity + self.grad_weight * gradient

        return total, (intensity, gradient)


if __name__ == "__main__":
    # Smoke test
    # loss_fn = MemAELoss()
    # recon = torch.randn(2, 16, 1, 128, 128)
    # target = torch.randn(2, 16, 1, 128, 128)
    # attn = torch.softmax(torch.randn(2, 1024, 2000), dim=-1)   # valid distribution
    # total, (rl, ent) = loss_fn(recon, target, attn)
    # print(f"total: {total.item():.4f}, recon: {rl.item():.4f}, entropy: {ent.item():.4f}")

    # Prediction loss smoke test
    loss_pred = PredictionLoss()
    pred = torch.randn(2, 1, 128, 128)
    target = torch.randn(2, 1, 128, 128)
    total, (inten, grad) = loss_pred(pred, target)
    print(f"total: {total.item():.4f}, intensity: {inten.item():.4f}, gradient: {grad.item():.4f}")