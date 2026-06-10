"""
Evaluate the MemAE model on UCSD Ped2 test split.
Frame-level AUC + EER + error histogram + sparsity check.
"""

import torch
from src.models.memory_ae import MemoryAE
from src.data.ucsd_loader import UCSDDataset
from src.data.video_transforms import transform
from src.inference.scoring import compute_frame_errors, aggregate_all
from src.eval.visualization import plot_error_distribution
from src.eval.metrics import compute_auc, compute_eer


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Model: MemoryAE + memae checkpoint ──
    n_slots = 500
    model = MemoryAE(n_slots=n_slots).to(device)
    ckpt = torch.load("checkpoints/memae_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Test data (all 12 clip, split=test)
    test_ds = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="test", transform=transform)

    # Scoring
    results = compute_frame_errors(model, test_ds, device)
    all_scores, all_labels = aggregate_all(results)

    # Histogram
    plot_error_distribution(all_scores, all_labels, save_path="docs/error_dist_memae.png")

    # Sparsity
    model.eval()
    with torch.no_grad():
        sample, _ = test_ds[0]
        sample = sample.unsqueeze(0).to(device)   # (1, T, C, H, W)
        _, attn = model(sample)
        active_frac = (attn > 0).float().mean()
        active_per_q = (attn > 0).float().sum(dim=-1).mean()
        print(f"trained active fraction: {active_frac:.4f}")
        print(f"avg active slots/query: {active_per_q:.1f} / {n_slots}")
    
    # AUC + EER
    auc = compute_auc(all_scores, all_labels)
    eer = compute_eer(all_scores, all_labels)
    print(f"MemAE Frame-level AUC: {auc:.4f}")
    print(f"MemAE EER: {eer:.4f}")