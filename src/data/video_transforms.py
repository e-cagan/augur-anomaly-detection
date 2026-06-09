"""
Video transforms for UCSD anomaly detection.
M1 minimal: Resize + Normalize. No random augmentation.
"""

import torch
from torchvision.transforms import v2


# Input: float32 tensor, shape (T, 1, H, W), range [0, 1]
# Output: float32 tensor, shape (T, 1, 128, 128), range ~[-1, 1]
transform = v2.Compose([
    v2.Resize(size=(128, 128), antialias=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])


if __name__ == "__main__":
    # Dummy window
    x = torch.rand(16, 1, 240, 360)  # (T, C, H, W)
    print(f"Before: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
    
    y = transform(x)
    print(f"After:  shape={y.shape}, range=[{y.min():.3f}, {y.max():.3f}]")
    
    # Mean/std after normalize, beklenen ~0 mean ~0.577 std (uniform [-1,1] için)
    print(f"After mean={y.mean():.3f}, std={y.std():.3f}")