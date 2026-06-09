"""
Module for dataset and dataloaders of UCSD dataset.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.video_transforms import transform


class UCSDDataset(Dataset):
    """
    UCSD Anomaly Detection Dataset.
    
    Train: only normal clips.
    Test: clips with frame-level ground truth annotations.
    
    Args:
        root: Dataset root path (containing UCSDped1/, UCSDped2/)
        subset: 'Ped1' or 'Ped2'
        split: 'train' or 'test'
        window_size: Number of frames per sample (sliding window)
        stride: Stride between windows
        transform: Optional transform applied to each frame
    """
    
    def __init__(
        self,
        root: str,
        subset: str = "Ped2",
        split: str = "train",
        window_size: int = 16,
        stride: int = 8,
        transform: Optional[callable] = None,
        clip_indices: Optional[List[int]] = None
    ):
        super().__init__()
        self.root = Path(root)
        self.subset = subset.lower()
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.clip_indices = clip_indices

        # Subset check
        assert subset in ("ped1", "ped2"), f"subset must be ped1 or ped2, got {subset}"

        # Read the subset and store the clip directories
        self.subset_split = self.root / f"UCSD{self.subset}" / f"{split.title()}"

        # Sanity check to ensure the files and clip directories exist
        if not self.subset_split.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.subset_split}")
        
        self.clip_dirs = sorted([
            d for d in self.subset_split.iterdir() 
            if d.is_dir() and not d.name.endswith("_gt")
        ])

        # Filter out the clips
        if self.clip_indices is not None:
            self.clip_dirs = [self.clip_dirs[i] for i in self.clip_indices]
        
        if len(self.clip_dirs) == 0:
            raise RuntimeError(f"No clip directories found in {self.subset_split}")

        # Collect the clip paths
        self.clips = []
        for clip_dir in self.clip_dirs:  # clip_dir = Path("Train001")
            frame_paths = sorted(clip_dir.glob("*.tif"))  # liste of frame paths
            frames = np.stack([np.array(Image.open(p)) for p in frame_paths])
            self.clips.append(frames)

        # Create labels based on split
        if self.split == "test":
            m_file = self.subset_split / f"UCSD{subset}.m"  # path case dikkat
            content = m_file.read_text()
            matches = re.findall(r"\[(\d+):(\d+)\]", content)
            
            self.labels = []
            for clip_idx, (start_str, end_str) in enumerate(matches):
                start, end = int(start_str), int(end_str)
                n_frames = len(self.clips[clip_idx])  # clip's frame length
                label = np.zeros(n_frames, dtype=np.int64)
                label[start-1:end] = 1  # 1-indexed -> 0-indexed slice
                self.labels.append(label)
        else:
            self.labels = None  # train, no label

        # Collect the window indexes
        self.windows = []  # list of (clip_idx, start_frame)
        for clip_idx, frames in enumerate(self.clips):
            n_frames = len(frames)
            for start in range(0, n_frames - window_size + 1, stride):
                self.windows.append((clip_idx, start))

    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
           frames: (T, C, H, W) tensor
           label: (T,) tensor of 0/1 (train: all zeros, test: from gt)
        """
        # Read frames and label
        clip_idx, start_frame = self.windows[idx]
        
        # Take the frames within frame range
        window_frames = self.clips[clip_idx][start_frame : start_frame + self.window_size] # shape: (T, H, W) uint8

        # Check labels based on split
        if self.split == "test":
            labels_np = self.labels[clip_idx][start_frame : start_frame + self.window_size]
            labels = torch.from_numpy(labels_np)  # int64 tensor
        else:
            labels = torch.zeros(self.window_size, dtype=torch.long)

        # Convert window array to tensor and reshape it
        window_tensor = torch.from_numpy(window_frames).float() / 255.0
        window_tensor = window_tensor.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
        
        # Check for transforms
        if self.transform is not None:
            window_tensor = self.transform(window_tensor)

        return window_tensor, labels

if __name__ == "__main__":
    # Run sanity check
    train_clips = [0,1,2,3,4,5,6,7,8,9,10,11,12]   # 13 clip
    val_clips   = [13,14,15]                       # 3 clip

    # Train
    ds_train = UCSDDataset(root="data/ucsd/raw", subset="ped2", clip_indices=train_clips, transform=transform, split="train")
    print(f"Train: {len(ds_train.clips)} clips, {len(ds_train)} windows")
    print(f"First clip shape: {ds_train.clips[0].shape}")

    # Validation
    ds_val = UCSDDataset(root="data/ucsd/raw", subset="ped2", clip_indices=val_clips, transform=transform, split="train")
    print(f"Val: {len(ds_val.clips)} clips, {len(ds_val)} windows")
    print(f"Val labels: {ds_val.labels}") # Should be None

    # Test
    ds_test = UCSDDataset(root="data/ucsd/raw", subset="ped2", split="test", transform=transform)
    print(f"Test: {len(ds_test.clips)} clips, {len(ds_test)} windows")
    print(f"First label sum: {ds_test.labels[0].sum()}/{len(ds_test.labels[0])}")

    # Test getitem
    sample, label = ds_train[0]
    print(f"\nSample 0 (train):")
    print(f"  Sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"  Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"  Label shape: {label.shape}, sum: {label.sum()}")

    sample, label = ds_val[0]
    print(f"\nSample 0 (test):")
    print(f"  Sample shape: {sample.shape}")
    print(f"  Label shape: {label.shape}, sum: {label.sum()}")

    # Random middle sample
    sample, label = ds_train[len(ds_train) // 2]
    print(f"\nMiddle train sample shape: {sample.shape}")

    # Transform check
    print(sample.shape)  # torch.Size([16, 1, 256, 256])