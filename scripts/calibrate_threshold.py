"""
Calibrate the anomaly threshold from the normal training distribution.

Runs the streaming predictor over the normal training clips (all normal by
construction), collects per-frame scores, and reports mean + k*std thresholds.
The chosen value goes into backend/main.py THRESHOLD.
"""

import glob
import os
import numpy as np
from src.inference.stream import process_frames


if __name__ == "__main__":
    onnx_path = "checkpoints/model.onnx"

    # Normal training clips — Train split is 100% normal (UCSD guarantee).
    # NOTE: use the TRAIN dirs (Train001, Train002, ...), not Test.
    train_root = "data/ucsd/raw/UCSDped2/Train"

    # Collect all clip directories under train_root
    search_pattern = os.path.join(train_root, "Train*")
    clip_dirs = sorted(glob.glob(search_pattern))

    # Run the stream over each clip, collect all non-None scores
    all_scores = []
    for clip_dir in clip_dirs:
        # Make sure the processed path is a directory
        if not os.path.isdir(clip_dir):
            continue
            
        # Process the frames
        scores = process_frames(clip_dir, onnx_path)
        
        # Filter out the None values
        if scores:
            valid_scores = [s for s in scores if s is not None]
            all_scores.extend(valid_scores)

    arr = np.array(all_scores)
    print(f"normal frames scored: {len(arr)}")
    print(f"mean: {arr.mean():.6f}")
    print(f"std:  {arr.std():.6f}")
    print(f"max:  {arr.max():.6f}")
    print(f"threshold (mean + 2*std): {arr.mean() + 2*arr.std():.6f}")
    print(f"threshold (mean + 3*std): {arr.mean() + 3*arr.std():.6f}")