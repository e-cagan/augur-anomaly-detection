"""
Streaming anomaly detection for the M3 predictor.
Rolling 15-frame buffer -> predict next frame -> per-frame anomaly score (+ heatmap).
"""

import os
import glob
import numpy as np
import cv2
import onnxruntime as ort
from collections import deque
from src.data.video_transforms import transform   # same transform as training
import torch


class AnomalyStream:
    def __init__(self, onnx_path: str, buffer_size: int = 15):
        self.sess = ort.InferenceSession(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)   # last 15 preprocessed frame

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Raw video frame (H,W,3 BGR) -> training format (1, H, W) [-1,1] grayscale 128x128.
        TODO:
          - BGR -> grayscale (cv2.cvtColor)
          - to tensor, float, /255 -> [0,1]
          - shape (1, H, W) -- unsqueeze channel
          - apply `transform` (resize 128, normalize [-1,1])
        Return: tensor (1, 128, 128), single frame
        """
        # BGR -> grayscale (numpy, uint8, (H,W))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # numpy -> torch, float, [0,1]
        gray_t = torch.from_numpy(gray).float() / 255.0   # (H, W)

        # transform expects (T, C, H, W) — make it (1, 1, H, W): T=1, C=1
        gray_t = gray_t.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)

        # apply training transform (resize 128, normalize [-1,1])
        gray_t = transform(gray_t)                          # (1, 1, 128, 128)

        # drop the T axis -> (1, 128, 128) = (C, H, W) for buffering
        return gray_t.squeeze(0)                            # (1, 128, 128)

    def push(self, frame_bgr: np.ndarray):
        """
        Add one frame. Returns (score, heatmap) or None if still warming up.
        """
        frame_t = self.preprocess(frame_bgr)                    # (1,128,128)

        # Cold start: wait for buffer to warm up (first 15 frames)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(frame_t)
            return None   # warming up

        # buffer: 15 frame, every frame shaped (1,128,128) = (C,H,W)
        # stack -> (15, 1, 128, 128), then batch axis -> (1, 15, 1, 128, 128)
        stacked = torch.stack(list(self.buffer))                # (15, 1, 128, 128)
        inp = stacked.unsqueeze(0).numpy()                      # (1, 15, 1, 128, 128) numpy
        pred = self.sess.run(None, {self.input_name: inp})[0]   # (1,1,128,128)

        # Real frame (target) = this new frame
        actual = frame_t.numpy()[None, ...]                     # (1,1,128,128) -- shape matching

        # Per-pixel error -> heatmap, mean -> score
        error_map = (pred - actual) ** 2                        # (1, 1, 128, 128)
        heatmap = error_map[0, 0]                               # (128, 128) — spatial harita, frontend için
        score = float(error_map.mean())                         # scaler anomaly score

        # Update the buffer
        self.buffer.append(frame_t)

        return score, heatmap


def process_video(video_path: str, onnx_path: str):
    """Run the stream over a video file, collect per-frame scores."""
    stream = AnomalyStream(onnx_path)
    cap = cv2.VideoCapture(video_path)

    scores = []
    frame_idx = 0
    while True:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        result = stream.push(frame)
        if result is None:
            scores.append(None)   # warming up
        else:
            score, heatmap = result
            scores.append(score)
        frame_idx += 1

    cap.release()
    return scores


def process_frames(frame_dir: str, onnx_path: str):
    """
    Run the stream over a directory of .tif frames (UCSD format).
    Mirrors process_video but reads ordered image files instead of decoding video.
    Used to verify the streaming pipeline matches eval scoring.
    """
    stream = AnomalyStream(onnx_path)

    # UCSD frames: sorted .tif files in the clip dir
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.tif")))

    scores = []
    for path in frame_paths:
        # cv2.imread reads as BGR (H,W,3) even for grayscale .tif -> preprocess handles BGR->gray
        frame = cv2.imread(path)
        result = stream.push(frame)
        if result is None:
            scores.append(None)              # warming up (first 15)
        else:
            score, heatmap = result
            scores.append(score)

    return scores


if __name__ == "__main__":
    # Smoke test for streaming
    onnx_path = "checkpoints/model.onnx"
    clip_dir = "data/ucsd/raw/UCSDped2/Test/Test001"   # a test clip

    scores = process_frames(clip_dir, onnx_path)
    valid = [s for s in scores if s is not None]

    print(f"total frames: {len(scores)}")
    print(f"scored: {len(valid)}, warmup (None): {len(scores) - len(valid)}")
    print(f"score range: [{min(valid):.6e}, {max(valid):.6e}]")
    print(f"first 5 scores: {[f'{s:.6e}' for s in valid[:5]]}")