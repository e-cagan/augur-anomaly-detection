"""
Measure per-frame inference latency for the streaming detector.
Real-time claim needs a number, not a guess.
"""

import time
import numpy as np
from src.inference.stream import AnomalyStream
import cv2


if __name__ == "__main__":
    onnx_path = "checkpoints/model.onnx"
    video_path = "/tmp/test001.mp4"

    stream = AnomalyStream(onnx_path)
    cap = cv2.VideoCapture(video_path)

    latencies = []   # per-frame inference time (ms), excluding warmup
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        result = stream.push(frame)
        t1 = time.perf_counter()
        # Only take scored frames
        if result is not None:
            latencies.append((t1 - t0) * 1000)   # ms
    cap.release()

    arr = np.array(latencies)
    print(f"scored frames: {len(arr)}")
    print(f"mean:   {arr.mean():.2f} ms")
    print(f"median: {np.median(arr):.2f} ms")
    print(f"p95:    {np.percentile(arr, 95):.2f} ms")
    print(f"max:    {arr.max():.2f} ms")
    print(f"throughput: {1000/arr.mean():.1f} fps")