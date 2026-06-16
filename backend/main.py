"""
FastAPI backend for video anomaly detection (M4).

Serves the M3 U-Net future-frame predictor (via ONNX) over HTTP.
Minimal first slice: a single POST /predict endpoint that accepts a video
upload and returns per-frame anomaly scores as JSON.

Run:
    uvicorn backend.main:app --reload --port 8000

Then POST a video file to http://localhost:8000/predict (multipart form-data,
field name "file").
"""

import os
import tempfile
from contextlib import asynccontextmanager
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI yok, sadece render — server'da şart
import matplotlib.cm as cm

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Gauge

from src.inference.stream import process_frames  # reused; see note on video below
from src.inference.stream import process_video


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_PATH = "checkpoints/model.onnx"

# Anomaly threshold on the raw per-frame MSE score.
#
# DECISION POINT (yours): this should be calibrated from the NORMAL training
# distribution, not guessed. The principled way:
#   1. Run the stream over the normal training clips.
#   2. Collect all per-frame scores (these are all "normal" by construction).
#   3. Set THRESHOLD = mean + k * std  (k ~ 2-3), i.e. "how far above normal
#      counts as anomalous".
# The value below is a placeholder based on the M3 histogram (normal frames
# clustered ~0.0002-0.0003, anomalies tailing to ~0.0011). Replace it with a
# calibrated value and document the choice in milestones.md.
THRESHOLD = 0.000291


def make_overlay_png(frame_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> str:
    """
    Overlay the anomaly heatmap on the grayscale frame, return base64 PNG.

    frame_img: (128,128) preprocessed frame in [-1,1]
    heatmap:   (128,128) prediction error (>=0), arbitrary scale
    Returns: base64-encoded PNG string (no data: prefix)
    """
    from PIL import Image

    # 1. Frame [-1,1] -> [0,1] grayscale, then to RGB
    frame01 = (frame_img + 1.0) / 2.0
    frame01 = np.clip(frame01, 0, 1)
    base_rgb = np.stack([frame01] * 3, axis=-1)   # (128,128,3)

    # 2. Heatmap -> [0,1] normalized, apply colormap (inferno: dark->bright)
    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-12)                  # [0,1]
    heat_rgb = cm.inferno(hm)[..., :3]            # (128,128,3), drop alpha

    # 3. Blend: base frame + heatmap overlay
    blended = (1 - alpha) * base_rgb + alpha * heat_rgb
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

    # 4. To PNG -> base64
    img = Image.fromarray(blended)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# App lifespan: load the model once at startup, not per request
# ---------------------------------------------------------------------------

# We store shared state on app.state so every request reuses it.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify the model file exists. (The ONNX session itself is created
    # inside the stream pipeline per call; see the note in /predict below.)
    if not os.path.exists(ONNX_PATH):
        raise RuntimeError(f"ONNX model not found at {ONNX_PATH}. Run the export first.")
    app.state.onnx_path = ONNX_PATH
    app.state.threshold = THRESHOLD
    yield
    # Shutdown: nothing to clean up.


app = FastAPI(title="Video Anomaly Detection", lifespan=lifespan)

# Cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

# Anomaly ratio per processed video (flagged frames / scored frames)
ANOMALY_RATIO = Gauge(
    "augur_anomaly_ratio",
    "Fraction of scored frames flagged as anomalous in the last request",
)

# Mean anomaly score per processed video — watch for drift vs the calibrated
# normal mean (~0.000153). A rising mean suggests the input distribution shifted.
MEAN_SCORE = Gauge(
    "augur_mean_score",
    "Mean per-frame anomaly score (raw MSE) of the last processed video",
)

# Distribution of per-frame scores — a histogram to see the spread, not just mean
SCORE_HIST = Histogram(
    "augur_frame_score",
    "Per-frame anomaly score (raw MSE) distribution",
    buckets=[1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3],
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok", "model": app.state.onnx_path, "threshold": app.state.threshold}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a video upload, run the streaming anomaly detector over it, and
    return per-frame anomaly scores.

    Response JSON:
        {
          "total_frames": int,
          "scored_frames": int,
          "warmup_frames": int,        # first 15 frames have no score (cold start)
          "threshold": float,
          "frames": [
             {"frame_idx": int, "score": float|null, "is_anomaly": bool|null}
          ],
          "top_anomalies": [{"frame_idx": int, "score": float, "overlay": "<base64 png>"}]
        }
    A null score / null is_anomaly marks a warm-up frame (model needs 15 past
    frames before it can predict; those frames cannot be scored).
    """
    # Basic content-type guard (not bulletproof, just a friendly check).
    if file.content_type is None or not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="Please upload a video file.")

    # cv2.VideoCapture needs a file path, so we write the upload to a temp file.
    # Preserve the original suffix so the decoder picks the right backend.
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run the streaming pipeline (this is the code you wrote in stream.py).
        # NOTE: process_video currently builds its own AnomalyStream (and thus
        # its own ONNX session) per call. That is fine for a demo. To optimize
        # later, refactor AnomalyStream to accept a pre-loaded session and share
        # it across requests.
        raw_scores, top_anomalies, fps = process_video(tmp_path, app.state.onnx_path)
        fps = fps if fps and fps > 0 else 10.0   # fallback to 10 (UCSD default)

        # Convert top anomalies to an overlay PNG
        top = [
            {
                "frame_idx": a["frame_idx"],
                "score": a["score"],
                "overlay": make_overlay_png(a["frame"], a["heatmap"]),
            }
            for a in top_anomalies
        ]

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        # Always clean up the temp file.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Build the per-frame response. raw_scores contains None for warm-up frames.
    threshold = app.state.threshold
    frames = []
    scored = 0
    for idx, score in enumerate(raw_scores):
        if score is None:
            frames.append({"frame_idx": idx, "score": None, "is_anomaly": None})
        else:
            scored += 1
            frames.append({
                "frame_idx": idx,
                "score": float(score),
                "is_anomaly": bool(score > threshold),
            })

    # Filter out the valid scores
    valid_scores = [f["score"] for f in frames if f["score"] is not None]
    if valid_scores:
        flagged = sum(1 for f in frames if f["is_anomaly"])
        ANOMALY_RATIO.set(flagged / len(valid_scores))
        MEAN_SCORE.set(sum(valid_scores) / len(valid_scores))
        for s in valid_scores:
            SCORE_HIST.observe(s)

    return {
        "total_frames": len(raw_scores),
        "scored_frames": scored,
        "warmup_frames": len(raw_scores) - scored,
        "threshold": threshold,
        "fps": fps,
        "frames": frames,
        "top_anomalies": top
    }