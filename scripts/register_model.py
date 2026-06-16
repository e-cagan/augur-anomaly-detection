"""
Register the M3 model in MLflow with its metrics, params, and ONNX artifact.
Local tracking (mlruns/ dir) — view with: mlflow ui
"""

import onnx as ox
import mlflow


if __name__ == "__main__":
    # Local file-based tracking (creates ./mlruns)
    mlflow.set_experiment("augur-anomaly-detection")

    with mlflow.start_run(run_name="m3-future-frame-predictor"):
        # Parameters: what defines this model
        mlflow.log_params({
            "model_type": "unet-future-frame-prediction",
            "paradigm": "prediction",
            "input_frames": 15,
            "predict_frames": 1,
            "resolution": "128x128",
            "loss": "intensity+gradient",
            "threshold": 0.000291,
            "threshold_calibration": "normal_mean+2std",
            "dataset": "ucsd-ped2",
        })

        # Metrics: how it performs
        mlflow.log_metrics({
            "frame_auc": 0.840,
            "eer": 0.279,
            "normal_mean": 0.000153,
            "normal_std": 0.000069,
            "latency_ms": 16.9,
            "throughput_fps": 59.0,
        })

        # Register in the model registry
        # Load ONNX (onnx.load resolves external .data automatically)
        onnx_model = ox.load("checkpoints/model.onnx")

        # Log with the ONNX flavor AND register in one call
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            name="model",                                    # artifact path within the run
            registered_model_name="augur-anomaly-detector",  # registry name -> auto-versions
        )