"""
Module to download UCSD dataset.
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("karthiknm1/ucsd-anomaly-detection-dataset")

print("Path to dataset files:", path)