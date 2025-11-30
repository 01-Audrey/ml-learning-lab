#!/usr/bin/env python3
"""
Download ML models for AI Security System
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("/app/models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

def download_model(name, url):
    """Download model if not exists."""
    filepath = MODELS_DIR / name

    if filepath.exists():
        print(f"✅ Model exists: {name}")
        return

    print(f"📥 Downloading: {name}")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded: {name} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")

if __name__ == "__main__":
    print("🔽 Checking ML models...")

    for name, url in MODELS.items():
        download_model(name, url)

    print("✅ All models ready!")
