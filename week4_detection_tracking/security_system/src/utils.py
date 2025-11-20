"""
Utility functions.
"""

import numpy as np


def yolo_to_deepsort(detections):
    """
    Convert YOLO detections to DeepSORT format.

    Args:
        detections: YOLO detection boxes

    Returns:
        list: DeepSORT format detections
    """
    deepsort_input = []

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        w = x2 - x1
        h = y2 - y1
        deepsort_input.append(([x1, y1, w, h], conf, 'person'))

    return deepsort_input


def generate_dummy_embeddings(count: int, dim: int = 128):
    """
    Generate dummy embeddings for DeepSORT.

    Args:
        count: Number of embeddings
        dim: Embedding dimension

    Returns:
        list: Dummy embeddings
    """
    return [np.random.rand(dim).astype(np.float32) for _ in range(count)]
