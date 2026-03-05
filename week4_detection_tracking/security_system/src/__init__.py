"""
Security System - Object Detection and Tracking

A professional security system using YOLOv8 and DeepSORT.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import Config
from .models import ModelManager
from .processor import VideoProcessor
from .video_io import VideoInput, VideoOutput

__all__ = [
    "Config",
    "ModelManager", 
    "VideoProcessor",
    "VideoInput",
    "VideoOutput"
]
