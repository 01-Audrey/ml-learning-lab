"""
Model loading and initialization.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class ModelManager:
    """Manages YOLO and DeepSORT models."""

    def __init__(self, yolo_model: str = 'yolov8n.pt', 
                 max_age: int = 30, n_init: int = 3):
        """
        Initialize models.

        Args:
            yolo_model: YOLO model path
            max_age: DeepSORT max age
            n_init: DeepSORT n_init
        """
        self.yolo = self._load_yolo(yolo_model)
        self.tracker = self._init_tracker(max_age, n_init)

    def _load_yolo(self, model_path: str):
        """Load YOLO model."""
        print(f"Loading YOLO: {model_path}")
        return YOLO(model_path)

    def _init_tracker(self, max_age: int, n_init: int):
        """Initialize DeepSORT tracker."""
        print("Initializing DeepSORT tracker")
        return DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            embedder=None
        )

    def reset_tracker(self):
        """Reset tracker (for new video)."""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            embedder=None
        )
