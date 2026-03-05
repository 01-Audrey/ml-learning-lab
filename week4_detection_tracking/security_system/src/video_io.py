"""
Video input and output handling.
"""

import cv2
import os
from typing import Tuple, Optional


class VideoInput:
    """Video input handler."""

    def __init__(self, source, retry_count: int = 3):
        """
        Initialize video input.

        Args:
            source: Video source (file, camera, stream)
            retry_count: Number of retry attempts
        """
        self.source = source
        self.retry_count = retry_count
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0

        self._open()

    def _open(self):
        """Open video source."""
        for attempt in range(self.retry_count):
            self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                return

        raise RuntimeError(f"Failed to open: {self.source}")

    def read(self) -> Tuple[bool, Optional[object]]:
        """Read next frame."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


class VideoOutput:
    """Video output handler."""

    def __init__(self, output_path: str, width: int, height: int, 
                 fps: float, codec: str = 'mp4v'):
        """
        Initialize video output.

        Args:
            output_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_count = 0

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self._open()

    def _open(self):
        """Open video writer."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {self.output_path}")

    def write(self, frame):
        """Write frame to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
