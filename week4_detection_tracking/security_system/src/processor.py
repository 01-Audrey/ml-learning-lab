"""
Video processing pipeline with logging.
"""

import cv2
import time
from tqdm import tqdm
from .video_io import VideoInput, VideoOutput
from .utils import yolo_to_deepsort, generate_dummy_embeddings
from .logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Main video processing pipeline."""

    def __init__(self, model_manager, show_display: bool = False):
        """Initialize processor."""
        self.model = model_manager.yolo
        self.tracker = model_manager.tracker
        self.show_display = show_display

        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'unique_tracks': set()
        }

        logger.info("VideoProcessor initialized")

    def process_video(self, input_path: str, output_path: str = None):
        """Process video file."""
        logger.info(f"Starting video processing: {input_path}")
        start_time = time.time()

        try:
            # Open input
            video_in = VideoInput(input_path)
            logger.info(f"Video opened: {video_in.width}x{video_in.height} @ {video_in.fps:.1f}fps")

            # Setup output
            video_out = None
            if output_path:
                video_out = VideoOutput(
                    output_path,
                    video_in.width,
                    video_in.height,
                    video_in.fps
                )
                logger.info(f"Output video: {output_path}")

            # Process frames
            pbar = tqdm(total=video_in.total_frames, desc="Processing")

            while True:
                ret, frame = video_in.read()
                if not ret:
                    break

                # Process frame
                processed = self._process_frame(frame)

                # Save
                if video_out:
                    video_out.write(processed)

                # Display
                if self.show_display:
                    cv2.imshow('Processing', processed)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.warning("Processing interrupted by user")
                        break

                pbar.update(1)

            pbar.close()
            video_in.release()
            if video_out:
                video_out.release()
            if self.show_display:
                cv2.destroyAllWindows()

            # Log results
            elapsed = time.time() - start_time
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

            logger.info("="*60)
            logger.info("Processing Complete")
            logger.info(f"Frames processed: {self.stats['frames_processed']}")
            logger.info(f"Total detections: {self.stats['total_detections']}")
            logger.info(f"Unique tracks: {len(self.stats['unique_tracks'])}")
            logger.info(f"Processing time: {elapsed:.1f}s")
            logger.info(f"Average FPS: {fps:.1f}")
            logger.info("="*60)

            return self.stats

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise

    def _process_frame(self, frame):
        """Process single frame."""
        # Detection
        results = self.model.predict(frame, conf=0.5, verbose=False)
        detections = results[0].boxes

        # Convert to DeepSORT
        deepsort_input = yolo_to_deepsort(detections)

        # Track
        if len(deepsort_input) > 0:
            embeddings = generate_dummy_embeddings(len(deepsort_input))
            tracks = self.tracker.update_tracks(
                deepsort_input, embeds=embeddings, frame=frame
            )
        else:
            tracks = []

        # Update stats
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        for track in tracks:
            if track.is_confirmed():
                self.stats['unique_tracks'].add(track.track_id)

        # Visualize
        annotated = self._draw_results(frame, tracks)

        return annotated

    def _draw_results(self, frame, tracks):
        """Draw tracking results."""
        annotated = frame.copy()

        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'ID: {track.track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated
