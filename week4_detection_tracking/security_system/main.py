#!/usr/bin/env python3
"""
Security System - Command Line Interface

Main entry point for the security system application.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.models import ModelManager
from src.processor import VideoProcessor
from src.logging_config import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Security System - Object Detection and Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python main.py input.mp4 -o output.mp4

  # Use custom config
  python main.py input.mp4 --config custom_config.yaml

  # Show live display
  python main.py input.mp4 --display

  # Process from webcam
  python main.py 0 --display
        """
    )

    # Required arguments
    parser.add_argument(
        'input',
        help='Input video file or camera index (0 for webcam)'
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output video file path'
    )

    parser.add_argument(
        '-c', '--config',
        default='config/default.yaml',
        help='Configuration file path (default: config/default.yaml)'
    )

    parser.add_argument(
        '-d', '--display',
        action='store_true',
        help='Show live display window'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable log file (console only)'
    )

    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = None if args.no_log_file else "logs/system.log"
    setup_logging(log_file=log_file, level=args.log_level)
    logger = get_logger(__name__)

    try:
        logger.info("="*60)
        logger.info("SECURITY SYSTEM STARTING")
        logger.info("="*60)

        # Load configuration
        logger.info(f"Loading configuration: {args.config}")
        config = Config(args.config)

        # Initialize models
        logger.info("Initializing models...")
        model_manager = ModelManager(
            yolo_model=config.get('model.yolo_model', 'yolov8n.pt'),
            max_age=config.get('tracking.max_age', 30),
            n_init=config.get('tracking.n_init', 3)
        )

        # Create processor
        processor = VideoProcessor(
            model_manager=model_manager,
            show_display=args.display
        )

        # Determine input source
        try:
            input_source = int(args.input)  # Camera index
            logger.info(f"Using camera: {input_source}")
        except ValueError:
            input_source = args.input  # File path
            logger.info(f"Using video file: {input_source}")

        # Process video
        stats = processor.process_video(
            input_path=input_source,
            output_path=args.output
        )

        logger.info("="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
