"""
Logging system for security system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_file: str = "logs/system.log", 
                  level: str = "INFO",
                  console: bool = True):
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file
        level: Logging level
        console: Enable console output
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_level = getattr(logging, level.upper())

    # Format
    log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Handlers
    handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Security System Started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {level}")
    logger.info("="*60)


def get_logger(name: str):
    """
    Get logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
