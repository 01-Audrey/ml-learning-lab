"""
Configuration management for security system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """Initialize configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return self._get_default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'yolo_model': 'yolov8n.pt',
                'confidence': 0.5,
                'device': 'cpu'
            },
            'tracking': {
                'max_age': 30,
                'n_init': 3,
                'nms_max_overlap': 1.0
            },
            'video': {
                'target_resolution': [640, 480],
                'skip_frames': 0,
                'save_output': True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
