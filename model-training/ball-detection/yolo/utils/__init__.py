"""
Utilities Initialization

Makes the utils directory a Python package.
"""

__version__ = "1.0.0"

# Import utility functions for convenience
from .config_utils import ConfigManager, setup_logging, validate_configurations
from .visualization_utils import (
    load_soccernet_json,
    make_scaler,
    draw_detections,
    validate_bbox,
    bbox_to_yolo_format,
    get_video_info
)

__all__ = [
    'ConfigManager',
    'setup_logging', 
    'validate_configurations',
    'load_soccernet_json',
    'make_scaler',
    'draw_detections',
    'validate_bbox',
    'bbox_to_yolo_format',
    'get_video_info'
]