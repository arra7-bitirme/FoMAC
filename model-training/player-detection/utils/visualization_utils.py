"""
Visualization Utilities

Contains the visualization functions moved from the training directory
for use in the modular YOLO training pipeline.
"""

import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_soccernet_json(json_path: str) -> Tuple[List[Dict], Optional[List]]:
    """
    Load SoccerNet MaskRCNN JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (predictions, size_info)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    preds = data.get("predictions", [])
    size = data.get("size", None)  # [N,H,W,3] format
    
    return preds, size


def make_scaler(
    src_wh: Tuple[int, int],
    dst_wh: Tuple[int, int],
    mode: str = "scale"
) -> Callable[[float, float, float, float], Tuple[int, int, int, int]]:
    """
    Create a bbox scaling function.
    
    Args:
        src_wh: Source (width, height)
        dst_wh: Destination (width, height)
        mode: 'scale' for independent w/h scaling, 'letterbox' for aspect-preserving
        
    Returns:
        Scaling function that takes (x1, y1, x2, y2) and returns scaled coordinates
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    
    if sw <= 0 or sh <= 0:  # Safety check
        return lambda x1, y1, x2, y2: (int(x1), int(y1), int(x2), int(y2))
    
    if mode == "letterbox":
        # Preserve aspect ratio
        r = min(dw / sw, dh / sh)
        new_w, new_h = int(round(sw * r)), int(round(sh * r))
        off_x = (dw - new_w) // 2
        off_y = (dh - new_h) // 2
        
        def _letterbox_map(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
            return (
                int(round(x1 * r + off_x)),
                int(round(y1 * r + off_y)),
                int(round(x2 * r + off_x)),
                int(round(y2 * r + off_y)),
            )
        return _letterbox_map
    
    else:  # scale mode
        # Independent w/h scaling
        sx, sy = (dw / sw), (dh / sh)
        
        def _scale_map(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
            return (
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
            )
        return _scale_map


def bgr_from_colors_entry(color_entry: Any) -> Tuple[int, int, int]:
    """
    Convert color entry to BGR tuple.
    
    Args:
        color_entry: Color specification (list/tuple of RGB values)
        
    Returns:
        BGR color tuple
    """
    if not isinstance(color_entry, (list, tuple)) or len(color_entry) < 3:
        return (0, 255, 0)  # Default green
    
    r, g, b = int(color_entry[0]), int(color_entry[1]), int(color_entry[2])
    return (b, g, r)  # Convert RGB to BGR for OpenCV


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box detections on frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries with 'bbox' and optional 'color'
        thickness: Line thickness for bounding boxes
        
    Returns:
        Frame with drawn detections
    """
    h, w = frame.shape[:2]
    
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
            
        x1, y1, x2, y2 = bbox
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        
        # Get color (default to green)
        color = det.get("color", (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame


def validate_bbox(
    x1: float, y1: float, x2: float, y2: float,
    frame_width: int, frame_height: int,
    min_size: int = 1
) -> Tuple[bool, Tuple[int, int, int, int]]:
    """
    Validate and clamp bounding box coordinates.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        frame_width, frame_height: Frame dimensions
        min_size: Minimum valid size in pixels
        
    Returns:
        Tuple of (is_valid, (clamped_x1, clamped_y1, clamped_x2, clamped_y2))
    """
    # Clamp to frame bounds
    x1_clamped = max(0, min(frame_width - 1, int(x1)))
    y1_clamped = max(0, min(frame_height - 1, int(y1)))
    x2_clamped = max(0, min(frame_width, int(x2)))
    y2_clamped = max(0, min(frame_height, int(y2)))
    
    # Check if valid
    width = x2_clamped - x1_clamped
    height = y2_clamped - y1_clamped
    
    is_valid = (
        width >= min_size and
        height >= min_size and
        x2_clamped > x1_clamped and
        y2_clamped > y1_clamped
    )
    
    return is_valid, (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def bbox_to_yolo_format(
    x1: int, y1: int, x2: int, y2: int,
    frame_width: int, frame_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bbox coordinates to YOLO format (normalized).
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        frame_width, frame_height: Frame dimensions
        
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    # Calculate center and dimensions
    x_center = ((x1 + x2) / 2) / frame_width
    y_center = ((y1 + y2) / 2) / frame_height
    width = (x2 - x1) / frame_width
    height = (y2 - y1) / frame_height
    
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    fourcc: str = "mp4v"
) -> cv2.VideoWriter:
    """
    Create a video writer with error handling.
    
    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: (width, height) of frames
        fourcc: Video codec fourcc
        
    Returns:
        OpenCV VideoWriter instance
    """
    try:
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        return writer
        
    except Exception as e:
        logger.error(f"Error creating video writer: {e}")
        raise


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Add text overlay to frame.
    
    Args:
        frame: Input frame
        text: Text to add
        position: (x, y) position for text
        font_scale: Font scale factor
        color: Text color in BGR
        thickness: Text thickness
        
    Returns:
        Frame with text overlay
    """
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )
    return frame


def compute_detection_times(
    num_predictions: int,
    det_fps: Optional[float] = None,
    video_duration: Optional[float] = None,
    det_start_sec: float = 0.0
) -> np.ndarray:
    """
    Compute detection times for frame alignment.
    
    Args:
        num_predictions: Number of predictions
        det_fps: Detection FPS (None for auto-calculation)
        video_duration: Video duration in seconds
        det_start_sec: Detection start time offset
        
    Returns:
        Array of detection times in seconds
    """
    if det_fps is None:
        if video_duration is None or video_duration <= 0:
            det_fps = 2.0  # Default fallback
        else:
            det_fps = num_predictions / video_duration
    
    times = np.array([
        i / det_fps + det_start_sec
        for i in range(num_predictions)
    ], dtype=float)
    
    return times


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    try:
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS) or 25.0,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        
        # Calculate duration
        if info['frame_count'] > 0 and info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        else:
            info['duration'] = 0.0
        
        return info
        
    finally:
        cap.release()


def safe_video_read(cap: cv2.VideoCapture, frame_idx: int) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Safely read a frame from video capture.
    
    Args:
        cap: OpenCV VideoCapture object
        frame_idx: Frame index to read
        
    Returns:
        Tuple of (success, frame)
    """
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return ret, frame if ret else None
    except Exception as e:
        logger.warning(f"Error reading frame {frame_idx}: {e}")
        return False, None