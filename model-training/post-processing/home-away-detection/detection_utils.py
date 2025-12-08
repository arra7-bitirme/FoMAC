#!/usr/bin/env python3
"""
Player detection utilities using YOLO model.
Handles player detection with optional mock detections for testing.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    YOLO-based player detection with fallback to mock detections.
    Handles model loading, inference, and detection formatting.
    """
    
    def __init__(self, model_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize PlayerDetector
        
        Args:
            model_path: Path to YOLO model weights. If None, uses default path.
            verbose: Enable verbose logging
        """
        self.model_path = model_path or self._get_default_model_path()
        self.verbose = verbose
        self.model = self._load_model()
        
        # Track detection history for potential frame skip optimization
        self.last_detections = []
        self.frame_skip_count = 0
    
    @staticmethod
    def _get_default_model_path() -> str:
        """Get default YOLO model path"""
        return "/home/alperen/fomac/FoMAC/model-training/ball-detection/models/player_ball_detector/weights/best.pt"
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            from ultralytics import YOLO
            import logging as ul_logging
            
            # Suppress ultralytics verbose output
            ul_logging.getLogger('ultralytics').setLevel(ul_logging.ERROR)
            os.environ['YOLO_VERBOSE'] = 'False'
            
            if os.path.exists(self.model_path):
                if self.verbose:
                    logger.info(f"Loading YOLO model from: {self.model_path}")
                model = YOLO(self.model_path, verbose=False)
                model.overrides['verbose'] = False
                return model
            else:
                if self.verbose:
                    logger.warning(f"YOLO model not found at {self.model_path}, using mock detections")
                return None
                
        except ImportError:
            if self.verbose:
                logger.warning("ultralytics not available, using mock detections")
            return None
        except Exception as e:
            if self.verbose:
                logger.warning(f"YOLO model error: {e}, using mock detections")
            return None
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect players in frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'center'
        """
        if self.model is None:
            return self._create_mock_detections(frame)
        
        try:
            results = self.model.predict(
                frame, 
                conf=conf_threshold, 
                classes=[0],  # Player class
                verbose=False, 
                stream=False,
                imgsz=640,
                half=False,
                device='cpu'
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detection = {
                            'bbox': tuple(bbox),
                            'confidence': confidence,
                            'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        }
                        detections.append(detection)
            
            self.last_detections = detections
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}, using mock detections")
            return self._create_mock_detections(frame)
    
    def _create_mock_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create mock detections for testing without YOLO model
        
        Args:
            frame: Input frame
            
        Returns:
            List of mock detection dictionaries
        """
        height, width = frame.shape[:2]
        
        # Define mock player positions (varied sizes and positions)
        positions = [
            (width//4, height//3, width//4 + 80, height//3 + 160),
            (3*width//4, height//2, 3*width//4 + 70, height//2 + 140),
            (width//2, 2*height//3, width//2 + 75, 2*height//3 + 150),
            (width//6, height//2, width//6 + 60, height//2 + 120),
            (5*width//6, height//3, 5*width//6 + 65, height//3 + 130),
        ]
        
        detections = []
        
        for i, (x1, y1, x2, y2) in enumerate(positions):
            # Ensure bounding boxes are within frame
            x1 = max(0, min(x1, width - 80))
            y1 = max(0, min(y1, height - 160))
            x2 = max(x1 + 60, min(x2, width))
            y2 = max(y1 + 120, min(y2, height))
            
            detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': 0.6 + i * 0.08,  # Varying confidence
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            }
            detections.append(detection)
        
        return detections
    
    def detect_with_tracking(self, frame: np.ndarray, conf_threshold: float = 0.3, 
                           skip_frames: int = 0) -> List[Dict[str, Any]]:
        """
        Detect players with optional frame skipping for performance optimization
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            skip_frames: Number of frames to skip between detections (0 = detect every frame)
            
        Returns:
            List of detections (either new or cached)
        """
        if skip_frames > 0 and self.frame_skip_count < skip_frames:
            self.frame_skip_count += 1
            return self.last_detections
        
        self.frame_skip_count = 0
        return self.detect(frame, conf_threshold)
    
    def get_detection_count(self) -> int:
        """Get number of detections from last frame"""
        return len(self.last_detections)
    
    def is_model_loaded(self) -> bool:
        """Check if YOLO model is successfully loaded"""
        return self.model is not None
