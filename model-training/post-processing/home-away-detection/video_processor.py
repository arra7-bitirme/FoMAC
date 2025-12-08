#!/usr/bin/env python3
"""
Video processing coordinator for team classification.
Coordinates all components to process video frames and classify players into teams.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from config import TeamClassifierConfig
from detection_utils import PlayerDetector
from field_analysis import FieldAnalyzer
from color_profile import ColorProfileExtractor
from team_classifier import TeamClassifier
from visualization_utils import Visualizer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Coordinates video processing pipeline for team classification.
    Manages all components and coordinates data flow between them.
    """
    
    def __init__(self, config: TeamClassifierConfig = None, **kwargs):
        """
        Initialize VideoProcessor
        
        Args:
            config: TeamClassifierConfig object (recommended)
            **kwargs: Individual parameters for backward compatibility
        """
        if config is not None:
            self.config = config
        else:
            # Backward compatibility: create config from kwargs
            self.config = TeamClassifierConfig(**{
                k: v for k, v in kwargs.items() 
                if k in TeamClassifierConfig.__dataclass_fields__
            })
        
        # Initialize all components
        self.detector = PlayerDetector(
            model_path=self.config.model_path,
            verbose=self.config.verbose
        )
        
        self.field_analyzer = FieldAnalyzer()
        
        self.color_extractor = ColorProfileExtractor(
            green_rgb_threshold=self.config.green_rgb_threshold,
            green_cosine_similarity_min=self.config.green_cosine_similarity_min
        )
        
        self.team_classifier = TeamClassifier(
            team_colors=self.config.team_colors,
            group_limits=self.config.group_limits,
            gk_max_distance=self.config.gk_max_distance,
            verbose=self.config.verbose
        )
        
        self.visualizer = Visualizer()
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'team_totals': {team: 0 for team in self.config.team_colors.keys()}
        }
        
        if self.config.verbose:
            logger.info("VideoProcessor initialized successfully")
            logger.info(f"Team colors: {self.config.team_colors}")
            logger.info(f"Model loaded: {self.detector.is_model_loaded()}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process single frame through complete pipeline.
        
        Pipeline steps:
        1. Detect players (YOLO)
        2. Analyze field region
        3. Extract green reference color
        4. Create player blocks
        5. Extract color profiles
        6. Classify teams
        7. Generate visualizations
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with all processing results, or None if failed
        """
        try:
            # Step 1: Detect players
            detections = self.detector.detect(frame)
            if not detections or len(detections) < 2:
                return None
            
            # Step 2: Analyze field region
            player_center = self.field_analyzer.calculate_player_center(detections)
            field_region = self.field_analyzer.determine_field_region(
                frame, player_center, detections
            )
            
            # Step 3: Extract green reference
            green_reference = self.field_analyzer.extract_weighted_green_reference(
                frame, field_region, player_center
            )
            
            # Step 4: Create player blocks
            player_blocks = self.color_extractor.create_player_bbox_blocks(
                detections, frame
            )
            
            # Step 5: Extract color profiles
            player_color_profiles = self.color_extractor.extract_player_color_profiles(
                frame, player_blocks, green_reference
            )
            
            # Step 6: Classify teams
            clustering_result = self.team_classifier.classify_players(
                player_color_profiles, use_normalization=True
            )
            
            # Step 7: Generate visualizations
            visualization = self.visualizer.visualize_clustering(
                frame, clustering_result, green_reference, field_region,
                self.team_classifier.team_colors
            )
            
            # Optional: Block filtering visualization (debug mode only)
            block_filtering_vis = None
            if self.config.debug_mode:
                block_filtering_vis = self.visualizer.visualize_block_filtering(
                    frame, player_color_profiles, green_reference
                )
            
            # Calculate team statistics
            team_counts = {}
            for result in clustering_result:
                team = result['team']
                team_counts[team] = team_counts.get(team, 0) + 1
            
            return {
                'detections': detections,
                'player_center': player_center,
                'field_region': field_region,
                'green_reference': green_reference,
                'clustering_result': clustering_result,
                'player_color_profiles': player_color_profiles,
                'team_counts': team_counts,
                'visualization': visualization,
                'block_filtering_vis': block_filtering_vis,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    
    def update_config(self, config: TeamClassifierConfig):
        """
        Update processor configuration dynamically.
        
        Args:
            config: New configuration
        """
        self.config = config
        
        # Recreate components with new config
        self.color_extractor = ColorProfileExtractor(
            green_rgb_threshold=config.green_rgb_threshold,
            green_cosine_similarity_min=config.green_cosine_similarity_min
        )
        
        self.team_classifier = TeamClassifier(
            team_colors=config.team_colors,
            group_limits=config.group_limits,
            gk_max_distance=config.gk_max_distance,
            verbose=config.verbose
        )
        
        logger.info("Configuration updated")
    
    def update_team_colors(self, team_colors: Dict[str, Tuple[int, int, int]]):
        """
        Update team colors dynamically.
        
        Args:
            team_colors: New team color dictionary
        """
        self.config.team_colors = team_colors
        self.team_classifier.update_team_colors(team_colors)
        logger.info(f"Team colors updated: {team_colors}")
    
    def update_stats(self, result: Optional[Dict[str, Any]]):
        """
        Update processing statistics.
        
        Args:
            result: Frame processing result
        """
        self.stats['total_frames'] += 1
        
        if result is not None and result.get('success', False):
            self.stats['successful_frames'] += 1
            
            team_counts = result.get('team_counts', {})
            for team, count in team_counts.items():
                if team in self.stats['team_totals']:
                    self.stats['team_totals'][team] += count
        else:
            self.stats['failed_frames'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'team_totals': {team: 0 for team in self.config.team_colors.keys()}
        }
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """
        Get detailed team statistics.
        
        Returns:
            Dictionary with team statistics
        """
        if self.stats['successful_frames'] == 0:
            return {}
        
        avg_per_frame = {
            team: total / self.stats['successful_frames']
            for team, total in self.stats['team_totals'].items()
        }
        
        return {
            'total_detections': sum(self.stats['team_totals'].values()),
            'team_totals': self.stats['team_totals'],
            'avg_per_frame': avg_per_frame,
            'success_rate': self.stats['successful_frames'] / self.stats['total_frames'] * 100
                           if self.stats['total_frames'] > 0 else 0
        }


class VideoWriter:
    """Helper class for video writing with logging support"""
    
    def __init__(self, output_path: str, fps: int, width: int, height: int,
                 enable_logging: bool = False):
        """
        Initialize VideoWriter
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            enable_logging: Enable block count logging
        """
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.enable_logging = enable_logging
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize log file if enabled
        self.log_file = None
        if enable_logging:
            log_path = str(Path(output_path).with_suffix('.log'))
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self._write_log_header()
            logger.info(f"Block count log: {log_path}")
    
    def _write_log_header(self):
        """Write log file header"""
        if self.log_file:
            self.log_file.write("=" * 120 + "\n")
            self.log_file.write("TEAM CLASSIFICATION BLOCK-COUNT LOG\n")
            self.log_file.write(f"Output: {self.output_path}\n")
            self.log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 120 + "\n\n")
            self.log_file.write("Frame | Player | Team      | Confidence | Total Blocks | " 
                              "Home | Away | Home_GK | Away_GK | Method\n")
            self.log_file.write("-" * 120 + "\n")
            self.log_file.flush()
    
    def write_frame(self, frame: np.ndarray, result: Optional[Dict[str, Any]] = None,
                   frame_number: int = 0):
        """
        Write frame to video and optionally log results.
        
        Args:
            frame: Frame to write
            result: Processing result (for logging)
            frame_number: Current frame number
        """
        self.writer.write(frame)
        
        if self.log_file and result and result.get('success'):
            self._log_frame_results(frame_number, result)
    
    def _log_frame_results(self, frame_number: int, result: Dict[str, Any]):
        """Log frame classification results"""
        clustering_result = result.get('clustering_result', [])
        
        for idx, detection_result in enumerate(clustering_result):
            team = detection_result.get('team', 'unknown')
            confidence = detection_result.get('confidence', 0.0)
            total_blocks = detection_result.get('total_blocks', 0)
            block_counts = detection_result.get('block_counts', {})
            reason = detection_result.get('reason', 'N/A')
            
            method = reason.split(' - ')[0] if ' - ' in reason else reason
            
            home_count = block_counts.get('home', 0)
            away_count = block_counts.get('away', 0)
            home_gk_count = block_counts.get('home_gk', 0)
            away_gk_count = block_counts.get('away_gk', 0)
            
            log_line = f"{frame_number:5d} | {idx:6d} | {team:9s} | " \
                      f"{confidence:10.3f} | {total_blocks:12d} | " \
                      f"{home_count:4d} | {away_count:4d} | " \
                      f"{home_gk_count:7d} | {away_gk_count:7d} | {method}\n"
            self.log_file.write(log_line)
        
        self.log_file.flush()
    
    def release(self):
        """Release video writer and close log file"""
        self.writer.release()
        
        if self.log_file:
            self.log_file.write("\n" + "=" * 120 + "\n")
            self.log_file.write("END OF LOG\n")
            self.log_file.write("=" * 120 + "\n")
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
