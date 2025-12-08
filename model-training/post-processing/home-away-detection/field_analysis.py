#!/usr/bin/env python3
"""
Field analysis utilities for soccer video processing.
Handles field region detection and green reference color extraction.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class FieldAnalyzer:
    """
    Analyzes soccer field regions and extracts green reference colors.
    Uses player positions to identify field area and compute weighted green reference.
    """
    
    def __init__(self):
        """Initialize FieldAnalyzer"""
        pass
    
    @staticmethod
    def calculate_player_center(detections: List[Dict]) -> Tuple[int, int]:
        """
        Calculate the centroid of all player positions
        
        Args:
            detections: List of player detections with 'center' key
            
        Returns:
            Tuple of (x, y) coordinates of player centroid
        """
        if not detections:
            return (0, 0)
        
        centers = [det['center'] for det in detections]
        avg_x = int(np.mean([c[0] for c in centers]))
        avg_y = int(np.mean([c[1] for c in centers]))
        
        return (avg_x, avg_y)
    
    @staticmethod
    def determine_field_region(frame: np.ndarray, 
                              player_center: Tuple[int, int],
                              detections: List[Dict],
                              margin: int = 100) -> Dict[str, Any]:
        """
        Determine field region based on player positions
        
        Args:
            frame: Input frame
            player_center: Centroid of player positions
            detections: List of player detections
            margin: Margin around players to include in field region
            
        Returns:
            Dictionary with field region mask, bounds, size, and center
        """
        height, width = frame.shape[:2]
        center_x, center_y = player_center
        
        # Extract all player coordinates
        player_xs = [det['center'][0] for det in detections]
        player_ys = [det['center'][1] for det in detections]
        
        # Calculate bounding box around all players with margin
        min_x = max(0, min(player_xs) - margin)
        max_x = min(width, max(player_xs) + margin)
        min_y = max(0, min(player_ys) - 50)
        max_y = min(height, max(player_ys) + margin)
        
        # Create binary mask for field region
        field_mask = np.zeros((height, width), dtype=np.uint8)
        field_mask[min_y:max_y, min_x:max_x] = 255
        
        region_size = np.sum(field_mask > 0)
        
        return {
            'mask': field_mask,
            'bounds': (min_x, min_y, max_x, max_y),
            'size': region_size,
            'center': player_center
        }
    
    @staticmethod
    def extract_weighted_green_reference(frame: np.ndarray,
                                        field_region: Dict[str, Any],
                                        player_center: Tuple[int, int]) -> Dict[str, Any]:
        """
        Extract weighted green reference color from field region.
        Uses distance-based weighting to prioritize pixels near player center.
        
        Args:
            frame: Input frame (BGR)
            field_region: Field region dictionary from determine_field_region
            player_center: Center point for weighting calculation
            
        Returns:
            Dictionary with green reference color, coverage statistics
        """
        field_mask = field_region['mask']
        center_x, center_y = player_center
        height, width = frame.shape[:2]
        
        # Create distance-based weight map (vectorized)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Normalize distances and create weights
        max_distance = np.sqrt(width**2 + height**2)
        weights = 1.0 - (distance_from_center / max_distance)
        weights = np.clip(weights, 0.1, 1.0)
        
        # Combine field mask with distance weights
        field_weights = weights * (field_mask / 255.0)
        
        # Extract field pixels and their weights
        field_pixels = frame[field_mask > 0]
        field_pixel_weights = field_weights[field_mask > 0]
        
        if len(field_pixels) == 0:
            # Fallback: default green
            return {
                'rgb': (50, 150, 50),
                'bgr': (50, 150, 50),
                'coverage': 0.0,
                'weighted_pixels': 0
            }
        
        # Identify green pixels (G > B and G > R)
        green_mask = (field_pixels[:, 1] > field_pixels[:, 0]) & \
                    (field_pixels[:, 1] > field_pixels[:, 2])
        green_pixels = field_pixels[green_mask]
        green_weights = field_pixel_weights[green_mask]
        
        if len(green_pixels) > 100:
            # Compute weighted average of green pixels
            weighted_sum = np.sum(green_pixels * green_weights.reshape(-1, 1), axis=0)
            total_weight = np.sum(green_weights)
            green_reference_bgr = (weighted_sum / total_weight).astype(int)
            
            coverage = len(green_pixels) / len(field_pixels) * 100
            weighted_pixels = float(total_weight)
        else:
            # Fallback: use all field pixels with enhanced green channel
            weighted_sum = np.sum(field_pixels * field_pixel_weights.reshape(-1, 1), axis=0)
            total_weight = np.sum(field_pixel_weights)
            avg_color = (weighted_sum / total_weight).astype(int)
            
            # Enhance green channel
            green_reference_bgr = np.array([
                int(avg_color[0] * 0.7),
                int(min(255, avg_color[1] * 1.2)),
                int(avg_color[2] * 0.8)
            ])
            
            coverage = 50.0
            weighted_pixels = float(total_weight)
        
        # Convert BGR to RGB
        green_reference_rgb = green_reference_bgr[::-1]
        
        return {
            'rgb': tuple(int(x) for x in green_reference_rgb),
            'bgr': tuple(int(x) for x in green_reference_bgr),
            'coverage': coverage,
            'weighted_pixels': weighted_pixels,
            'field_region': field_region
        }
    
    @staticmethod
    def extract_simple_green_reference(frame: np.ndarray, 
                                      field_region: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Extract simple green reference color (unweighted average).
        Faster alternative to weighted method.
        
        Args:
            frame: Input frame (BGR)
            field_region: Field region dictionary
            
        Returns:
            BGR tuple of green reference color
        """
        field_mask = field_region['mask']
        field_pixels = frame[field_mask > 0]
        
        if len(field_pixels) == 0:
            return (50, 150, 50)
        
        # Filter for green-dominant pixels
        green_mask = (field_pixels[:, 1] > field_pixels[:, 0]) & \
                    (field_pixels[:, 1] > field_pixels[:, 2])
        green_pixels = field_pixels[green_mask]
        
        if len(green_pixels) > 100:
            # Average of green pixels
            avg_green = np.mean(green_pixels, axis=0).astype(int)
            return tuple(avg_green)
        else:
            # Average of all field pixels with green boost
            avg_color = np.mean(field_pixels, axis=0).astype(int)
            green_boosted = np.array([
                int(avg_color[0] * 0.7),
                int(min(255, avg_color[1] * 1.2)),
                int(avg_color[2] * 0.8)
            ])
            return tuple(green_boosted)
