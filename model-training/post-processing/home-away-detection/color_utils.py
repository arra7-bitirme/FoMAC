#!/usr/bin/env python3
"""
Color analysis utilities for team classification.
Handles color space conversions, distance calculations, and dominant color extraction.
Optimized with vectorized NumPy operations where possible.
"""

import cv2
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ColorAnalyzer:
    """
    Optimized color analysis and distance calculation utility.
    Provides various color distance metrics and color space conversions.
    """
    
    def __init__(self):
        """Initialize ColorAnalyzer"""
        # Pre-compute common constants
        self._pi_over_180 = np.pi / 180.0
        self._max_cylindrical_distance = np.sqrt(2**2 + 1**2)
    
    @staticmethod
    def bgr_to_hsv(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert BGR color to HSV color space"""
        bgr_array = np.uint8([[list(color_bgr)]])
        hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_array[0][0]
        return (float(h), float(s), float(v))
    
    @staticmethod
    def bgr_to_hsv_batch(colors_bgr: np.ndarray) -> np.ndarray:
        """
        Convert batch of BGR colors to HSV (vectorized).
        
        Args:
            colors_bgr: Array of shape (N, 3) with BGR colors
            
        Returns:
            Array of shape (N, 3) with HSV colors
        """
        if len(colors_bgr.shape) == 1:
            colors_bgr = colors_bgr.reshape(1, -1)
        
        colors_bgr_uint8 = colors_bgr.astype(np.uint8).reshape(-1, 1, 3)
        hsv_array = cv2.cvtColor(colors_bgr_uint8, cv2.COLOR_BGR2HSV)
        return hsv_array.reshape(-1, 3).astype(np.float32)
    
    def normalize_color(self, color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Normalize color in HSV space to reduce lighting effects"""
        h, s, v = self.bgr_to_hsv(color_bgr)
        
        if v > 0:
            s_normalized = min(255, s * (200.0 / v))
        else:
            s_normalized = s
        
        return (h, s_normalized, 180.0)
    
    def calculate_normalized_distance(self, color1_bgr: Tuple[int, int, int], 
                                     color2_bgr: Tuple[int, int, int]) -> float:
        """Calculate distance between normalized HSV colors"""
        h1, s1, v1 = self.normalize_color(color1_bgr)
        h2, s2, v2 = self.normalize_color(color2_bgr)
        
        dh = min(abs(h1 - h2), 180 - abs(h1 - h2))
        ds = abs(s1 - s2)
        dv = abs(v1 - v2)
        
        distance = np.sqrt((dh * 2.0)**2 + (ds * 1.5)**2 + (dv * 0.1)**2)
        return distance
    
    def hsv_to_cylindrical(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to cylindrical coordinates for better color comparison"""
        h_rad = h * self._pi_over_180
        s_norm = s / 255.0
        
        x = s_norm * np.cos(h_rad)
        y = s_norm * np.sin(h_rad)
        z = v / 255.0
        
        return (x, y, z)
    
    def hsv_to_cylindrical_batch(self, hsv_array: np.ndarray) -> np.ndarray:
        """
        Convert batch of HSV colors to cylindrical coordinates (vectorized).
        
        Args:
            hsv_array: Array of shape (N, 3) with HSV colors
            
        Returns:
            Array of shape (N, 3) with cylindrical coordinates
        """
        h = hsv_array[:, 0] * self._pi_over_180
        s_norm = hsv_array[:, 1] / 255.0
        v_norm = hsv_array[:, 2] / 255.0
        
        x = s_norm * np.cos(h)
        y = s_norm * np.sin(h)
        z = v_norm
        
        return np.column_stack([x, y, z])
    
    def calculate_cosine_similarity(self, color1_bgr: Tuple[int, int, int], 
                                   color2_bgr: Tuple[int, int, int]) -> float:
        """Calculate cosine similarity between two colors in cylindrical HSV space"""
        h1, s1, v1 = self.bgr_to_hsv(color1_bgr)
        h2, s2, v2 = self.bgr_to_hsv(color2_bgr)
        
        x1, y1, z1 = self.hsv_to_cylindrical(h1, s1, v1)
        x2, y2, z2 = self.hsv_to_cylindrical(h2, s2, v2)
        
        vec1 = np.array([x1, y1, z1], dtype=np.float64)
        vec2 = np.array([x2, y2, z2], dtype=np.float64)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        cosine_sim = np.clip(cosine_sim, 0.0, 1.0)
        
        return float(cosine_sim)
    
    def calculate_angular_distance(self, color1_bgr: Tuple[int, int, int], 
                                  color2_bgr: Tuple[int, int, int]) -> float:
        """Calculate angular distance between colors in cylindrical space"""
        h1, s1, v1 = self.bgr_to_hsv(color1_bgr)
        h2, s2, v2 = self.bgr_to_hsv(color2_bgr)
        
        x1, y1, z1 = self.hsv_to_cylindrical(h1, s1, v1)
        x2, y2, z2 = self.hsv_to_cylindrical(h2, s2, v2)
        
        distance_3d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        normalized = min(distance_3d / self._max_cylindrical_distance, 1.0)
        angle_degrees = normalized * 180.0
        
        return angle_degrees
    
    def calculate_hsv_distance(self, color1_bgr: Tuple[int, int, int], 
                              color2_bgr: Tuple[int, int, int]) -> float:
        """Calculate weighted distance in HSV color space"""
        h1, s1, v1 = self.bgr_to_hsv(color1_bgr)
        h2, s2, v2 = self.bgr_to_hsv(color2_bgr)
        
        dh = min(abs(h1 - h2), 180 - abs(h1 - h2))
        ds = abs(s1 - s2)
        dv = abs(v1 - v2)
        
        distance = np.sqrt((dh * 2.0)**2 + (ds * 1.0)**2 + (dv * 0.5)**2)
        
        return distance
    
    @staticmethod
    def calculate_rgb_distance(color1_bgr: Tuple[int, int, int], 
                              color2_bgr: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance in RGB color space"""
        b1, g1, r1 = color1_bgr
        b2, g2, r2 = color2_bgr
        
        distance = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        return distance
    
    def calculate_weighted_bgr_distance(self, color1_bgr: Tuple[int, int, int], 
                                      color2_bgr: Tuple[int, int, int],
                                      use_normalization: bool = True,
                                      use_angular: bool = True) -> float:
        """Calculate weighted color distance combining multiple metrics"""
        hsv_distance = self.calculate_hsv_distance(color1_bgr, color2_bgr)
        distance = hsv_distance
        
        if use_normalization:
            normalized_distance = self.calculate_normalized_distance(color1_bgr, color2_bgr)
            distance = 0.7 * distance + 0.3 * normalized_distance
        
        if use_angular:
            angular_dist = self.calculate_angular_distance(color1_bgr, color2_bgr)
            normalized_angular = (angular_dist / 90.0) * 100.0
            distance = distance * 0.8 + normalized_angular * 0.2
        
        return distance
    
    def extract_dominant_colors(self, pixels: np.ndarray, n_colors: int = 3) -> List[Tuple[int, ...]]:
        """
        Extract dominant colors from pixel array.
        Optimized with vectorized operations and color grouping.
        
        Args:
            pixels: BGR pixel array of shape (N, 3)
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as BGR tuples
        """
        if len(pixels) == 0:
            return []
        
        try:
            # Constants
            BRIGHTNESS_THRESHOLD = 30
            MIN_GROUP_SIZE = 8
            COLOR_RANGE_THRESHOLD = 25
            
            # Filter out dark pixels (vectorized)
            pixel_brightness = np.sum(pixels, axis=1)
            bright_mask = pixel_brightness > BRIGHTNESS_THRESHOLD
            bright_pixels = pixels[bright_mask]
            
            if len(bright_pixels) == 0:
                return []
            
            # Convert to RGB for color grouping
            rgb_pixels = bright_pixels[:, ::-1]
            
            # Vectorized color grouping
            r, g, b = rgb_pixels[:, 0], rgb_pixels[:, 1], rgb_pixels[:, 2]
            color_range = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            
            # Group pixels by dominant channel
            blue_mask = (b > r) & (b > g) & (b > 70) & (color_range >= COLOR_RANGE_THRESHOLD)
            red_mask = (r > g) & (r > b) & (r > 70) & (color_range >= COLOR_RANGE_THRESHOLD)
            green_mask = (g > r) & (g > b) & (g > 70) & (color_range >= COLOR_RANGE_THRESHOLD)
            mixed_mask = ~(blue_mask | red_mask | green_mask)
            
            color_groups = {
                'blue': rgb_pixels[blue_mask],
                'red': rgb_pixels[red_mask],
                'green': rgb_pixels[green_mask],
                'mixed': rgb_pixels[mixed_mask]
            }
            
            dominant_colors = []
            
            # Process each color group
            for group_name, group_pixels in color_groups.items():
                if len(group_pixels) >= MIN_GROUP_SIZE:
                    # Use median for robustness against outliers
                    median_color_rgb = np.median(group_pixels, axis=0).astype(int)
                    
                    if np.sum(median_color_rgb) > BRIGHTNESS_THRESHOLD and np.max(median_color_rgb) > 50:
                        median_color_bgr = median_color_rgb[::-1]
                        dominant_colors.append((tuple(median_color_bgr), len(group_pixels), group_name))
            
            # Sort by count (most common first)
            dominant_colors.sort(key=lambda x: x[1], reverse=True)
            result_colors = [color for color, count, group in dominant_colors[:n_colors]]
            
            # Fallback: color quantization if no groups found
            if len(result_colors) == 0:
                # Reduce color space for clustering (vectorized)
                reduced_pixels_rgb = (rgb_pixels // 30) * 30
                unique_colors, counts = np.unique(reduced_pixels_rgb, axis=0, return_counts=True)
                
                # Filter valid colors (vectorized)
                valid_mask = (np.sum(unique_colors, axis=1) > BRIGHTNESS_THRESHOLD) & \
                           (np.max(unique_colors, axis=1) > 50)
                
                if np.any(valid_mask):
                    valid_colors = unique_colors[valid_mask]
                    valid_counts = counts[valid_mask]
                    
                    # Sort by count
                    sort_indices = np.argsort(valid_counts)[::-1]
                    top_colors_rgb = valid_colors[sort_indices[:n_colors]]
                    
                    # Convert to BGR tuples
                    result_colors = [tuple(color[::-1].astype(int)) for color in top_colors_rgb]
            
            return result_colors
            
        except Exception as e:
            logger.warning(f"Error in dominant color extraction: {e}")
            # Fallback: simple average
            if len(pixels) > 0:
                pixel_brightness = np.sum(pixels, axis=1)
                bright_mask = pixel_brightness > 30
                bright_pixels = pixels[bright_mask]
                
                if len(bright_pixels) > 0:
                    avg_color = np.mean(bright_pixels, axis=0).astype(int)
                    if np.sum(avg_color) > 30:
                        return [tuple(avg_color)]
            return []
