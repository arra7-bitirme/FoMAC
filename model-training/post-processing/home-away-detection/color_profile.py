#!/usr/bin/env python3
"""
Color profile extraction for player jersey analysis.
Divides player bounding boxes into blocks and extracts color profiles
while filtering out green field pixels.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from color_utils import ColorAnalyzer

logger = logging.getLogger(__name__)


class ColorProfileExtractor:
    """
    Extracts player color profiles using block-based analysis.
    Divides player bounding boxes into fixed-size blocks and filters green field pixels.
    """
    
    def __init__(self, green_rgb_threshold: float = 80, 
                 green_cosine_similarity_min: float = 0.85):
        """
        Initialize ColorProfileExtractor
        
        Args:
            green_rgb_threshold: RGB distance threshold for green field detection
            green_cosine_similarity_min: Cosine similarity minimum for green detection
        """
        self.green_rgb_threshold = green_rgb_threshold
        self.green_cosine_similarity_min = green_cosine_similarity_min
        self.color_analyzer = ColorAnalyzer()
    
    def create_player_bbox_blocks(self, detections: List[Dict], 
                                  frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create fixed-size blocks within player bounding boxes.
        Uses upper 60% of bbox to focus on jersey (exclude legs).
        
        Args:
            detections: List of player detections
            frame: Input frame
            
        Returns:
            List of player block data dictionaries
        """
        player_blocks = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Use only upper 60% of bounding box (focus on jersey)
            bbox_height = y2 - y1
            upper_60_height = int(bbox_height * 0.6)
            y2 = y1 + upper_60_height
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Determine block pixel size based on bbox size
            # Use 3x3 pixel blocks for larger bboxes, 2x2 for smaller
            if bbox_width >= 60 or bbox_height >= 90:
                block_width = 3
                block_height = 3
            else:
                block_width = 2
                block_height = 2
            
            # Calculate number of blocks that fit
            block_cols = bbox_width // block_width
            block_rows = bbox_height // block_height
            
            blocks = {}
            block_weights = {}
            center_col, center_row = block_cols // 2, block_rows // 2
            
            # Create blocks with distance-based weights
            for row in range(block_rows):
                for col in range(block_cols):
                    block_x1 = x1 + col * block_width
                    block_y1 = y1 + row * block_height
                    block_x2 = min(block_x1 + block_width, x2)
                    block_y2 = min(block_y1 + block_height, y2)
                    
                    # Weight by distance from center (center blocks more important)
                    dx = abs(col - center_col) * block_width
                    dy = abs(row - center_row) * block_height
                    distance = math.sqrt(dx*dx + dy*dy)
                    weight = 10.0 / (1 + distance / 30) if distance > 0 else 10.0
                    
                    blocks[(col, row)] = {
                        'coords': (block_x1, block_y1, block_x2, block_y2),
                        'center': ((block_x1 + block_x2) // 2, (block_y1 + block_y2) // 2),
                        'size': (block_x2 - block_x1, block_y2 - block_y1)
                    }
                    block_weights[(col, row)] = weight
            
            player_blocks.append({
                'player_id': i,
                'detection': detection,
                'grid_size': (block_cols, block_rows),
                'center_block': (center_col, center_row),
                'blocks': blocks,
                'weights': block_weights,
                'block_pixel_size': (block_width, block_height)
            })
        
        return player_blocks
    
    def extract_player_color_profiles(self, frame: np.ndarray,
                                     player_blocks: List[Dict[str, Any]],
                                     green_reference: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract color profiles from player blocks with green field filtering.
        Includes fallback mechanism if all blocks are filtered out.
        
        Args:
            frame: Input frame (BGR)
            player_blocks: Player block data from create_player_bbox_blocks
            green_reference: Green reference color dictionary
            
        Returns:
            List of player color profile dictionaries
        """
        player_profiles = []
        green_bgr = np.array(green_reference['bgr'])
        
        for player_data in player_blocks:
            player_id = player_data['player_id']
            blocks = player_data['blocks']
            weights = player_data['weights']
            
            weighted_color_sum = np.zeros(3, dtype=np.float64)
            total_weight = 0.0
            included_blocks = []
            skipped_blocks = []
            block_colors = {}
            fallback_used = False
            
            # Process each block
            for (col, row), block_info in blocks.items():
                block_x1, block_y1, block_x2, block_y2 = block_info['coords']
                weight = weights[(col, row)]
                
                block_region = frame[block_y1:block_y2, block_x1:block_x2]
                
                if block_region.size > 0:
                    block_pixels = block_region.reshape(-1, 3)
                    
                    if len(block_pixels) > 0:
                        # Filter out black pixels
                        non_black_mask = ~np.all(block_pixels == [0, 0, 0], axis=1)
                        filtered_pixels = block_pixels[non_black_mask]
                        
                        if len(filtered_pixels) > 3:
                            block_avg_color = np.mean(filtered_pixels, axis=0)
                            
                            # Green filtering: RGB distance + cosine similarity
                            rgb_distance_from_green = self.color_analyzer.calculate_rgb_distance(
                                tuple(block_avg_color.astype(int)), 
                                tuple(green_bgr.astype(int))
                            )
                            
                            cosine_similarity = self.color_analyzer.calculate_cosine_similarity(
                                tuple(block_avg_color.astype(int)), 
                                tuple(green_bgr.astype(int))
                            )
                            
                            # Block is green if BOTH conditions are met
                            if rgb_distance_from_green < self.green_rgb_threshold and \
                               cosine_similarity >= self.green_cosine_similarity_min:
                                skipped_blocks.append((col, row, 'green_skip', 
                                                     rgb_distance_from_green, cosine_similarity))
                                continue
                            
                            # Filter out dark blocks
                            if np.sum(block_avg_color) < 30 or np.max(block_avg_color) < 40:
                                skipped_blocks.append((col, row, 'too_dark'))
                                continue
                            
                            # Valid block - store its color
                            block_colors[(col, row)] = tuple(block_avg_color.astype(int))
                            
                            weighted_color_sum += block_avg_color * weight
                            total_weight += weight
                            included_blocks.append((col, row, 'included'))
                        else:
                            skipped_blocks.append((col, row, 'insufficient_pixels'))
                    else:
                        skipped_blocks.append((col, row, 'no_pixels'))
                else:
                    skipped_blocks.append((col, row, 'empty_region'))
            
            # Fallback mechanism if no valid blocks found
            if total_weight == 0 or len(included_blocks) == 0:
                fallback_used = True
                fallback_profile = self._extract_fallback_profile(
                    frame, player_data, green_bgr
                )
                
                weighted_color_sum = fallback_profile['weighted_color_sum']
                total_weight = fallback_profile['total_weight']
                included_blocks = fallback_profile['included_blocks']
                skipped_blocks.extend(fallback_profile['skipped_blocks'])
                block_colors = fallback_profile['block_colors']
            
            # Calculate final weighted average color
            if total_weight > 0:
                weighted_avg_color = (weighted_color_sum / total_weight).astype(int)
                weighted_avg_color = tuple(weighted_avg_color)
            else:
                weighted_avg_color = (128, 128, 128)  # Gray fallback
            
            # Calculate tightened bbox from included blocks
            tightened_bbox = self._calculate_tightened_bbox(
                included_blocks, blocks, player_data['detection']['bbox']
            )
            
            profile = {
                'player_id': player_data['player_id'],
                'detection': player_data['detection'],
                'dominant_colors': [weighted_avg_color],
                'weighted_avg_color': weighted_avg_color,
                'total_weight': total_weight,
                'blocks_data': player_data,
                'included_blocks': included_blocks,
                'skipped_blocks': skipped_blocks,
                'block_colors': block_colors,
                'tightened_bbox': tightened_bbox,
                'fallback_used': fallback_used
            }
            
            player_profiles.append(profile)
        
        return player_profiles
    
    def _extract_fallback_profile(self, frame: np.ndarray, 
                                  player_data: Dict[str, Any],
                                  green_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Fallback color extraction using top 60% of cropped bbox.
        Used when all initial blocks are filtered out.
        
        Args:
            frame: Input frame
            player_data: Player block data
            green_bgr: Green reference color (BGR)
            
        Returns:
            Dictionary with fallback color profile data
        """
        original_bbox = player_data['detection']['bbox']
        x1_orig, y1_orig, x2_orig, y2_orig = original_bbox
        
        # Already cropped to upper 60%, now take top 60% of that
        bbox_height_orig = y2_orig - y1_orig
        upper_60_height = int(bbox_height_orig * 0.6)
        y2_crop = y1_orig + upper_60_height
        
        # Further take top 60% of this region
        fallback_height = y2_crop - y1_orig
        fallback_top_60 = int(fallback_height * 0.6)
        fallback_y2 = y1_orig + fallback_top_60
        
        fallback_width = x2_orig - x1_orig
        fallback_height_final = fallback_y2 - y1_orig
        
        # Use same block sizing logic
        if fallback_width >= 60 or fallback_height_final >= 90:
            fb_block_width = 3
            fb_block_height = 3
        else:
            fb_block_width = 2
            fb_block_height = 2
        
        fb_cols = fallback_width // fb_block_width
        fb_rows = fallback_height_final // fb_block_height
        
        # Reset accumulators
        weighted_color_sum = np.zeros(3, dtype=np.float64)
        total_weight = 0.0
        included_blocks = []
        skipped_blocks = []
        block_colors = {}
        
        center_col, center_row = fb_cols // 2, fb_rows // 2
        
        # Process fallback blocks
        for row in range(fb_rows):
            for col in range(fb_cols):
                block_x1 = x1_orig + col * fb_block_width
                block_y1 = y1_orig + row * fb_block_height
                block_x2 = min(block_x1 + fb_block_width, x2_orig)
                block_y2 = min(block_y1 + fb_block_height, fallback_y2)
                
                # Calculate weight
                dx = abs(col - center_col) * fb_block_width
                dy = abs(row - center_row) * fb_block_height
                distance = math.sqrt(dx*dx + dy*dy)
                weight = 10.0 / (1 + distance / 30) if distance > 0 else 10.0
                
                block_region = frame[block_y1:block_y2, block_x1:block_x2]
                
                if block_region.size > 0:
                    block_pixels = block_region.reshape(-1, 3)
                    
                    if len(block_pixels) > 0:
                        non_black_mask = ~np.all(block_pixels == [0, 0, 0], axis=1)
                        filtered_pixels = block_pixels[non_black_mask]
                        
                        if len(filtered_pixels) > 3:
                            block_avg_color = np.mean(filtered_pixels, axis=0)
                            
                            # Green filtering
                            rgb_distance_from_green = self.color_analyzer.calculate_rgb_distance(
                                tuple(block_avg_color.astype(int)), 
                                tuple(green_bgr.astype(int))
                            )
                            
                            cosine_similarity = self.color_analyzer.calculate_cosine_similarity(
                                tuple(block_avg_color.astype(int)), 
                                tuple(green_bgr.astype(int))
                            )
                            
                            if rgb_distance_from_green < self.green_rgb_threshold and \
                               cosine_similarity >= self.green_cosine_similarity_min:
                                skipped_blocks.append((col, row, 'green_skip_fallback', 
                                                     rgb_distance_from_green, cosine_similarity))
                                continue
                            
                            if np.sum(block_avg_color) < 30 or np.max(block_avg_color) < 40:
                                skipped_blocks.append((col, row, 'too_dark_fallback'))
                                continue
                            
                            block_colors[(col, row)] = tuple(block_avg_color.astype(int))
                            
                            weighted_color_sum += block_avg_color * weight
                            total_weight += weight
                            included_blocks.append((col, row, 'included_fallback'))
                        else:
                            skipped_blocks.append((col, row, 'insufficient_pixels_fallback'))
                    else:
                        skipped_blocks.append((col, row, 'no_pixels_fallback'))
                else:
                    skipped_blocks.append((col, row, 'empty_region_fallback'))
        
        return {
            'weighted_color_sum': weighted_color_sum,
            'total_weight': total_weight,
            'included_blocks': included_blocks,
            'skipped_blocks': skipped_blocks,
            'block_colors': block_colors
        }
    
    @staticmethod
    def _calculate_tightened_bbox(included_blocks: List[Tuple],
                                 blocks: Dict[Tuple[int, int], Dict],
                                 original_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate tightened bounding box based on included (non-green) blocks.
        
        Args:
            included_blocks: List of included block coordinates
            blocks: Block coordinate dictionary
            original_bbox: Original player bounding box
            
        Returns:
            Tightened bounding box tuple (x1, y1, x2, y2)
        """
        if not included_blocks:
            return original_bbox
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for col, row, status in included_blocks:
            if 'included' in status:
                block_info = blocks.get((col, row))
                if block_info:
                    bx1, by1, bx2, by2 = block_info['coords']
                    min_x = min(min_x, bx1)
                    min_y = min(min_y, by1)
                    max_x = max(max_x, bx2)
                    max_y = max(max_y, by2)
        
        if min_x != float('inf'):
            return (int(min_x), int(min_y), int(max_x), int(max_y))
        
        return original_bbox
