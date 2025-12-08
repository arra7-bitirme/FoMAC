#!/usr/bin/env python3
"""
Visualization utilities for team classification results.
Provides visualization of player detections, team assignments, and block filtering.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualization utilities for team classification system.
    Handles drawing of bounding boxes, labels, statistics, and debug visualizations.
    """
    
    def __init__(self):
        """Initialize Visualizer"""
        self.bbox_colors = {
            'home': (0, 255, 0),       # Green
            'away': (0, 0, 255),        # Red
            'home_gk': (0, 255, 255),   # Yellow
            'away_gk': (255, 0, 255)    # Magenta
        }
        
        self.team_labels = {
            'home': 'HOME (Light Gray)',
            'away': 'AWAY (Dark)',
            'home_gk': 'HOME GK (Red)',
            'away_gk': 'AWAY GK (Olive)'
        }
    
    def visualize_block_filtering(self, frame: np.ndarray,
                                  player_profiles: List[Dict[str, Any]],
                                  green_ref: Dict[str, Any]) -> np.ndarray:
        """
        Visualize block filtering process (debug mode).
        Shows which blocks are excluded (green) vs included (jersey).
        
        Args:
            frame: Input frame
            player_profiles: Player color profiles with block data
            green_ref: Green reference color dictionary
            
        Returns:
            Visualization frame
        """
        vis = frame.copy()
        
        for profile in player_profiles:
            player_id = profile['player_id']
            detection = profile['detection']
            included_blocks = profile.get('included_blocks', [])
            skipped_blocks = profile.get('skipped_blocks', [])
            blocks_data = profile.get('blocks_data', {})
            blocks = blocks_data.get('blocks', {})
            
            # Draw original bounding box (white)
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Draw tightened bbox (cyan)
            tightened_bbox = profile.get('tightened_bbox')
            if tightened_bbox:
                tx1, ty1, tx2, ty2 = tightened_bbox
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
            
            fallback_used = profile.get('fallback_used', False)
            
            green_count = 0
            included_count = 0
            
            # Draw excluded blocks (red)
            for skip_info in skipped_blocks:
                if len(skip_info) >= 3:
                    col, row, reason = skip_info[:3]
                    block_info = blocks.get((col, row))
                    if block_info:
                        bx1, by1, bx2, by2 = block_info['coords']
                        
                        if 'green_skip' in reason:
                            green_count += 1
                            # Semi-transparent red for green blocks
                            overlay = vis.copy()
                            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                        elif 'too_dark' in reason:
                            # Dark gray outline
                            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (50, 50, 50), 1)
            
            # Draw included blocks (green or cyan if fallback)
            for incl_info in included_blocks:
                if len(incl_info) >= 3:
                    col, row, status = incl_info[:3]
                    if 'included' in status:
                        included_count += 1
                        block_info = blocks.get((col, row))
                        if block_info:
                            bx1, by1, bx2, by2 = block_info['coords']
                            block_color = (255, 255, 0) if 'fallback' in status else (0, 255, 0)
                            cv2.rectangle(vis, (bx1, by1), (bx2, by2), block_color, 1)
            
            # Add block count text
            text_y = y1 - 30
            player_label = f"Player {player_id}" + (" [FALLBACK]" if fallback_used else "")
            cv2.putText(vis, player_label, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            text_y += 20
            cv2.putText(vis, f"Green: {green_count} | Included: {included_count}", 
                       (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add legend
        vis = self._draw_block_filtering_legend(vis, green_ref)
        
        return vis
    
    def visualize_clustering(self, frame: np.ndarray,
                           clustering_result: List[Dict[str, Any]],
                           green_ref: Dict[str, Any],
                           field_region: Dict[str, Any],
                           team_colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Visualize team classification results.
        
        Args:
            frame: Input frame
            clustering_result: Team classification results
            green_ref: Green reference color
            field_region: Field region data
            team_colors: Team colors dictionary
            
        Returns:
            Visualization frame
        """
        vis = frame.copy()
        
        if team_colors is None:
            team_colors = {
                'home': (201, 201, 199),
                'away': (111, 85, 82),
                'home_gk': (80, 80, 180),
                'away_gk': (185, 185, 130)
            }
        
        # Draw field region overlay
        vis = self._draw_field_overlay(vis, field_region)
        
        # Draw player bounding boxes and labels
        for result in clustering_result:
            vis = self._draw_player_result(vis, result)
        
        # Draw statistics panel
        vis = self._draw_statistics_panel(vis, clustering_result, green_ref)
        
        # Draw reference colors panel
        vis = self._draw_reference_colors_panel(vis, team_colors)
        
        return vis
    
    def _draw_player_result(self, vis: np.ndarray, 
                           result: Dict[str, Any]) -> np.ndarray:
        """Draw single player result on visualization"""
        profile = result['profile']
        team = result['team']
        confidence = result.get('confidence', 1.0)
        detection = profile['detection']
        distance = result.get('distance', 0)
        
        # Use tightened bbox if available
        tightened_bbox = profile.get('tightened_bbox')
        if tightened_bbox:
            x1, y1, x2, y2 = tightened_bbox
        else:
            x1, y1, x2, y2 = detection['bbox']
        
        color = self.bbox_colors.get(team, (255, 255, 255))
        
        # Adjust color opacity based on confidence
        if confidence < 0.7:
            alpha = confidence
            color = tuple(int(c * alpha + 128 * (1 - alpha)) for c in color)
        
        thickness = 3 if confidence >= 0.7 else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        weighted_color = profile.get('weighted_avg_color', profile['dominant_colors'][0])
        b, g, r = int(weighted_color[0]), int(weighted_color[1]), int(weighted_color[2])
        
        # Draw team label
        label = team.upper()
        if confidence < 0.7:
            label += f" ({confidence:.2f})"
        cv2.putText(vis, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw color info
        vis = self._draw_color_info(vis, x1, y2, b, g, r, distance)
        
        # Draw color swatch
        swatch_x, swatch_y = x1, y2 + 5
        bgr_color = (b, g, r)
        cv2.rectangle(vis, (swatch_x, swatch_y), 
                     (swatch_x+30, swatch_y+20), bgr_color, -1)
        cv2.rectangle(vis, (swatch_x, swatch_y), 
                     (swatch_x+30, swatch_y+20), (255, 255, 255), 2)
        
        return vis
    
    def _draw_color_info(self, vis: np.ndarray, x: int, y: int,
                        b: int, g: int, r: int, distance: float) -> np.ndarray:
        """Draw color information text"""
        color_text = f"BGR({b},{g},{r})"
        text_y = y + 30
        
        text_size = cv2.getTextSize(color_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(vis, (x, text_y - text_size[1] - 2),
                     (x + text_size[0] + 4, text_y + 2), (255, 255, 255), -1)
        cv2.putText(vis, color_text, (x + 2, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        dist_text = f"d:{distance:.1f}"
        text_y += 15
        
        text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(vis, (x, text_y - text_size[1] - 2),
                     (x + text_size[0] + 4, text_y + 2), (255, 255, 255), -1)
        cv2.putText(vis, dist_text, (x + 2, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return vis
    
    def _draw_field_overlay(self, vis: np.ndarray, 
                           field_region: Dict[str, Any]) -> np.ndarray:
        """Draw field region overlay"""
        field_mask = field_region['mask']
        field_overlay = vis.copy()
        field_overlay[field_mask > 0] = [0, 255, 255]  # Yellow
        cv2.addWeighted(field_overlay, 0.15, vis, 0.85, 0, vis)
        
        # Draw field bounds
        min_x, min_y, max_x, max_y = field_region['bounds']
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
        cv2.putText(vis, "Field Region", (min_x, min_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return vis
    
    def _draw_statistics_panel(self, vis: np.ndarray,
                              clustering_result: List[Dict[str, Any]],
                              green_ref: Dict[str, Any]) -> np.ndarray:
        """Draw statistics panel on visualization"""
        team_counts = {}
        team_confidences = {}
        
        for result in clustering_result:
            team = result['team']
            confidence = result.get('confidence', 1.0)
            team_counts[team] = team_counts.get(team, 0) + 1
            if team not in team_confidences:
                team_confidences[team] = []
            team_confidences[team].append(confidence)
        
        y_pos = 30
        cv2.putText(vis, "Team Classification", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        # Draw team counts
        for team in ['home', 'away', 'home_gk', 'away_gk']:
            if team in team_counts and team_counts[team] > 0:
                count = team_counts[team]
                color = self.bbox_colors.get(team, (255, 255, 255))
                avg_confidence = np.mean(team_confidences[team])
                
                label = f"{self.team_labels.get(team, team.upper())}: {count}"
                if avg_confidence < 0.9:
                    label += f" (conf:{avg_confidence:.2f})"
                
                cv2.putText(vis, label, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 25
        
        # Draw green reference
        y_pos += 10
        cv2.rectangle(vis, (10, y_pos), (40, y_pos+20), green_ref['bgr'], -1)
        cv2.rectangle(vis, (10, y_pos), (40, y_pos+20), (255, 255, 255), 2)
        
        green_b, green_g, green_r = green_ref['bgr']
        green_text = f"Field Green: RGB({green_r},{green_g},{green_b})"
        cv2.putText(vis, green_text, (45, y_pos+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Total players
        total_detections = len(clustering_result)
        cv2.putText(vis, f"Total: {total_detections} players", (10, y_pos+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def _draw_reference_colors_panel(self, vis: np.ndarray,
                                    team_colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
        """Draw reference colors panel"""
        height, width = vis.shape[:2]
        panel_x = width - 250
        panel_y = 30
        
        # Semi-transparent background
        overlay = vis.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y - 10),
                     (width - 10, panel_y + 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        
        cv2.putText(vis, "Reference Colors (BGR)", (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        panel_y += 25
        
        reference_colors = {
            'HOME': team_colors.get('home', (201, 201, 199)),
            'AWAY': team_colors.get('away', (111, 85, 82)),
            'HOME_GK': team_colors.get('home_gk', (80, 60, 200)),
            'AWAY_GK': team_colors.get('away_gk', (185, 185, 130))
        }
        
        for group_name, bgr_color in reference_colors.items():
            cv2.rectangle(vis, (panel_x, panel_y), 
                         (panel_x + 30, panel_y + 20), bgr_color, -1)
            cv2.rectangle(vis, (panel_x, panel_y), 
                         (panel_x + 30, panel_y + 20), (255, 255, 255), 1)
            
            b, g, r = bgr_color
            color_info = f"{group_name}: ({b},{g},{r})"
            cv2.putText(vis, color_info, (panel_x + 35, panel_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            panel_y += 30
        
        return vis
    
    def _draw_block_filtering_legend(self, vis: np.ndarray,
                                    green_ref: Dict[str, Any]) -> np.ndarray:
        """Draw legend for block filtering visualization"""
        legend_y = 30
        cv2.putText(vis, "Block Filtering Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        
        # Red: Excluded (Green)
        cv2.rectangle(vis, (10, legend_y), (30, legend_y+15), (0, 0, 255), -1)
        cv2.putText(vis, "Excluded (Green)", (35, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        
        # Green: Included (Jersey)
        cv2.rectangle(vis, (10, legend_y), (30, legend_y+15), (0, 255, 0), 2)
        cv2.putText(vis, "Included (Jersey)", (35, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        
        # Cyan: Fallback Region
        cv2.rectangle(vis, (10, legend_y), (30, legend_y+15), (255, 255, 0), 2)
        cv2.putText(vis, "Fallback Region", (35, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        
        # Cyan: Tightened BBox
        cv2.rectangle(vis, (10, legend_y), (30, legend_y+15), (255, 255, 0), 2)
        cv2.putText(vis, "Tightened BBox", (35, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        
        # Green reference color
        green_bgr = green_ref['bgr']
        cv2.rectangle(vis, (10, legend_y), (40, legend_y+20), green_bgr, -1)
        cv2.rectangle(vis, (10, legend_y), (40, legend_y+20), (255, 255, 255), 1)
        b, g, r = green_bgr
        cv2.putText(vis, f"Field Green RGB: ({r},{g},{b})", (45, legend_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
