#!/usr/bin/env python3
"""
Team classification module for soccer player analysis.
Classifies players into teams using block-count majority vote method.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from color_utils import ColorAnalyzer

logger = logging.getLogger(__name__)


class TeamClassifier:
    """
    Classifies players into teams using block-count majority vote.
    Each player's blocks are analyzed and assigned to closest team color,
    then majority vote determines final team assignment.
    """
    
    def __init__(self, 
                 team_colors: Dict[str, Tuple[int, int, int]] = None,
                 group_limits: Dict[str, int] = None,
                 gk_max_distance: float = 80,
                 angular_threshold_deg: float = 30.0,
                 verbose: bool = True):
        """
        Initialize TeamClassifier
        
        Args:
            team_colors: BGR colors for each team
            group_limits: Maximum players per team
            gk_max_distance: Max distance for goalkeeper classification
            angular_threshold_deg: Angular threshold for color matching
            verbose: Enable verbose logging
        """
        self.team_colors = team_colors or {
            'home': (201, 201, 199),
            'away': (111, 85, 82),
            'home_gk': (80, 60, 200),
            'away_gk': (185, 185, 130)
        }
        
        self.group_limits = group_limits or {
            'home': 11,
            'away': 11,
            'home_gk': 1,
            'away_gk': 1
        }
        
        self.gk_max_distance = gk_max_distance
        self.angular_threshold_deg = angular_threshold_deg
        self.verbose = verbose
        self.color_analyzer = ColorAnalyzer()
    
    def classify_players(self, player_profiles: List[Dict[str, Any]],
                        use_normalization: bool = True) -> List[Dict[str, Any]]:
        """
        Classify players into teams using block-count majority vote method.
        
        Process:
        1. For each player, iterate through non-green blocks
        2. Assign each block to closest team color
        3. Count blocks per team
        4. Assign player to team with most blocks (majority vote)
        
        Args:
            player_profiles: List of player color profiles
            use_normalization: Use color normalization in distance calculation
            
        Returns:
            List of classification results with team, confidence, block counts
        """
        if len(player_profiles) < 2:
            return [{'profile': profile, 'team': 'home', 'confidence': 1.0} 
                   for profile in player_profiles]
        
        player_data = []
        
        for i, profile in enumerate(player_profiles):
            included_blocks = profile.get('included_blocks', [])
            blocks_data = profile.get('blocks_data', {})
            
            if not included_blocks or not blocks_data:
                # Fallback: use weighted average color
                fallback_result = self._classify_by_average_color(
                    profile, i, use_normalization
                )
                player_data.append(fallback_result)
                continue
            
            # Block-count majority vote method
            block_colors = profile.get('block_colors', {})
            team_block_counts = {team: 0 for team in self.team_colors.keys()}
            total_valid_blocks = 0
            
            # Process each non-green block
            for col, row, status in included_blocks:
                if status != 'included' and 'included_fallback' not in status:
                    continue
                
                block_color = block_colors.get((col, row))
                if not block_color:
                    continue
                
                # Find closest team color for this block
                min_distance = float('inf')
                closest_team = None
                
                for team, team_color in self.team_colors.items():
                    distance = self.color_analyzer.calculate_hsv_distance(
                        block_color, team_color
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_team = team
                
                if closest_team:
                    team_block_counts[closest_team] += 1
                    total_valid_blocks += 1
            
            # Determine team by majority vote
            if total_valid_blocks == 0:
                assigned_team = 'home'
                confidence = 0.3
                distance = 150.0
            else:
                sorted_teams = sorted(team_block_counts.items(), 
                                    key=lambda x: x[1], reverse=True)
                assigned_team = sorted_teams[0][0]
                max_count = sorted_teams[0][1]
                
                # Confidence based on majority percentage
                confidence = max(0.3, min(1.0, max_count / total_valid_blocks))
                
                # Calculate distance to assigned team color
                distance = self.color_analyzer.calculate_weighted_bgr_distance(
                    profile['weighted_avg_color'], 
                    self.team_colors[assigned_team],
                    use_normalization, 
                    use_angular=True
                )
            
            player_data.append({
                'id': i,
                'profile': profile,
                'team': assigned_team,
                'distance': distance,
                'confidence': confidence,
                'block_counts': team_block_counts,
                'total_blocks': total_valid_blocks,
                'method': 'block_majority_vote'
            })
        
        # Apply team member limits and create final results
        results = self._apply_team_limits(player_data)
        
        return results
    
    def _classify_by_average_color(self, profile: Dict[str, Any], 
                                   player_id: int,
                                   use_normalization: bool) -> Dict[str, Any]:
        """
        Fallback classification using weighted average color.
        Used when block data is unavailable.
        
        Args:
            profile: Player color profile
            player_id: Player ID
            use_normalization: Use normalized distance
            
        Returns:
            Classification result dictionary
        """
        color = profile['weighted_avg_color']
        distances = {
            team: self.color_analyzer.calculate_weighted_bgr_distance(
                color, team_color, use_normalization, use_angular=True
            )
            for team, team_color in self.team_colors.items()
        }
        
        sorted_teams = sorted(distances.items(), key=lambda x: x[1])
        best_team = sorted_teams[0][0]
        best_distance = sorted_teams[0][1]
        
        return {
            'id': player_id,
            'profile': profile,
            'team': best_team,
            'distance': best_distance,
            'confidence': max(0.3, 1.0 - best_distance / 300),
            'block_counts': {},
            'total_blocks': 0,
            'method': 'fallback_average'
        }
    
    def _apply_team_limits(self, player_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply team member limits and handle goalkeeper constraints.
        
        Args:
            player_data: List of player classification data
            
        Returns:
            Final classification results with team limits enforced
        """
        team_members = {team: [] for team in self.team_colors.keys()}
        results = []
        
        # Sort by confidence (higher confidence first)
        player_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        for player in player_data:
            assigned_team = player['team']
            
            # Check goalkeeper constraints
            if '_gk' in assigned_team:
                if player['distance'] > self.gk_max_distance:
                    # Try non-GK version of same side
                    base_team = assigned_team.replace('_gk', '')
                    if len(team_members[base_team]) < self.group_limits[base_team]:
                        assigned_team = base_team
                    else:
                        assigned_team = 'home'  # Fallback
            
            # Check team limit
            if len(team_members[assigned_team]) >= self.group_limits[assigned_team]:
                # Try alternative teams
                for alt_team in self.team_colors.keys():
                    if len(team_members[alt_team]) < self.group_limits[alt_team]:
                        assigned_team = alt_team
                        break
            
            team_members[assigned_team].append(player)
            
            results.append({
                'profile': player['profile'],
                'team': assigned_team,
                'confidence': player['confidence'],
                'color_center': player['profile']['weighted_avg_color'],
                'distance': player['distance'],
                'reason': f"{player['method']} - blocks:{player.get('total_blocks', 0)}",
                'block_counts': player.get('block_counts', {}),
                'total_blocks': player.get('total_blocks', 0)
            })
        
        return results
    
    def update_team_colors(self, team_colors: Dict[str, Tuple[int, int, int]]):
        """
        Update team colors dynamically.
        
        Args:
            team_colors: New team color dictionary
        """
        self.team_colors = team_colors
        logger.info(f"Updated team colors: {team_colors}")
    
    def get_team_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about team classification results.
        
        Args:
            results: Classification results
            
        Returns:
            Statistics dictionary
        """
        team_counts = {}
        team_confidences = {}
        
        for result in results:
            team = result['team']
            confidence = result['confidence']
            
            team_counts[team] = team_counts.get(team, 0) + 1
            if team not in team_confidences:
                team_confidences[team] = []
            team_confidences[team].append(confidence)
        
        stats = {
            'total_players': len(results),
            'team_counts': team_counts,
            'team_avg_confidence': {
                team: np.mean(confs) if confs else 0.0
                for team, confs in team_confidences.items()
            }
        }
        
        return stats
