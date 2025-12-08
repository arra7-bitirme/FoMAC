#!/usr/bin/env python3
"""
Configuration module for team classification system.
Provides centralized configuration management for all components.
"""

from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TeamClassifierConfig:
    """Configuration class for team classification parameters"""
    
    # Team colors in BGR format
    team_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'home': (201, 201, 199),
        'away': (111, 85, 82),
        'home_gk': (80, 60, 200),
        'away_gk': (185, 185, 130)
    })
    
    # Maximum players per team
    group_limits: Dict[str, int] = field(default_factory=lambda: {
        'home': 11,
        'away': 11,
        'home_gk': 1,
        'away_gk': 1
    })
    
    # Green field detection thresholds
    green_rgb_threshold: float = 80
    green_cosine_similarity_min: float = 0.85
    
    # Goalkeeper classification threshold
    gk_max_distance: float = 80
    
    # Model configuration
    model_path: Optional[str] = None
    
    # Debug and logging
    debug_mode: bool = False
    verbose: bool = True
    
    # Video processing parameters
    sample_rate: int = 1
    max_frames: Optional[int] = None
    
    # Display and output
    display: bool = False
    save_video: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TeamClassifierConfig':
        """Create configuration from dictionary (useful for JSON/YAML configs)"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return asdict(self)
    
    def update(self, **kwargs):
        """Update configuration parameters dynamically"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")


# Default configurations for common scenarios
DEFAULT_CONFIG = TeamClassifierConfig()

FAST_CONFIG = TeamClassifierConfig(
    debug_mode=False,
    verbose=False,
    sample_rate=2,
    display=False
)

DEBUG_CONFIG = TeamClassifierConfig(
    debug_mode=True,
    verbose=True,
    sample_rate=1,
    display=True
)
