"""
Utility functions for ReID module.
"""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: dict, updates: dict) -> dict:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Dictionary with updates
        
    Returns:
        Updated configuration
    """
    import copy
    
    config = copy.deepcopy(config)
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = recursive_update(base[key], value)
            else:
                base[key] = value
        return base
    
    return recursive_update(config, updates)
