#!/usr/bin/env python3
"""
Example usage of the modular YOLO training pipeline.

This script demonstrates different ways to use the new modular system.
"""

import sys
from pathlib import Path

# Add the yolo directory to Python path
yolo_dir = Path(__file__).parent
sys.path.insert(0, str(yolo_dir))

from main import YOLOTrainingPipeline


def example_basic_usage():
    """Example: Basic usage with default configurations."""
    print("=== Basic Usage Example ===")
    pipeline = YOLOTrainingPipeline()
    
    # This will use all default configurations from config files
    # pipeline.run()


def example_cli_override():
    """Example: Override configuration via CLI arguments."""
    print("=== CLI Override Example ===")
    pipeline = YOLOTrainingPipeline()
    
    # Simulate command line arguments
    args = [
        '--epochs', '50',           # Override training epochs
        '--batch', '8',             # Override batch size  
        '--max-samples-per-half', '500',  # Limit dataset size for testing
        '--run-extraction',         # Force dataset extraction
        '--log-level', 'DEBUG'      # More verbose logging
    ]
    
    # pipeline.run(args)


def example_extract_only():
    """Example: Extract dataset only."""
    print("=== Extract Only Example ===")
    pipeline = YOLOTrainingPipeline()
    
    args = [
        '--extract-only',           # Only extract dataset
        '--max-samples-per-half', '100'  # Small dataset for testing
    ]
    
    # pipeline.run(args)


def example_train_only():
    """Example: Train only (skip extraction)."""
    print("=== Train Only Example ===")
    pipeline = YOLOTrainingPipeline()
    
    args = [
        '--train-only',             # Skip extraction
        '--epochs', '20',           # Quick training
        '--batch', '4'              # Small batch for testing
    ]
    
    # pipeline.run(args)


def example_evaluation():
    """Example: Evaluate trained model."""
    print("=== Evaluation Example ===")
    pipeline = YOLOTrainingPipeline()
    
    args = ['--evaluate']
    
    # pipeline.run(args)


def example_inference():
    """Example: Run inference on new data."""
    print("=== Inference Example ===")
    pipeline = YOLOTrainingPipeline()
    
    # Replace with actual path to your images/video
    source_path = "/path/to/your/images"
    
    args = ['--predict', source_path]
    
    # pipeline.run(args)


def example_custom_config():
    """Example: Using custom configuration directory."""
    print("=== Custom Config Example ===")
    pipeline = YOLOTrainingPipeline()
    
    args = [
        '--config-dir', '/path/to/custom/configs',
        '--run-extraction'
    ]
    
    # pipeline.run(args)


if __name__ == "__main__":
    print("YOLO Training Pipeline Examples")
    print("=" * 50)
    print()
    
    # Show examples (commented out to avoid actual execution)
    example_basic_usage()
    print()
    
    example_cli_override()
    print()
    
    example_extract_only()
    print()
    
    example_train_only()
    print()
    
    example_evaluation()
    print()
    
    example_inference()
    print()
    
    example_custom_config()
    print()
    
    print("To actually run any example, uncomment the pipeline.run() calls.")
    print()
    print("For real usage, run:")
    print("  python main.py")
    print("  python main.py --help")