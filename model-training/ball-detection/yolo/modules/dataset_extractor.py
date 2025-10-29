"""
Player and Ball Dataset Extraction Module

Handles extraction of player and ball detection dataset from multiple sources
and conversion to YOLO format for multi-class training.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import logging
import sys
import json
import shutil
from PIL import Image

try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face datasets not available. Install with: pip install datasets")

logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Extracts and converts player and ball detection datasets to YOLO format."""
    
    def __init__(self, config: Dict[str, Any], paths_config: Dict[str, Any]):
        """
        Initialize dataset extractor.
        
        Args:
            config: Extraction configuration
            paths_config: Path configuration
        """
        self.config = config
        self.paths_config = paths_config
        
    def extract_dataset(self) -> Path:
        """
        Extract and convert player & ball detection datasets to YOLO format.
        
        Returns:
            Path to the output dataset root
        """
        if not self.config.get('run_extraction', True):
            logger.info("Dataset extraction skipped (run_extraction=False)")
            return self._get_output_root()
            
        logger.info("=" * 80)
        logger.info("PLAYER & BALL DETECTION DATASET EXTRACTION")
        logger.info("=" * 80)
        
        # Check if we have prepared ballDataset
        ballDataset_path = self._get_ball_dataset_path()
        output_root = self._get_output_root()
        
        logger.info(f"Ball Dataset: {ballDataset_path}")
        logger.info(f"Output: {output_root}")
        
        # Check if ballDataset already exists in correct YOLO format
        if self._check_existing_prepared_dataset(ballDataset_path):
            logger.info("✅ Using existing prepared ballDataset in YOLO format")
            self._validate_dataset_yaml(ballDataset_path)
            return ballDataset_path
        
        # Fallback to original extraction logic
        logger.info("Prepared ballDataset not found, falling back to extraction...")
        player_dataset_path = self._get_player_dataset_path()
        
        # Check if dataset already exists in correct format
        if self._check_existing_yolo_format(output_root):
            logger.info("✅ YOLO format dataset already exists, skipping conversion")
            self._create_dataset_yaml(output_root)
            return output_root
        
        # Extract and combine datasets
        self._extract_multi_class_dataset(ballDataset_path, player_dataset_path, output_root)
        
        self._create_dataset_yaml(output_root)
        
        logger.info("✅ Dataset extraction completed")
        return output_root
    
    def _get_ball_dataset_path(self) -> Path:
        """Get ball dataset path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / "ballDataset"
    
    def _get_player_dataset_path(self) -> Path:
        """Get player dataset path (from player-detection module)."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        # Try to find existing player detection dataset
        player_paths = [
            workspace_root.parent / "player-detection" / "datasets",
            workspace_root.parent / "player-detection" / "models",
            workspace_root / "player_dataset",
            # Could also point to SoccerNet if available
            workspace_root / "soccerNet"
        ]
        
        for path in player_paths:
            if path.exists():
                return path
        
        # Default fallback
        return workspace_root / "player_dataset"
    
    def _get_output_root(self) -> Path:
        """Get output dataset root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['output_root']
    
    def _check_existing_yolo_format(self, output_root: Path) -> bool:
        """Check if YOLO format dataset already exists."""
        required_dirs = [
            output_root / "images" / "train",
            output_root / "labels" / "train", 
            output_root / "images" / "test",
            output_root / "labels" / "test"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists() or len(list(dir_path.glob("*"))) == 0:
                return False
        
        return True
    
    def _check_existing_prepared_dataset(self, ballDataset_path: Path) -> bool:
        """Check if prepared ballDataset exists in YOLO format."""
        required_dirs = [
            ballDataset_path / "images" / "train",
            ballDataset_path / "labels" / "train", 
            ballDataset_path / "images" / "test",
            ballDataset_path / "labels" / "test"
        ]
        
        # Also check for data.yaml
        data_yaml = ballDataset_path / "data.yaml"
        
        for dir_path in required_dirs:
            if not dir_path.exists() or len(list(dir_path.glob("*"))) == 0:
                return False
        
        return data_yaml.exists()
    
    def _validate_dataset_yaml(self, ballDataset_path: Path):
        """Validate the existing data.yaml file."""
        data_yaml_path = ballDataset_path / "data.yaml"
        if data_yaml_path.exists():
            try:
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                logger.info(f"Dataset configuration:")
                logger.info(f"  Path: {data.get('path')}")
                logger.info(f"  Classes: {data.get('names', {})}")
                logger.info(f"  Number of classes: {data.get('nc', len(data.get('names', {})))}")
                logger.info(f"  Train split: {data.get('train')}")
                logger.info(f"  Val split: {data.get('val')}")
                
            except Exception as e:
                logger.warning(f"Could not read data.yaml: {e}")
        else:
            logger.warning("data.yaml not found in ballDataset")
    
    def _check_existing_yolo_format_old(self, output_root: Path) -> bool:
        """Check if YOLO format dataset already exists (old format)."""
        required_dirs = [
            output_root / "train" / "images",
            output_root / "train" / "labels", 
            output_root / "val" / "images",
            output_root / "val" / "labels"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists() or len(list(dir_path.glob("*"))) == 0:
                return False
        
        return True
    
    def _convert_huggingface_to_yolo(self, ballDataset_path: Path, output_root: Path):
        """Convert Hugging Face dataset to YOLO format."""
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets library not available!")
            self._find_and_copy_prepared_dataset(output_root)
            return
            
        try:
            # Load the Hugging Face dataset
            logger.info("Loading Hugging Face dataset...")
            dataset = load_dataset("arrow", data_dir=str(ballDataset_path))
            
            # Create output directories
            train_images_dir = output_root / "train" / "images"
            train_labels_dir = output_root / "train" / "labels"
            val_images_dir = output_root / "val" / "images"
            val_labels_dir = output_root / "val" / "labels"
            
            for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Get the train split
            train_dataset = dataset['train']
            total_samples = len(train_dataset)
            
            # Split into train/val (80/20)
            train_split = int(0.8 * total_samples)
            
            logger.info(f"Converting {total_samples} samples to YOLO format...")
            logger.info(f"Train: {train_split}, Val: {total_samples - train_split}")
            
            # Process training samples
            for i in tqdm(range(train_split), desc="Converting training data"):
                sample = train_dataset[i]
                self._process_sample(sample, f"train_{i:06d}", train_images_dir, train_labels_dir)
            
            # Process validation samples
            for i in tqdm(range(train_split, total_samples), desc="Converting validation data"):
                sample = train_dataset[i]
                self._process_sample(sample, f"val_{i:06d}", val_images_dir, val_labels_dir)
                
        except Exception as e:
            logger.error(f"Error converting dataset: {e}")
            logger.info("Trying to find prepared dataset structure...")
            self._find_and_copy_prepared_dataset(output_root)
    
    def _process_sample(self, sample: Dict[str, Any], name: str, images_dir: Path, labels_dir: Path):
        """Process a single sample from the dataset."""
        try:
            # Save image
            image = sample['image']
            if isinstance(image, Image.Image):
                image_path = images_dir / f"{name}.jpg"
                image.save(image_path, quality=95)
                
                # Create corresponding label file
                label_path = labels_dir / f"{name}.txt"
                
                # For ball detection, you might need to extract bounding boxes
                # from the dataset. For now, creating placeholder files.
                # You'll need to modify this based on your actual annotation format
                with open(label_path, 'w') as f:
                    # Ball class = 0
                    # If annotations exist in the sample, process them
                    if 'objects' in sample or 'annotations' in sample:
                        # Extract and convert annotations to YOLO format
                        self._extract_annotations(sample, f, image.size)
                    # Otherwise, empty file (no ball detected in this frame)
                    
        except Exception as e:
            logger.warning(f"Error processing sample {name}: {e}")
    
    def _extract_annotations(self, sample: Dict[str, Any], label_file, image_size: Tuple[int, int]):
        """Extract annotations from sample and write to YOLO format."""
        # This is a placeholder - you'll need to implement based on your dataset format
        # Common formats include:
        # - COCO format: {"bbox": [x, y, w, h], "category_id": 1}
        # - Custom format with ball coordinates
        
        width, height = image_size
        
        # Example: if your dataset has 'objects' with bounding boxes
        if 'objects' in sample:
            for obj in sample['objects']:
                if obj.get('category') == 'ball' or obj.get('label') == 'ball':
                    # Convert to YOLO format (normalized coordinates)
                    bbox = obj.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        # Convert to normalized YOLO format
                        x_center = (x + w/2) / width
                        y_center = (y + h/2) / height
                        norm_w = w / width
                        norm_h = h / height
                        
                        # Class 1 for ball (since player is class 0)
                        label_file.write(f"1 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    def _extract_multi_class_dataset(self, ballDataset_path: Path, player_dataset_path: Path, output_root: Path):
        """Extract and combine player and ball datasets for multi-class training."""
        logger.info("Extracting multi-class dataset (player + ball)...")
        
        # Create output directories
        train_images_dir = output_root / "train" / "images"
        train_labels_dir = output_root / "train" / "labels"
        val_images_dir = output_root / "val" / "images"
        val_labels_dir = output_root / "val" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract ball dataset first
        ball_count = 0
        if ballDataset_path.exists():
            logger.info("Processing ball dataset...")
            ball_count = self._process_ball_dataset(ballDataset_path, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)
        
        # Extract player dataset
        player_count = 0
        if player_dataset_path.exists():
            logger.info("Processing player dataset...")
            player_count = self._process_player_dataset(player_dataset_path, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, ball_count)
        
        logger.info(f"✅ Multi-class dataset created:")
        logger.info(f"   Ball samples: {ball_count}")
        logger.info(f"   Player samples: {player_count}")
        logger.info(f"   Total samples: {ball_count + player_count}")
    
    def _process_ball_dataset(self, ballDataset_path: Path, train_images_dir: Path, train_labels_dir: Path, 
                            val_images_dir: Path, val_labels_dir: Path) -> int:
        """Process ball dataset with class ID 1."""
        try:
            if HF_AVAILABLE:
                dataset = load_dataset("arrow", data_dir=str(ballDataset_path))
                train_dataset = dataset['train']
                total_samples = len(train_dataset)
                train_split = int(0.8 * total_samples)
                
                # Process training samples
                for i in tqdm(range(train_split), desc="Converting ball training data"):
                    sample = train_dataset[i]
                    self._process_ball_sample(sample, f"ball_train_{i:06d}", train_images_dir, train_labels_dir)
                
                # Process validation samples
                for i in tqdm(range(train_split, total_samples), desc="Converting ball validation data"):
                    sample = train_dataset[i]
                    self._process_ball_sample(sample, f"ball_val_{i:06d}", val_images_dir, val_labels_dir)
                
                return total_samples
            else:
                logger.warning("Hugging Face datasets not available, skipping ball dataset")
                return 0
        except Exception as e:
            logger.error(f"Error processing ball dataset: {e}")
            return 0
    
    def _process_ball_sample(self, sample: Dict[str, Any], name: str, images_dir: Path, labels_dir: Path):
        """Process a ball detection sample with class ID 1."""
        try:
            image = sample['image']
            if isinstance(image, Image.Image):
                image_path = images_dir / f"{name}.jpg"
                image.save(image_path, quality=95)
                
                label_path = labels_dir / f"{name}.txt"
                with open(label_path, 'w') as f:
                    if 'objects' in sample or 'annotations' in sample:
                        self._extract_ball_annotations(sample, f, image.size)
                        
        except Exception as e:
            logger.warning(f"Error processing ball sample {name}: {e}")
    
    def _extract_ball_annotations(self, sample: Dict[str, Any], label_file, image_size: Tuple[int, int]):
        """Extract ball annotations with class ID 1."""
        width, height = image_size
        
        if 'objects' in sample:
            for obj in sample['objects']:
                if obj.get('category') == 'ball' or obj.get('label') == 'ball':
                    bbox = obj.get('bbox', [])
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        x_center = (x + w/2) / width
                        y_center = (y + h/2) / height
                        norm_w = w / width
                        norm_h = h / height
                        
                        # Class 1 for ball
                        label_file.write(f"1 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    def _process_player_dataset(self, player_dataset_path: Path, train_images_dir: Path, train_labels_dir: Path, 
                              val_images_dir: Path, val_labels_dir: Path, start_idx: int) -> int:
        """Process player dataset with class ID 0."""
        logger.info("Looking for existing player detection dataset...")
        
        # Check for existing YOLO format player dataset
        potential_player_datasets = [
            player_dataset_path / "datasets" / "yolo_detection_extracted",
            player_dataset_path.parent / "player-detection" / "datasets" / "yolo_detection_extracted",
            player_dataset_path / "yolo_detection_extracted"
        ]
        
        for dataset_path in potential_player_datasets:
            if self._check_existing_yolo_format(dataset_path):
                logger.info(f"Found player dataset at: {dataset_path}")
                return self._copy_player_samples(dataset_path, train_images_dir, train_labels_dir, 
                                               val_images_dir, val_labels_dir, start_idx)
        
        logger.warning("No existing player dataset found!")
        logger.info("To include player detection, you can:")
        logger.info("1. Run player-detection training first to generate dataset")
        logger.info("2. Manually prepare player dataset in YOLO format")
        logger.info("3. Continue with ball-only detection for now")
        
        return 0
    
    def _copy_player_samples(self, source_dataset: Path, train_images_dir: Path, train_labels_dir: Path,
                           val_images_dir: Path, val_labels_dir: Path, start_idx: int) -> int:
        """Copy player samples and update class IDs to 0."""
        total_copied = 0
        
        # Copy training samples
        source_train_images = source_dataset / "train" / "images"
        source_train_labels = source_dataset / "train" / "labels"
        
        if source_train_images.exists():
            for i, img_path in enumerate(source_train_images.glob("*.jpg")):
                new_name = f"player_train_{start_idx + i:06d}.jpg"
                shutil.copy2(img_path, train_images_dir / new_name)
                
                # Copy and update label file
                label_path = source_train_labels / img_path.with_suffix('.txt').name
                if label_path.exists():
                    self._copy_and_update_label(label_path, train_labels_dir / new_name.replace('.jpg', '.txt'))
                total_copied += 1
        
        # Copy validation samples
        source_val_images = source_dataset / "val" / "images"
        source_val_labels = source_dataset / "val" / "labels"
        
        if source_val_images.exists():
            for i, img_path in enumerate(source_val_images.glob("*.jpg")):
                new_name = f"player_val_{start_idx + i:06d}.jpg"
                shutil.copy2(img_path, val_images_dir / new_name)
                
                # Copy and update label file
                label_path = source_val_labels / img_path.with_suffix('.txt').name
                if label_path.exists():
                    self._copy_and_update_label(label_path, val_labels_dir / new_name.replace('.jpg', '.txt'))
                total_copied += 1
        
        return total_copied
    
    def _copy_and_update_label(self, source_label: Path, dest_label: Path):
        """Copy label file and update class IDs (player = 0)."""
        with open(source_label, 'r') as src, open(dest_label, 'w') as dst:
            for line in src:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Set class to 0 for player, keep other coordinates
                    dst.write(f"0 {' '.join(parts[1:])}\n")
                elif line.strip():  # Non-empty line
                    dst.write(line)
    
    def _find_and_copy_prepared_dataset(self, output_root: Path):
        """Find and copy existing prepared dataset structure."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        
        # Check various possible locations for prepared dataset
        potential_locations = [
            workspace_root / "ballDataset" / "yolo_format",
            workspace_root / "ballDataset" / "dataset",
            workspace_root / "datasets" / "ball_detection",
            workspace_root.parent / "datasets" / "ball_detection",
            # Check if extracted dataset already exists
            workspace_root / "datasets" / "ball_detection_extracted"
        ]
        
        for location in potential_locations:
            if self._check_existing_yolo_format(location):
                logger.info(f"Found prepared dataset at: {location}")
                self._copy_dataset_structure(location, output_root)
                return
        
        # If no prepared dataset found, create minimal structure with warning
        logger.warning("No prepared ball detection dataset found!")
        logger.warning("Please prepare your dataset in YOLO format with the following structure:")
        logger.warning("  train/images/ - training images")
        logger.warning("  train/labels/ - training labels (.txt files)")
        logger.warning("  val/images/ - validation images") 
        logger.warning("  val/labels/ - validation labels (.txt files)")
        
        # Create empty directories
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                (output_root / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def _copy_dataset_structure(self, source: Path, destination: Path):
        """Copy dataset structure from source to destination."""
        logger.info(f"Copying dataset from {source} to {destination}")
        
        if destination.exists():
            shutil.rmtree(destination)
        
        shutil.copytree(source, destination)
        logger.info("✅ Dataset copied successfully")
    
    def _create_dataset_yaml(self, output_root: Path):
        """Create dataset.yaml file for YOLO training."""
        class_names = self.config.get('class_names', ['player', 'ball'])
        
        dataset_config = {
            'path': str(output_root.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = output_root / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"✅ Dataset YAML created: {yaml_path}")
        logger.info(f"Classes: {class_names}")
        logger.info(f"Class mapping: {dict(enumerate(class_names))}")
    
    def ensure_dataset_yaml(self) -> Path:
        """Ensure dataset.yaml exists and return its path."""
        # First check if we have prepared ballDataset
        ballDataset_path = self._get_ball_dataset_path()
        ballDataset_yaml = ballDataset_path / "data.yaml"
        
        if ballDataset_yaml.exists() and self._check_existing_prepared_dataset(ballDataset_path):
            logger.info(f"Using prepared ballDataset: {ballDataset_yaml}")
            return ballDataset_yaml
        
        # Fallback to original logic
        output_root = self._get_output_root()
        yaml_path = output_root / "dataset.yaml"
        
        if not yaml_path.exists():
            if not output_root.exists():
                logger.warning("Dataset not extracted yet. Please run extraction first.")
                # Create minimal structure
                self.extract_dataset()
            else:
                self._create_dataset_yaml(output_root)
        
        return yaml_path


def create_extractor(extraction_config: Dict[str, Any], paths_config: Dict[str, Any]) -> DatasetExtractor:
    """Factory function to create dataset extractor."""
    return DatasetExtractor(extraction_config, paths_config)