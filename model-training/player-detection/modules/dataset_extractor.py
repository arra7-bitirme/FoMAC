"""
Dataset Extraction Module

Handles extraction of frames and labels from SoccerNet dataset
with frame-label alignment preservation.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import logging
import sys

logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Extracts frames and YOLO-format labels from SoccerNet dataset."""
    
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
        Extract complete dataset from SoccerNet.
        
        Returns:
            Path to the output dataset root
        """
        if not self.config.get('run_extraction', True):
            logger.info("Dataset extraction skipped (run_extraction=False)")
            return self._get_output_root()
            
        logger.info("=" * 80)
        logger.info("DATASET EXTRACTION (alignment-preserving)")
        logger.info("=" * 80)
        
        soccernet_root = self._get_soccernet_root()
        output_root = self._get_output_root()
        
        logger.info(f"Source: {soccernet_root}")
        logger.info(f"Output: {output_root}")
        
        self._batch_extract_dataset(soccernet_root, output_root)
        self._create_dataset_yaml(output_root)
        
        logger.info("✅ Dataset extraction completed")
        return output_root
    
    def _get_soccernet_root(self) -> Path:
        """Get SoccerNet root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['soccernet_root']
    
    def _get_output_root(self) -> Path:
        """Get output dataset root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['output_root']
    
    def _batch_extract_dataset(self, soccernet_root: Path, output_root: Path):
        """Extract dataset from all matches."""
        matches = self._find_matches(soccernet_root)
        splits = self._split_by_season(matches)
        
        for split_name, match_list in splits.items():
            if not match_list:
                continue
                
            logger.info(f"Processing {split_name} split: {len(match_list)} matches")
            split_output = output_root / split_name
            
            for match_dir in tqdm(match_list, desc=f"Extracting {split_name}"):
                self._extract_match(match_dir, split_output)
    
    def _find_matches(self, root: Path) -> List[Path]:
        """Find all valid match directories."""
        matches = []
        video_pattern = self.paths_config['video_pattern']
        annotation_pattern = self.paths_config['annotation_pattern']
        
        for league in root.iterdir():
            if not league.is_dir():
                continue
            for season in league.iterdir():
                if not season.is_dir():
                    continue
                for match_dir in season.iterdir():
                    if not match_dir.is_dir():
                        continue
                    
                    # Check if both halves have required files
                    valid = False
                    for half in (1, 2):
                        video_file = match_dir / video_pattern.format(half=half)
                        annotation_file = match_dir / annotation_pattern.format(half=half)
                        if video_file.exists() and annotation_file.exists():
                            valid = True
                            break
                    
                    if valid:
                        matches.append(match_dir)
        
        return sorted(matches)
    
    def _split_by_season(self, matches: List[Path]) -> Dict[str, List[Path]]:
        """Split matches by season according to configuration."""
        season_splits = self.config.get('season_splits', {
            'train': ['2014-2015'],
            'val': ['2015-2016'],
            'test': ['2016-2017']
        })
        
        splits = {split: [] for split in season_splits.keys()}
        
        for match in matches:
            season = match.parent.name
            for split_name, seasons in season_splits.items():
                if season in seasons:
                    splits[split_name].append(match)
                    break
        
        return splits
    
    def _extract_match(self, match_dir: Path, split_output: Path):
        """Extract frames and labels from a single match."""
        video_pattern = self.paths_config['video_pattern']
        annotation_pattern = self.paths_config['annotation_pattern']
        
        for half in (1, 2):
            video_file = match_dir / video_pattern.format(half=half)
            annotation_file = match_dir / annotation_pattern.format(half=half)
            
            if not (video_file.exists() and annotation_file.exists()):
                continue
            
            match_name = match_dir.name.replace(' ', '_').replace('-', '_')
            output_dir = split_output / f"{match_name}_half{half}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._extract_frames_and_labels(
                str(annotation_file),
                str(video_file),
                str(output_dir)
            )
    
    def _extract_frames_and_labels(
        self,
        json_path: str,
        video_path: str,
        output_dir: str
    ) -> int:
        """Extract frames and YOLO labels from a single video-annotation pair."""
        # Import visualization functions for alignment consistency
        try:
            # Try absolute import first
            from utils.visualization_utils import load_soccernet_json, make_scaler
        except ImportError:
            try:
                # Try adding current directory to path
                import sys
                from pathlib import Path as PathLib
                current_dir = PathLib(__file__).parent.parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                from utils.visualization_utils import load_soccernet_json, make_scaler
            except ImportError:
                # Fallback to original training directory
                try:
                    training_dir = PathLib(__file__).parent.parent.parent / "training"
                    if str(training_dir) not in sys.path:
                        sys.path.append(str(training_dir))
                    from visualize_maskrcnn_soccernet import load_soccernet_json, make_scaler
                except ImportError as e:
                    logger.error(f"Failed to import visualization functions: {e}")
                    raise
        
        out_root = Path(output_dir)
        img_dir = out_root / "images"
        lbl_dir = out_root / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        preds, size = load_soccernet_json(json_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_total_sec = (total_frames / fps) if total_frames > 0 else max(1.0, len(preds) / 2.0)
        
        # Get extraction parameters
        det_fps = self.config.get('det_fps')
        if det_fps is None:
            det_fps = (len(preds) / video_total_sec) if video_total_sec > 0 else 2.0
        
        det_start_sec = self.config.get('det_start_sec', 0.0)
        frame_shift = self.config.get('frame_shift', 0)
        max_samples = self.config.get('max_samples_per_half')
        min_box_size = self.config.get('min_box_size', 10)
        scale_mode = self.config.get('scale_mode', 'scale')
        jpeg_quality = self.config.get('jpeg_quality', 95)
        
        # Source resolution for bboxes
        if size and len(size) >= 3:
            srcH, srcW = int(size[1]), int(size[2])
        else:
            srcW, srcH = vW, vH
        scaler = make_scaler((srcW, srcH), (vW, vH), mode=scale_mode)
        
        det_count = len(preds)
        if det_count == 0:
            cap.release()
            return 0
        
        # Compute detection times
        t_det = np.array([i / det_fps + det_start_sec for i in range(det_count)], dtype=float)
        
        # Decide how many to export
        num_preds = det_count if max_samples is None else min(det_count, max_samples)
        
        exported = 0
        
        for i in range(num_preds):
            det = preds[i]
            frame_idx = int(np.floor(t_det[i] * fps)) + frame_shift
            frame_idx = max(0, min(total_frames - 1, frame_idx))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            bboxes = det.get("bboxes", []) or []
            if not bboxes:
                continue
            
            # Save image
            img_name = f"{Path(video_path).stem}_pred{i:06d}_frame{frame_idx:06d}.jpg"
            img_path = img_dir / img_name
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            # Write labels
            lbl_path = lbl_dir / img_name.replace(".jpg", ".txt")
            valid = 0
            
            with open(lbl_path, "w") as f:
                for bb in bboxes:
                    x1, y1, x2, y2 = map(float, bb)
                    
                    # Scale/clamp to frame size
                    X1, Y1, X2, Y2 = scaler(x1, y1, x2, y2)
                    
                    if self.config.get('validate_boxes', True):
                        X1 = max(0, min(X1, vW - 1))
                        Y1 = max(0, min(Y1, vH - 1))
                        X2 = max(0, min(X2, vW))
                        Y2 = max(0, min(Y2, vH))
                    
                    w = X2 - X1
                    h = Y2 - Y1
                    if w < min_box_size or h < min_box_size or X2 <= X1 or Y2 <= Y1:
                        continue
                    
                    # YOLO normalized format
                    x_center = ((X1 + X2) / 2) / vW
                    y_center = ((Y1 + Y2) / 2) / vH
                    nw = w / vW
                    nh = h / vH
                    
                    # Clamp to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))
                    
                    # For now, all detections are "player" class (index 0)
                    # TODO: Add class detection logic when multi-class data becomes available
                    cls_id = 0  # player class
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")
                    valid += 1
            
            if valid == 0:
                # Remove empty image/label
                try:
                    img_path.unlink(missing_ok=True)
                    lbl_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                exported += 1
        
        cap.release()
        return exported
    
    def _create_dataset_yaml(self, output_root: Path):
        """Create dataset.yaml configuration file."""
        class_names = self.config.get('class_names', ['player'])
        
        config = {
            'path': str(output_root.absolute()),
            'train': self.paths_config.get('train_split', 'train'),
            'val': self.paths_config.get('val_split', 'val'),
            'test': self.paths_config.get('test_split', 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = output_root / self.paths_config.get('dataset_yaml', 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        logger.info(f"Created dataset configuration: {yaml_path}")
    
    def ensure_dataset_yaml(self) -> Path:
        """Ensure dataset.yaml exists and return its path."""
        output_root = self._get_output_root()
        yaml_path = output_root / self.paths_config.get('dataset_yaml', 'dataset.yaml')
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"dataset.yaml not found at {yaml_path}. "
                f"Enable run_extraction or fix paths in configuration."
            )
        
        return yaml_path


def create_extractor(config: Dict[str, Any], paths_config: Dict[str, Any]) -> DatasetExtractor:
    """
    Factory function to create a dataset extractor.
    
    Args:
        config: Extraction configuration
        paths_config: Path configuration
        
    Returns:
        Configured DatasetExtractor instance
    """
    return DatasetExtractor(config, paths_config)