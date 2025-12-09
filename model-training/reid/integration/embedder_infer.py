"""
ReID Embedder for Inference

Provides interface for extracting embeddings from player crops.
"""

from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class ReIDEmbedder:
    """
    ReID embedding extractor for inference.
    
    Usage:
        embedder = ReIDEmbedder('path/to/best_reid.pt')
        embedding = embedder.get_embedding(image, bbox)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: Path to exported ReID model
            device: Compute device ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        
        # Load model
        from ..engine.export import load_exported_model
        
        print(f"Loading ReID model from {model_path}")
        self.model, self.cfg = load_exported_model(model_path, self.device)
        self.model.eval()
        
        # Setup transform
        self.transform = self._build_transform()
        
        print(f"✓ ReID embedder initialized on {self.device}")
    
    def _build_transform(self):
        """Build image preprocessing transform."""
        height = self.cfg['data']['height']
        width = self.cfg['data']['width']
        mean = self.cfg['data']['mean']
        std = self.cfg['data']['std']
        
        transform = T.Compose([
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return transform
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for ReID model.
        
        Args:
            image: BGR image (H, W, 3) as numpy array
            
        Returns:
            tensor: Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transform
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def get_embedding(
        self,
        image: Union[np.ndarray, str, Path],
        bbox: Tuple[int, int, int, int] = None
    ) -> np.ndarray:
        """
        Extract ReID embedding from image.
        
        Args:
            image: Input image (numpy array or path to image)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop
            
        Returns:
            embedding: L2-normalized embedding (256,) as numpy array
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        
        # Crop if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            image = image[y1:y2, x1:x2]
        
        if image.size == 0:
            raise ValueError("Empty crop")
        
        # Preprocess
        tensor = self.preprocess_image(image).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor, return_logits=False)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]  # (256,)
        
        return embedding
    
    def get_embeddings_batch(
        self,
        images: list,
        bboxes: list = None
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of images.
        
        Args:
            images: List of images (numpy arrays or paths)
            bboxes: Optional list of bounding boxes
            
        Returns:
            embeddings: (N, 256) numpy array
        """
        if bboxes is None:
            bboxes = [None] * len(images)
        
        batch_tensors = []
        
        for image, bbox in zip(images, bboxes):
            # Load image if needed
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
            
            # Crop if needed
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))
                image = image[y1:y2, x1:x2]
            
            # Preprocess
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor)
        
        # Stack batch
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(batch, return_logits=False)
        
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()  # (N, 256)
        
        return embeddings


# Global embedder instance (singleton pattern)
_embedder_instance = None


def get_embedding(
    image: Union[np.ndarray, str],
    bbox: Tuple[int, int, int, int] = None,
    model_path: str = None
) -> np.ndarray:
    """
    Convenience function to extract ReID embedding.
    
    This function maintains a global embedder instance for efficiency.
    
    Args:
        image: Input image (numpy array or path)
        bbox: Optional bounding box (x1, y1, x2, y2)
        model_path: Path to ReID model (required for first call)
        
    Returns:
        embedding: L2-normalized embedding (256,) as numpy array
    
    Example:
        >>> embedding = get_embedding(crop_image)
        >>> print(embedding.shape)  # (256,)
    """
    global _embedder_instance
    
    # Initialize embedder if needed
    if _embedder_instance is None:
        if model_path is None:
            raise ValueError(
                "model_path required for first call to get_embedding()"
            )
        _embedder_instance = ReIDEmbedder(model_path)
    
    return _embedder_instance.get_embedding(image, bbox)


def reset_embedder():
    """Reset global embedder instance."""
    global _embedder_instance
    _embedder_instance = None


def test_embedder():
    """Test embedder functionality."""
    print("Testing ReID Embedder...")
    
    # Create dummy model for testing
    import tempfile
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from models.head_bnneck import build_reid_model
    
    # Build dummy model
    model = build_reid_model(
        num_classes=10,
        emb_dim=256,
        pretrained=False
    )
    
    # Create dummy export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    export_data = {
        'model_state_dict': model.state_dict(),
        'cfg': {
            'model': {'emb_dim': 256, 'last_stride': 1},
            'data': {
                'height': 256,
                'width': 128,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'num_classes': 10,
        'emb_dim': 256
    }
    
    torch.save(export_data, temp_path)
    
    try:
        # Test embedder
        embedder = ReIDEmbedder(temp_path, device='cpu')
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Test single embedding
        embedding = embedder.get_embedding(dummy_image)
        
        print(f"✓ Single embedding shape: {embedding.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        assert embedding.shape == (256,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01
        
        # Test batch embeddings
        dummy_images = [dummy_image, dummy_image]
        embeddings = embedder.get_embeddings_batch(dummy_images)
        
        print(f"✓ Batch embeddings shape: {embeddings.shape}")
        
        assert embeddings.shape == (2, 256)
        
        print("✓ All tests passed!")
        
    finally:
        # Cleanup
        import os
        os.unlink(temp_path)


if __name__ == "__main__":
    test_embedder()
