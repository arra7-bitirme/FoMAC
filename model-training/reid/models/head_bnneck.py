"""
BNNeck Head for ReID

Implements BatchNorm Neck with classification head and L2 normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeck(nn.Module):
    """
    BatchNorm Neck for ReID.
    
    Architecture:
        Input features → BatchNorm → Classifier (for training)
                      → L2 Normalize → Embedding (for inference)
    
    The BatchNorm neck helps reduce intra-class variance and
    improves triplet loss training.
    """
    
    def __init__(self, in_feat: int, num_classes: int, emb_dim: int = 256):
        """
        Args:
            in_feat: Input feature dimension (e.g., 2048 for ResNet50)
            num_classes: Number of person identities
            emb_dim: Embedding dimension after projection
        """
        super(BNNeck, self).__init__()
        
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        
        # Bottleneck layer to reduce dimension
        self.bottleneck = nn.Linear(in_feat, emb_dim)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(emb_dim)
        self.bn.bias.requires_grad_(False)  # No bias for BN before classifier
        
        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        nn.init.kaiming_normal_(self.bottleneck.weight, mode='fan_out')
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def forward(self, features, return_logits: bool = True):
        """
        Forward pass.
        
        Args:
            features: Input features (B, in_feat)
            return_logits: If True, return (embedding, logits)
                          If False, return only embedding
            
        Returns:
            If return_logits:
                embedding: L2-normalized embedding (B, emb_dim)
                logits: Classification logits (B, num_classes)
            Else:
                embedding: L2-normalized embedding (B, emb_dim)
        """
        # Project to embedding dimension
        feat = self.bottleneck(features)  # (B, emb_dim)
        
        # Batch normalization
        bn_feat = self.bn(feat)  # (B, emb_dim)
        
        # L2 normalization for embedding
        embedding = F.normalize(feat, p=2, dim=1)  # (B, emb_dim)
        
        if return_logits:
            # Classification logits (using BN features)
            logits = self.classifier(bn_feat)  # (B, num_classes)
            return embedding, logits
        else:
            return embedding


class ReIDModelWithBNNeck(nn.Module):
    """
    Complete ReID model with backbone and BNNeck head.
    
    This combines the ResNet50 backbone with the BNNeck head
    for end-to-end training.
    """
    
    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 256,
        pretrained: bool = True,
        last_stride: int = 1
    ):
        """
        Args:
            num_classes: Number of person identities
            emb_dim: Embedding dimension
            pretrained: Use ImageNet pretrained weights
            last_stride: Stride for last conv layer in backbone
        """
        super(ReIDModelWithBNNeck, self).__init__()
        
        # Import backbone here to avoid circular imports
        from .backbone_resnet50 import ResNet50Backbone
        
        # Backbone
        self.backbone = ResNet50Backbone(
            pretrained=pretrained,
            last_stride=last_stride
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # BNNeck head
        self.head = BNNeck(
            in_feat=self.backbone.feat_dim,
            num_classes=num_classes,
            emb_dim=emb_dim
        )
        
        self.emb_dim = emb_dim
        self.num_classes = num_classes
    
    def forward(self, x, return_logits: bool = True):
        """
        Args:
            x: Input images (B, 3, H, W)
            return_logits: Whether to return classification logits
            
        Returns:
            If return_logits (training mode):
                embedding: L2-normalized embedding (B, emb_dim)
                logits: Classification logits (B, num_classes)
            Else (inference mode):
                embedding: L2-normalized embedding (B, emb_dim)
        """
        # Extract features
        feat_map = self.backbone(x)  # (B, 2048, H', W')
        
        # Global average pooling
        global_feat = self.gap(feat_map)  # (B, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.size(0), -1)  # (B, 2048)
        
        # BNNeck head
        return self.head(global_feat, return_logits=return_logits)


def build_reid_model(
    num_classes: int,
    emb_dim: int = 256,
    pretrained: bool = True,
    last_stride: int = 1
):
    """
    Factory function to build complete ReID model.
    
    Args:
        num_classes: Number of person identities
        emb_dim: Embedding dimension
        pretrained: Use ImageNet pretrained weights
        last_stride: Stride for last conv layer
        
    Returns:
        ReIDModelWithBNNeck
    """
    return ReIDModelWithBNNeck(
        num_classes=num_classes,
        emb_dim=emb_dim,
        pretrained=pretrained,
        last_stride=last_stride
    )


def test_bnneck():
    """Test function for BNNeck."""
    print("Testing BNNeck Head...")
    
    batch_size = 4
    in_feat = 2048
    num_classes = 100
    emb_dim = 256
    
    # Create model
    head = BNNeck(in_feat, num_classes, emb_dim)
    
    # Test forward pass (training)
    features = torch.randn(batch_size, in_feat)
    embedding, logits = head(features, return_logits=True)
    
    print(f"Input shape: {features.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Check L2 normalization
    norms = torch.norm(embedding, p=2, dim=1)
    print(f"Embedding norms (should be ~1.0): {norms}")
    
    assert embedding.shape == (batch_size, emb_dim)
    assert logits.shape == (batch_size, num_classes)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    # Test forward pass (inference)
    embedding_only = head(features, return_logits=False)
    assert embedding_only.shape == (batch_size, emb_dim)
    
    print("✓ BNNeck test passed!")


def test_full_model():
    """Test complete ReID model."""
    print("\nTesting Full ReID Model...")
    
    batch_size = 2
    num_classes = 50
    emb_dim = 256
    
    # Create model
    model = build_reid_model(
        num_classes=num_classes,
        emb_dim=emb_dim,
        pretrained=False
    )
    
    # Test forward pass
    x = torch.randn(batch_size, 3, 256, 128)
    embedding, logits = model(x, return_logits=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Logits shape: {logits.shape}")
    
    assert embedding.shape == (batch_size, emb_dim)
    assert logits.shape == (batch_size, num_classes)
    
    # Test inference mode
    embedding_only = model(x, return_logits=False)
    assert embedding_only.shape == (batch_size, emb_dim)
    
    print("✓ Full model test passed!")


if __name__ == "__main__":
    test_bnneck()
    test_full_model()
