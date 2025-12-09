"""
ResNet50 Backbone for ReID

Feature extraction using ResNet50 pretrained on ImageNet.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for feature extraction.
    
    Uses ImageNet pretrained weights and removes the final FC layer.
    """
    
    def __init__(self, pretrained: bool = True, last_stride: int = 1):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            last_stride: Stride for the last convolutional layer
                        (1 for higher resolution, 2 for lower)
        """
        super(ResNet50Backbone, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify last stride if needed
        if last_stride == 1:
            # Change stride from 2 to 1 in layer4
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        
        # Remove the final FC and pooling layers
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Output feature dimension
        self.feat_dim = 2048
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: Tensor of shape (B, 2048, H', W')
        """
        features = self.backbone(x)
        return features


class ReIDModel(nn.Module):
    """
    Complete ReID model with ResNet50 backbone.
    
    This is a simple wrapper that adds global average pooling
    after the backbone.
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        last_stride: int = 1
    ):
        """
        Args:
            num_classes: Number of person identities
            pretrained: Use ImageNet pretrained weights
            last_stride: Stride for last conv layer
        """
        super(ReIDModel, self).__init__()
        
        self.backbone = ResNet50Backbone(
            pretrained=pretrained,
            last_stride=last_stride
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension
        self.feat_dim = self.backbone.feat_dim
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            global_feat: Global features (B, feat_dim)
        """
        # Extract features
        feat_map = self.backbone(x)  # (B, 2048, H', W')
        
        # Global average pooling
        global_feat = self.gap(feat_map)  # (B, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.size(0), -1)  # (B, 2048)
        
        return global_feat


def build_resnet50_backbone(pretrained: bool = True, last_stride: int = 1):
    """
    Factory function to build ResNet50 backbone.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        last_stride: Stride for last conv layer
        
    Returns:
        ResNet50Backbone model
    """
    return ResNet50Backbone(pretrained=pretrained, last_stride=last_stride)


def test_backbone():
    """Test function for backbone."""
    print("Testing ResNet50 Backbone...")
    
    # Create model
    backbone = ResNet50Backbone(pretrained=False, last_stride=1)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 128)
    features = backbone(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dimension: {backbone.feat_dim}")
    
    assert features.shape[0] == 2
    assert features.shape[1] == 2048
    
    print("✓ Backbone test passed!")


if __name__ == "__main__":
    test_backbone()
