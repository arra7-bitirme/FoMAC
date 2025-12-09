"""ReID model architectures."""

from .backbone_resnet50 import ResNet50Backbone
from .head_bnneck import BNNeck

__all__ = ["ResNet50Backbone", "BNNeck"]
