"""
ResNet feature extractor for ProtoPNet.
Extracts intermediate layer features for prototype matching.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatures(nn.Module):
    """
    ResNet backbone that outputs intermediate features instead of classifications.

    Args:
        arch: ResNet architecture ('resnet50', 'resnet34', etc.)
        pretrained: Whether to load ImageNet pretrained weights
        layer_name: Which layer to extract features from ('layer3' recommended)
    """

    def __init__(self, arch='resnet50', pretrained=True, layer_name='layer3'):
        super().__init__()

        # Load pretrained ResNet
        if arch == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 1024 if layer_name == 'layer3' else 2048
        elif arch == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 256 if layer_name == 'layer3' else 512
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Extract layers up to specified layer
        self.layer_name = layer_name
        layers = []
        layers.append(resnet.conv1)
        layers.append(resnet.bn1)
        layers.append(resnet.relu)
        layers.append(resnet.maxpool)
        layers.append(resnet.layer1)
        layers.append(resnet.layer2)
        layers.append(resnet.layer3)

        if layer_name == 'layer4':
            layers.append(resnet.layer4)

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, 224, 224)
        Returns:
            features: Feature maps (B, feature_dim, H, W)
                     For layer3: (B, 1024, 14, 14)
                     For layer4: (B, 2048, 7, 7)
        """
        return self.features(x)

    def output_shape(self, input_size=224):
        """Calculate output spatial dimensions"""
        # For ImageNet-style ResNet with 224x224 input:
        # After initial conv+pool: 56x56
        # After layer1: 56x56
        # After layer2: 28x28
        # After layer3: 14x14
        # After layer4: 7x7
        if self.layer_name == 'layer3':
            return (self.feature_dim, 14, 14)
        else:
            return (self.feature_dim, 7, 7)
