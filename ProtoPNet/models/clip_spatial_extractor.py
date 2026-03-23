"""
CLIP Spatial Feature Extractor: Extract spatial features from CLIP ResNet-50 layer3.

This module extracts intermediate spatial features (B, 1024, 14, 14) from CLIP's ResNet-50
before the final attention pooling. These features serve as targets for training the
SpatialMSEProjector with per-location MSE loss.

Key Features:
- Extracts layer3 features from CLIP ResNet-50 (before layer4 and attention pooling)
- Handles normalization conversion (ImageNet → CLIP)
- Preserves spatial structure (14×14 resolution)
- Frozen weights (used only for feature extraction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional


class CLIPSpatialExtractor(nn.Module):
    """
    Extract spatial features from CLIP ResNet-50 layer3.

    CLIP's ResNet-50 architecture:
        Input (B, 3, 224, 224)
        → layer1: (B, 256, 56, 56)
        → layer2: (B, 512, 28, 28)
        → layer3: (B, 1024, 14, 14) ← WE EXTRACT HERE
        → layer4: (B, 2048, 7, 7)
        → Attention pooling: (B, 2048)
        → Projection: (B, 512)

    We extract from layer3 because:
    - Matches ProtoPNet's spatial resolution (14×14)
    - Rich semantic features before final pooling
    - 1024-dim feature dimension (good capacity)

    Args:
        layer: Which layer to extract ('layer3' or 'layer4')
        device: Device to load model ('cuda' or 'cpu')
        normalize_input: Whether to convert ImageNet normalization to CLIP

    Example:
        >>> extractor = CLIPSpatialExtractor(layer='layer3', device='cuda')
        >>> images = torch.randn(32, 3, 224, 224).cuda()  # ImageNet normalized
        >>> spatial_features = extractor(images)  # (32, 1024, 14, 14)
        >>> assert spatial_features.shape == (32, 1024, 14, 14)
    """

    def __init__(
        self,
        layer: str = 'layer3',
        device: str = 'cuda',
        normalize_input: bool = True,
        extract_final: bool = False
    ):
        super().__init__()
        self.layer_name = layer
        self.device = device
        self.normalize_input = normalize_input
        self.extract_final = extract_final

        # Load CLIP ResNet-50
        print(f"Loading CLIP ResNet-50 model...")
        self.clip_model, _ = clip.load('RN50', device=device)
        self.clip_resnet = self.clip_model.visual

        # Freeze all parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # Storage for hooked features
        self.features = None

        # Register forward hook
        self._register_hook()

        # Normalization parameters
        # ImageNet normalization (used by ProtoPNet)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # CLIP normalization
        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        print(f"CLIPSpatialExtractor initialized: extracting {layer} features")

    def _register_hook(self):
        """Register forward hook to capture intermediate features."""
        def hook_fn(module, input, output):
            self.features = output

        # Get the target layer
        if self.layer_name == 'layer3':
            target_layer = self.clip_resnet.layer3
        elif self.layer_name == 'layer4':
            target_layer = self.clip_resnet.layer4
        else:
            raise ValueError(f"Unsupported layer: {self.layer_name}. Use 'layer3' or 'layer4'.")

        # Register hook
        target_layer.register_forward_hook(hook_fn)

    def convert_normalization(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert ImageNet normalization to CLIP normalization.

        Args:
            images: (B, 3, H, W) - ImageNet normalized images

        Returns:
            images_clip: (B, 3, H, W) - CLIP normalized images
        """
        # Move buffers to same device as input
        device = images.device
        imagenet_mean = self.imagenet_mean.to(device)
        imagenet_std = self.imagenet_std.to(device)
        clip_mean = self.clip_mean.to(device)
        clip_std = self.clip_std.to(device)

        # Denormalize from ImageNet
        images_denorm = images * imagenet_std + imagenet_mean

        # Normalize to CLIP
        images_clip = (images_denorm - clip_mean) / clip_std

        return images_clip

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features or final embeddings from CLIP ResNet-50.

        Args:
            images: (B, 3, 224, 224) - Images (ImageNet normalized by default)

        Returns:
            If extract_final=True:
                final_embeddings: (B, 1024) - Final CLIP embeddings for retrieval
            If extract_final=False:
                spatial_features: (B, C, H, W) - Spatial features
                    - layer3: (B, 1024, 14, 14)
                    - layer4: (B, 2048, 7, 7)

        Note:
            - Model is in eval mode (no gradients)
            - Features are NOT normalized (do that during loss computation)
        """
        B = images.shape[0]

        with torch.no_grad():
            # Convert normalization if needed
            if self.normalize_input:
                images = self.convert_normalization(images)

            # Forward through CLIP ResNet
            final_output = self.clip_resnet(images)

            # Return final embeddings or hooked spatial features
            if self.extract_final:
                return final_output  # (B, 1024) - final embeddings after attention pooling
            else:
                # Get hooked spatial features
                spatial_features = self.features
                return spatial_features

    def get_feature_dim(self) -> tuple:
        """
        Get expected output dimensions for the selected layer.

        Returns:
            (channels, height, width)
        """
        if self.layer_name == 'layer3':
            return (1024, 14, 14)
        elif self.layer_name == 'layer4':
            return (2048, 7, 7)
        else:
            raise ValueError(f"Unknown layer: {self.layer_name}")

    def __repr__(self):
        return (f"CLIPSpatialExtractor("
                f"layer={self.layer_name}, "
                f"device={self.device}, "
                f"normalize_input={self.normalize_input}, "
                f"extract_final={self.extract_final})")


def extract_clip_layer3_features(
    images: torch.Tensor,
    extractor: Optional[CLIPSpatialExtractor] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convenience function to extract CLIP layer3 features.

    Args:
        images: (B, 3, 224, 224) - ImageNet normalized images
        extractor: Pre-initialized extractor (optional, will create if None)
        device: Device to use

    Returns:
        spatial_features: (B, 1024, 14, 14) - Layer3 spatial features
    """
    if extractor is None:
        extractor = CLIPSpatialExtractor(layer='layer3', device=device)

    return extractor(images)
