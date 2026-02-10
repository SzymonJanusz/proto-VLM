"""
Normalization Utilities: Convert between ImageNet and CLIP normalization schemes.

ProtoPNet uses ImageNet normalization while CLIP uses its own normalization.
This module provides utilities to convert between these two schemes.

Normalization Constants:
- ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- CLIP: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
"""

import torch
import torch.nn as nn


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def denormalize_imagenet(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Remove ImageNet normalization from tensor.

    Args:
        tensor: (B, 3, H, W) - ImageNet normalized tensor
        mean: ImageNet mean (default: IMAGENET_MEAN)
        std: ImageNet std (default: IMAGENET_STD)

    Returns:
        denormalized: (B, 3, H, W) - Denormalized tensor in [0, 1] range

    Formula:
        denormalized = tensor * std + mean
    """
    device = tensor.device
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)

    denormalized = tensor * std_tensor + mean_tensor

    return denormalized


def normalize_clip(
    tensor: torch.Tensor,
    mean: list = CLIP_MEAN,
    std: list = CLIP_STD
) -> torch.Tensor:
    """
    Apply CLIP normalization to tensor.

    Args:
        tensor: (B, 3, H, W) - Raw tensor in [0, 1] range
        mean: CLIP mean (default: CLIP_MEAN)
        std: CLIP std (default: CLIP_STD)

    Returns:
        normalized: (B, 3, H, W) - CLIP normalized tensor

    Formula:
        normalized = (tensor - mean) / std
    """
    device = tensor.device
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)

    normalized = (tensor - mean_tensor) / std_tensor

    return normalized


def convert_imagenet_to_clip(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert ImageNet normalized tensor to CLIP normalized tensor.

    Args:
        tensor: (B, 3, H, W) - ImageNet normalized tensor

    Returns:
        clip_tensor: (B, 3, H, W) - CLIP normalized tensor

    Process:
        1. Denormalize from ImageNet (get raw [0, 1] values)
        2. Normalize to CLIP
    """
    # Step 1: Denormalize ImageNet
    denorm = denormalize_imagenet(tensor, IMAGENET_MEAN, IMAGENET_STD)

    # Step 2: Normalize to CLIP
    clip_norm = normalize_clip(denorm, CLIP_MEAN, CLIP_STD)

    return clip_norm


def convert_clip_to_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert CLIP normalized tensor to ImageNet normalized tensor.

    Args:
        tensor: (B, 3, H, W) - CLIP normalized tensor

    Returns:
        imagenet_tensor: (B, 3, H, W) - ImageNet normalized tensor

    Process:
        1. Denormalize from CLIP (get raw [0, 1] values)
        2. Normalize to ImageNet
    """
    # Step 1: Denormalize CLIP (use denormalize function with CLIP params)
    device = tensor.device
    clip_mean_tensor = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std_tensor = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    denorm = tensor * clip_std_tensor + clip_mean_tensor

    # Step 2: Normalize to ImageNet
    imagenet_mean_tensor = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std_tensor = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    imagenet_norm = (denorm - imagenet_mean_tensor) / imagenet_std_tensor

    return imagenet_norm


class NormalizationConverter(nn.Module):
    """
    Module wrapper for normalization conversion.

    Useful when you want to include normalization conversion as part of
    a model's forward pass.

    Args:
        source: Source normalization ('imagenet' or 'clip')
        target: Target normalization ('imagenet' or 'clip')

    Example:
        >>> converter = NormalizationConverter(source='imagenet', target='clip')
        >>> imagenet_images = torch.randn(32, 3, 224, 224)  # ImageNet normalized
        >>> clip_images = converter(imagenet_images)  # CLIP normalized
    """

    def __init__(self, source: str = 'imagenet', target: str = 'clip'):
        super().__init__()
        self.source = source.lower()
        self.target = target.lower()

        # Register normalization constants as buffers
        if source == 'imagenet':
            self.register_buffer('source_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('source_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        elif source == 'clip':
            self.register_buffer('source_mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer('source_std', torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        else:
            raise ValueError(f"Unknown source normalization: {source}")

        if target == 'imagenet':
            self.register_buffer('target_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('target_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        elif target == 'clip':
            self.register_buffer('target_mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer('target_std', torch.tensor(CLIP_STD).view(1, 3, 1, 1))
        else:
            raise ValueError(f"Unknown target normalization: {target}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert normalization from source to target.

        Args:
            tensor: (B, 3, H, W) - Images normalized with source scheme

        Returns:
            converted: (B, 3, H, W) - Images normalized with target scheme
        """
        # Denormalize from source
        denorm = tensor * self.source_std + self.source_mean

        # Normalize to target
        converted = (denorm - self.target_mean) / self.target_std

        return converted

    def __repr__(self):
        return f"NormalizationConverter(source={self.source}, target={self.target})"
