"""
Class-Level Interpretation using CNN Features

Extracts and aggregates abstract CNN features (not prototype similarities)
from images for class-level interpretation with text decoders.

Key distinction: This uses CNN features (1024-dim) as abstract visual
representation, not prototype similarity scores (200-dim classification head).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple
import numpy as np
import clip


class CLIPViTFeatureExtractor:
    """
    Extracts CLIP ViT embeddings directly from images.

    Uses the official CLIP ViT-B/32 model to extract image embeddings.
    These embeddings are directly compatible with ClipCap and other CLIP-based
    text decoders.

    Args:
        device: Computation device
        clip_model: CLIP model name (default: 'ViT-B/32')

    Example:
        >>> extractor = CLIPViTFeatureExtractor(device='cuda')
        >>> image = torch.randn(1, 3, 224, 224).cuda()
        >>> features = extractor.extract_single_image_features(image)
        >>> assert features.shape == (512,)
    """

    def __init__(self, device='cuda', clip_model='ViT-B/32'):
        self.device = device
        print(f"Loading CLIP {clip_model} model...")
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model.eval()
        print(f"  CLIP model loaded on {device}")

    def extract_single_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP ViT features from a single image.

        Args:
            image: (3, 224, 224) or (1, 3, 224, 224) - Input image (preprocessed)

        Returns:
            features: (512,) - CLIP image embedding
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, 3, 224, 224)

        image = image.to(self.device)

        with torch.no_grad():
            # Extract CLIP image features
            image_features = self.model.encode_image(image)  # (1, 512)

            # L2 normalize (CLIP does this internally, but ensure it)
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Convert to float32 (CLIP may return float16)
            image_features = image_features.float()

            image_features = image_features.squeeze(0)  # (512,)

        return image_features

    def extract_batch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP ViT features from a batch of images.

        Args:
            images: (B, 3, 224, 224) - Batch of images

        Returns:
            features: (B, 512) - CLIP image embeddings
        """
        images = images.to(self.device)

        with torch.no_grad():
            # Extract features
            image_features = self.model.encode_image(images)  # (B, 512)

            # L2 normalize
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Convert to float32 (CLIP may return float16)
            image_features = image_features.float()

        return image_features

    def extract_class_features(
        self,
        dataloader,
        max_images: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract CLIP ViT features from all images in a dataloader.

        Args:
            dataloader: DataLoader providing images
            max_images: Optional limit on number of images to process

        Returns:
            features: (N, 512) - CLIP features for N images
            image_paths: List of image paths (if available)
        """
        all_features = []
        image_paths = []
        count = 0

        for batch in dataloader:
            if max_images and count >= max_images:
                break

            # Get images
            if isinstance(batch, dict):
                images = batch['image']
                paths = batch.get('image_path', [None] * len(images))
            else:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                paths = [None] * len(images)

            # Extract features
            features = self.extract_batch_features(images)
            all_features.append(features.cpu())
            image_paths.extend(paths)

            count += len(images)

            # Stop if we've reached the limit
            if max_images and count >= max_images:
                # Trim to exact limit
                excess = count - max_images
                if excess > 0:
                    all_features[-1] = all_features[-1][:-excess]
                    image_paths = image_paths[:-excess]
                break

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)

        return all_features, image_paths


class CNNFeatureExtractor:
    """
    Extracts CNN features from images using a frozen ResNet backbone.

    Uses global average pooling to convert spatial features (H, W) into
    a single feature vector suitable for aggregation and projection.

    Args:
        backbone: Frozen ResNet backbone (from trained ProtoPNet)
        device: Computation device

    Example:
        >>> extractor = CNNFeatureExtractor(model.image_encoder.backbone, device='cuda')
        >>> image = torch.randn(1, 3, 224, 224).cuda()
        >>> features = extractor.extract_single_image_features(image)
        >>> assert features.shape == (1024,)
    """

    def __init__(self, backbone, device='cuda'):
        self.backbone = backbone
        self.device = device
        self.backbone.eval()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_single_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from a single image.

        Args:
            image: (3, 224, 224) or (1, 3, 224, 224) - Input image

        Returns:
            features: (1024,) - Pooled CNN features
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, 3, 224, 224)

        image = image.to(self.device)

        with torch.no_grad():
            # Extract features from backbone
            cnn_features = self.backbone(image)  # (1, 1024, 14, 14)

            # Global average pooling
            pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1))  # (1, 1024, 1, 1)
            pooled = pooled.view(-1)  # (1024,)

        return pooled

    def extract_batch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from a batch of images.

        Args:
            images: (B, 3, 224, 224) - Batch of images

        Returns:
            features: (B, 1024) - Pooled CNN features
        """
        images = images.to(self.device)

        with torch.no_grad():
            # Extract features
            cnn_features = self.backbone(images)  # (B, 1024, 14, 14)

            # Global average pooling
            batch_size = cnn_features.size(0)
            pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1))  # (B, 1024, 1, 1)
            pooled = pooled.view(batch_size, -1)  # (B, 1024)

        return pooled

    def extract_class_features(
        self,
        dataloader,
        max_images: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract CNN features from all images in a dataloader.

        Args:
            dataloader: DataLoader providing images
            max_images: Optional limit on number of images to process

        Returns:
            features: (N, 1024) - CNN features for N images
            image_paths: List of image paths (if available)
        """
        all_features = []
        image_paths = []
        count = 0

        for batch in dataloader:
            if max_images and count >= max_images:
                break

            # Get images
            if isinstance(batch, dict):
                images = batch['image']
                paths = batch.get('image_path', [None] * len(images))
            else:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                paths = [None] * len(images)

            # Extract features
            features = self.extract_batch_features(images)
            all_features.append(features.cpu())
            image_paths.extend(paths)

            count += len(images)

            # Stop if we've reached the limit
            if max_images and count >= max_images:
                # Trim to exact limit
                excess = count - max_images
                if excess > 0:
                    all_features[-1] = all_features[-1][:-excess]
                    image_paths = image_paths[:-excess]
                break

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)

        return all_features, image_paths


class FeatureAggregator:
    """
    Aggregates CNN features across multiple images.

    Supports multiple aggregation strategies for combining features
    from a class of images into a single representative vector.

    Methods:
        - mean: Simple average (treats all images equally)
        - weighted_mean: Weighted by similarity/confidence scores
        - median: Robust to outliers
        - max: Element-wise maximum

    Example:
        >>> aggregator = FeatureAggregator()
        >>> features = torch.randn(50, 1024)  # 50 images
        >>> aggregated = aggregator.aggregate(features, method='mean')
        >>> assert aggregated.shape == (1024,)
    """

    @staticmethod
    def aggregate(
        features: torch.Tensor,
        method: str = 'mean',
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate features across multiple images.

        Args:
            features: (N, D) - Features from N images
            method: Aggregation method ('mean', 'weighted_mean', 'median', 'max')
            weights: Optional (N,) - Weights for weighted_mean

        Returns:
            aggregated: (D,) - Single aggregated feature vector

        Raises:
            ValueError: If unknown aggregation method
        """
        if method == 'mean':
            return features.mean(dim=0)

        elif method == 'weighted_mean':
            if weights is None:
                # If no weights provided, use uniform weights (same as mean)
                return features.mean(dim=0)

            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.unsqueeze(1)  # (N, 1)

            # Weighted average
            return (features * weights).sum(dim=0)

        elif method == 'median':
            return features.median(dim=0)[0]

        elif method == 'max':
            return features.max(dim=0)[0]

        else:
            raise ValueError(
                f"Unknown aggregation method: {method}. "
                f"Supported: 'mean', 'weighted_mean', 'median', 'max'"
            )

    @staticmethod
    def compute_prototype_weights(
        features: torch.Tensor,
        prototype_layer,
        pooling_layer,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Compute weights based on prototype activation strengths.

        This can be used to weight features by how strongly they activate
        the prototypes, emphasizing images with clearer patterns.

        Args:
            features: (N, 1024, 14, 14) - Unpooled CNN features
            prototype_layer: Prototype layer for computing similarities
            pooling_layer: Pooling layer for aggregating similarities
            device: Computation device

        Returns:
            weights: (N,) - Weight for each image (max prototype similarity)
        """
        with torch.no_grad():
            features = features.to(device)

            # Compute prototype similarities
            similarities = prototype_layer(features)  # (N, num_prototypes, 14, 14)

            # Pool similarities
            pooled_sims = pooling_layer(similarities)  # (N, num_prototypes)

            # Use max similarity as weight (confidence measure)
            weights = pooled_sims.max(dim=1)[0]  # (N,)

        return weights.cpu()


class ClassFilteredDataset(Dataset):
    """
    Wrapper that filters a dataset to specific class(es).

    Args:
        base_dataset: Base ImageNet dataset
        target_class_ids: Set or list of target class IDs to keep

    Example:
        >>> dataset = ImageNetWithCaptions(root='imagenet_tiny', split='val')
        >>> # Filter to corn (class 950), bell pepper (941), reel (650)
        >>> filtered = ClassFilteredDataset(dataset, [950, 941, 650])
        >>> print(len(filtered))  # Number of images from these 3 classes
    """

    def __init__(self, base_dataset, target_class_ids):
        self.base_dataset = base_dataset
        self.target_class_ids = set(target_class_ids)

        # Build index of matching samples
        self.indices = []

        # Check if dataset has 'samples' attribute (ImageFolder style)
        if hasattr(base_dataset, 'samples'):
            for i, (path, class_id) in enumerate(base_dataset.samples):
                if class_id in self.target_class_ids:
                    self.indices.append(i)
        # Otherwise, iterate through dataset
        else:
            for i in range(len(base_dataset)):
                try:
                    sample = base_dataset[i]
                    # Extract class ID from sample
                    if isinstance(sample, dict):
                        class_id = sample.get('class_id', sample.get('label'))
                    elif isinstance(sample, (tuple, list)):
                        class_id = sample[1]  # Assume (image, label) format
                    else:
                        continue

                    if class_id in self.target_class_ids:
                        self.indices.append(i)
                except:
                    continue

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def get_class_distribution(self):
        """Get distribution of classes in filtered dataset."""
        distribution = {}

        for idx in self.indices:
            sample = self.base_dataset[idx]

            # Extract class ID
            if isinstance(sample, dict):
                class_id = sample.get('class_id', sample.get('label'))
            elif isinstance(sample, (tuple, list)):
                class_id = sample[1]
            else:
                continue

            distribution[class_id] = distribution.get(class_id, 0) + 1

        return distribution
