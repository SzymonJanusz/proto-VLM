"""
CNN Feature Projector: Maps ResNet CNN features to CLIP embedding space.

This module provides a trainable projection head that maps abstract CNN features
(1024-dim from ResNet50 layer3) to CLIP-compatible embeddings (512-dim).

Unlike the prototype similarity-based projection (which uses classification scores),
this projector works with rich abstract visual features suitable for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureProjector(nn.Module):
    """
    Projects ResNet CNN features to CLIP embedding space.

    Architecture:
        CNN Features (1024-dim)
        → Linear(1024 → hidden_dim)
        → BatchNorm → ReLU → Dropout
        → Linear(hidden_dim → 512)
        → BatchNorm → L2-Normalize
        → CLIP Embedding (512-dim)

    Args:
        feature_dim: Dimension of input CNN features (default: 1024 for ResNet50 layer3)
        embedding_dim: Dimension of output CLIP embeddings (default: 512)
        hidden_dim: Dimension of hidden layer (default: 512)
        dropout: Dropout probability (default: 0.3)

    Input:
        cnn_features: (B, feature_dim) - Pooled CNN features

    Output:
        embeddings: (B, embedding_dim) - L2-normalized CLIP embeddings

    Example:
        >>> projector = CNNFeatureProjector()
        >>> cnn_features = torch.randn(32, 1024)  # Batch of 32 images
        >>> clip_embeddings = projector(cnn_features)  # (32, 512)
        >>> assert clip_embeddings.shape == (32, 512)
        >>> assert torch.allclose(torch.norm(clip_embeddings, dim=1), torch.ones(32), atol=1e-5)
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP with batch normalization and dropout
        self.projection = nn.Sequential(
            # First layer: feature_dim → hidden_dim
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Second layer: hidden_dim → embedding_dim
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projection weights with Xavier normal."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN features → CLIP embeddings.

        Args:
            cnn_features: (B, feature_dim) - Pooled CNN features from ResNet

        Returns:
            embeddings: (B, embedding_dim) - L2-normalized CLIP embeddings

        Note:
            L2 normalization is critical for cosine similarity in CLIP space.
        """
        # Project features
        embeddings = self.projection(cnn_features)  # (B, embedding_dim)

        # L2 normalize (critical for CLIP cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'feature_dim': self.feature_dim,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config: dict, state_dict: dict = None):
        """
        Create projector from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config()
            state_dict: Optional state dictionary to load weights

        Returns:
            CNNFeatureProjector instance with loaded weights
        """
        projector = cls(**config)
        if state_dict is not None:
            projector.load_state_dict(state_dict)
        return projector

    def save(self, path: str):
        """
        Save projector checkpoint.

        Args:
            path: Path to save checkpoint (.pt file)
        """
        torch.save({
            'config': self.get_config(),
            'state_dict': self.state_dict()
        }, path)
        print(f"Saved CNNFeatureProjector to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """
        Load projector from checkpoint.

        Args:
            path: Path to checkpoint (.pt file)
            device: Device to load model onto

        Returns:
            CNNFeatureProjector instance with loaded weights
        """
        checkpoint = torch.load(path, map_location=device)
        projector = cls.from_config(checkpoint['config'], checkpoint['state_dict'])
        projector.to(device)
        print(f"Loaded CNNFeatureProjector from: {path}")
        return projector
