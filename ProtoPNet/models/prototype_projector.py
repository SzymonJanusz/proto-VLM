"""
Prototype Projector: Maps spatial prototype similarities to CLIP embedding space.

This module provides trainable projection heads that map sparse prototype activations
(spatial similarity maps from ProtoPNet) to CLIP-compatible embeddings (512-dim).

Multiple projection techniques are supported:
- Spatial CNN: Convolutional layers that respect spatial structure (IMPLEMENTED)
- Direct Flatten: MLP on flattened spatial features (PLACEHOLDER)
- Attention-Weighted: Learnable attention over prototypes (PLACEHOLDER)
- Hierarchical Pooling: Multi-statistic pooling + MLP (PLACEHOLDER)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeProjector(nn.Module):
    """
    Base class for prototype projectors.

    Maps sparse spatial prototype similarities (B, M, H, W) to CLIP embeddings (B, 512).
    """

    def __init__(self, num_prototypes: int = 200, embedding_dim: int = 512):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

    def forward(self, spatial_similarities: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: spatial similarities → CLIP embeddings.

        Args:
            spatial_similarities: (B, M, H, W) - Spatial prototype activations

        Returns:
            embeddings: (B, embedding_dim) - L2-normalized CLIP embeddings
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'num_prototypes': self.num_prototypes,
            'embedding_dim': self.embedding_dim,
        }

    def save(self, path: str):
        """
        Save projector checkpoint.

        Args:
            path: Path to save checkpoint (.pt file)
        """
        torch.save({
            'config': self.get_config(),
            'state_dict': self.state_dict(),
            'projector_type': self.__class__.__name__
        }, path)
        print(f"Saved {self.__class__.__name__} to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """
        Load projector from checkpoint.

        Args:
            path: Path to checkpoint (.pt file)
            device: Device to load model onto

        Returns:
            PrototypeProjector instance with loaded weights
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Get the correct class
        projector_type = checkpoint.get('projector_type', cls.__name__)
        if projector_type == 'SpatialCNNProjector':
            projector_cls = SpatialCNNProjector
        elif projector_type == 'DirectFlattenProjector':
            projector_cls = DirectFlattenProjector
        elif projector_type == 'AttentionWeightedProjector':
            projector_cls = AttentionWeightedProjector
        elif projector_type == 'HierarchicalPoolingProjector':
            projector_cls = HierarchicalPoolingProjector
        else:
            projector_cls = cls

        # Create instance from config
        config = checkpoint['config']
        projector = projector_cls(**config)

        # Load weights
        projector.load_state_dict(checkpoint['state_dict'])
        projector.to(device)

        print(f"Loaded {projector_type} from: {path}")
        return projector


class SpatialCNNProjector(PrototypeProjector):
    """
    Spatial CNN Projector (Technique 2) - IMPLEMENTED

    Maps spatial prototype similarities to CLIP embeddings using convolutional layers.
    Respects spatial structure and learns local relationships between prototype activations.

    Architecture:
        Input: (B, 200, 14, 14) - Spatial prototype similarities
        → Conv2D(200 → 512, kernel=3, padding=1)
        → BatchNorm2d(512)
        → ReLU
        → Dropout2d(0.3)
        → Conv2D(512 → 512, kernel=3, padding=1)
        → BatchNorm2d(512)
        → ReLU
        → Global Average Pooling: (B, 512, 14, 14) → (B, 512)
        → L2 Normalize
        Output: (B, 512) - CLIP embeddings

    Args:
        num_prototypes: Number of prototypes (default: 200)
        embedding_dim: CLIP embedding dimension (default: 512)
        hidden_channels: Number of channels in intermediate conv layer (default: 512)
        dropout: Dropout probability (default: 0.3)

    Example:
        >>> projector = SpatialCNNProjector()
        >>> spatial_sims = torch.randn(32, 200, 14, 14)  # Batch of 32 images
        >>> clip_embeddings = projector(spatial_sims)  # (32, 512)
        >>> assert clip_embeddings.shape == (32, 512)
        >>> assert torch.allclose(torch.norm(clip_embeddings, dim=1), torch.ones(32), atol=1e-5)
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        embedding_dim: int = 512,
        hidden_channels: int = 512,
        dropout: float = 0.3
    ):
        super().__init__(num_prototypes, embedding_dim)
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Two-layer CNN with batch normalization and dropout
        self.projection = nn.Sequential(
            # First conv layer: num_prototypes → hidden_channels
            nn.Conv2d(num_prototypes, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),  # Spatial dropout

            # Second conv layer: hidden_channels → embedding_dim
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )

        # Global average pooling (will be applied in forward)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize conv weights with Xavier normal."""
        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_similarities: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: spatial similarities → CLIP embeddings.

        Args:
            spatial_similarities: (B, M, H, W) - Spatial prototype activations
                Expected: (B, 200, 14, 14)

        Returns:
            embeddings: (B, embedding_dim) - L2-normalized CLIP embeddings

        Note:
            L2 normalization is critical for cosine similarity in CLIP space.
        """
        # Input shape check
        B, M, H, W = spatial_similarities.shape
        assert M == self.num_prototypes, f"Expected {self.num_prototypes} prototypes, got {M}"

        # Apply convolutional layers
        features = self.projection(spatial_similarities)  # (B, embedding_dim, H, W)

        # Global average pooling
        pooled = self.global_pool(features)  # (B, embedding_dim, 1, 1)
        pooled = pooled.view(B, self.embedding_dim)  # (B, embedding_dim)

        # L2 normalize (critical for CLIP cosine similarity)
        embeddings = F.normalize(pooled, p=2, dim=-1)

        return embeddings

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        config = super().get_config()
        config.update({
            'hidden_channels': self.hidden_channels,
            'dropout': self.dropout,
        })
        return config


# =============================================================================
# PLACEHOLDER IMPLEMENTATIONS (Techniques 1, 3, 4)
# =============================================================================
# These classes define the architecture but raise NotImplementedError
# They serve as documentation and templates for future implementation
# =============================================================================


class DirectFlattenProjector(PrototypeProjector):
    """
    Technique 1: Direct Flatten + MLP Projection (PLACEHOLDER)

    Architecture:
        Input: (B, 200, 14, 14) - Spatial prototype similarities
        → Flatten: (B, 200*14*14 = 39,200)
        → Linear(39,200 → 2,048)
        → BatchNorm1d(2,048)
        → ReLU
        → Dropout(0.5)
        → Linear(2,048 → 512)
        → BatchNorm1d(512)
        → L2 Normalize
        Output: (B, 512) - CLIP embeddings

    Pros:
        - Straightforward MLP projection
        - Preserves all spatial information
        - End-to-end learnable

    Cons:
        - Very high input dimensionality (39,200)
        - May be hard to train
        - Loses explicit spatial structure
        - High memory usage

    Args:
        num_prototypes: Number of prototypes (default: 200)
        embedding_dim: CLIP embedding dimension (default: 512)
        spatial_size: Spatial dimension (H=W, default: 14)
        hidden_dim: Hidden layer dimension (default: 2048)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        embedding_dim: int = 512,
        spatial_size: int = 14,
        hidden_dim: int = 2048,
        dropout: float = 0.5
    ):
        super().__init__(num_prototypes, embedding_dim)
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Calculate input dimension
        self.input_dim = num_prototypes * spatial_size * spatial_size  # 200 * 14 * 14 = 39,200

        raise NotImplementedError(
            "Technique 1 (DirectFlattenProjector) is not yet implemented. "
            "This is a placeholder for future development. "
            "To implement: create MLP layers as shown in architecture above, "
            "initialize weights, and implement forward() method."
        )


class AttentionWeightedProjector(PrototypeProjector):
    """
    Technique 3: Attention-Weighted Prototype Embeddings (PLACEHOLDER)

    Architecture:
        Input: (B, 200, 14, 14) - Spatial prototype similarities
        → Max Pool per prototype: (B, 200) [max activation value]
        → Attention Module:
            - Linear(200 → 200)
            - Tanh
            - Linear(200 → 200)
            - Softmax → (B, 200) [attention weights]
        → Prototype Embeddings: (200, 512) [learnable parameter]
        → Weighted Sum: Σ attention[i] * prototype_embeddings[i]
        → L2 Normalize
        Output: (B, 512) - CLIP embeddings

    Mechanism:
        Each of the 200 prototypes has its own 512-dim embedding vector.
        The attention module learns to weight these based on activation patterns.
        Final embedding is a weighted combination of prototype embeddings.

    Pros:
        - Each prototype has interpretable embedding
        - Attention weights show which prototypes contribute
        - Compositional: final embedding is combination of prototype meanings
        - Fewer parameters than fully-connected
        - Highly interpretable

    Cons:
        - Loses spatial information (uses max pool)
        - Requires good initialization of prototype embeddings
        - Attention mechanism adds complexity

    Args:
        num_prototypes: Number of prototypes (default: 200)
        embedding_dim: CLIP embedding dimension (default: 512)
        attention_hidden_dim: Hidden dimension in attention module (default: 200)
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        embedding_dim: int = 512,
        attention_hidden_dim: int = 200
    ):
        super().__init__(num_prototypes, embedding_dim)
        self.attention_hidden_dim = attention_hidden_dim

        raise NotImplementedError(
            "Technique 3 (AttentionWeightedProjector) is not yet implemented. "
            "This is a placeholder for future development. "
            "To implement: "
            "1. Create learnable prototype embeddings (nn.Parameter): (M, 512) "
            "2. Create attention MLP: Linear(M → hidden) → Tanh → Linear(hidden → M) "
            "3. In forward(): max pool spatial dims, compute attention weights (softmax), "
            "   weighted sum of prototype embeddings, L2 normalize."
        )


class HierarchicalPoolingProjector(PrototypeProjector):
    """
    Technique 4: Hierarchical Pooling + MLP (PLACEHOLDER)

    Architecture:
        Input: (B, 200, 14, 14) - Spatial prototype similarities
        → Parallel Pooling:
            - Max Pool: max over (H, W) → (B, 200) [strongest activation]
            - Mean Pool: mean over (H, W) → (B, 200) [average activation]
            - Std Pool: std over (H, W) → (B, 200) [activation variance]
        → Concatenate: (B, 600)
        → MLP:
            - Linear(600 → 1024)
            - BatchNorm1d(1024)
            - ReLU
            - Dropout(0.3)
            - Linear(1024 → 512)
            - BatchNorm1d(512)
        → L2 Normalize
        Output: (B, 512) - CLIP embeddings

    Mechanism:
        Captures multiple statistics (max, mean, variance) from spatial activations.
        More informative than single pooling strategy.

    Pros:
        - Captures multiple statistics from spatial structure
        - More informative than single pooling
        - Moderate dimensionality (600)
        - Inspired by biological vision (multiple pooling pathways)

    Cons:
        - Loses fine-grained spatial structure
        - Three pooling types may be correlated (redundant)
        - More parameters than simple max pooling

    Args:
        num_prototypes: Number of prototypes (default: 200)
        embedding_dim: CLIP embedding dimension (default: 512)
        hidden_dim: Hidden layer dimension (default: 1024)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.3
    ):
        super().__init__(num_prototypes, embedding_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Concatenated features: max + mean + std = 3 * num_prototypes
        self.concat_dim = 3 * num_prototypes

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MLP weights with Xavier/He initialization."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spatial_similarities: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical pooling projector.

        Args:
            spatial_similarities: (B, M, H, W) - Spatial prototype activations

        Returns:
            embeddings: (B, 512) - L2-normalized CLIP embeddings
        """
        B, M, H, W = spatial_similarities.shape

        # Flatten spatial dimensions for pooling
        spatial_flat = spatial_similarities.view(B, M, H * W)  # (B, M, H*W)

        # Compute three types of pooling statistics
        max_pool = spatial_flat.max(dim=2)[0]  # (B, M) - strongest activation
        mean_pool = spatial_flat.mean(dim=2)    # (B, M) - average activation
        std_pool = spatial_flat.std(dim=2)      # (B, M) - activation variance

        # Concatenate all statistics
        features = torch.cat([max_pool, mean_pool, std_pool], dim=1)  # (B, 3*M)

        # Project through MLP
        embeddings = self.mlp(features)  # (B, 512)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# =============================================================================
# Factory function for creating projectors
# =============================================================================


def create_projector(
    projector_type: str = 'spatial_cnn',
    num_prototypes: int = 200,
    embedding_dim: int = 512,
    **kwargs
) -> PrototypeProjector:
    """
    Factory function to create prototype projectors.

    Args:
        projector_type: Type of projector to create
            - 'spatial_cnn': Spatial CNN (Technique 2) - IMPLEMENTED
            - 'direct_flatten': Direct Flatten + MLP (Technique 1) - PLACEHOLDER
            - 'attention': Attention-Weighted (Technique 3) - PLACEHOLDER
            - 'hierarchical': Hierarchical Pooling (Technique 4) - PLACEHOLDER
        num_prototypes: Number of prototypes (default: 200)
        embedding_dim: CLIP embedding dimension (default: 512)
        **kwargs: Additional arguments for specific projector types

    Returns:
        PrototypeProjector instance

    Examples:
        >>> # Spatial CNN (default, working)
        >>> projector = create_projector('spatial_cnn')

        >>> # Spatial CNN with custom parameters
        >>> projector = create_projector(
        ...     'spatial_cnn',
        ...     hidden_channels=1024,
        ...     dropout=0.5
        ... )

        >>> # Direct Flatten (will raise NotImplementedError)
        >>> projector = create_projector('direct_flatten')  # Not yet implemented
    """
    projector_type = projector_type.lower()

    if projector_type == 'spatial_cnn':
        return SpatialCNNProjector(num_prototypes, embedding_dim, **kwargs)

    elif projector_type == 'direct_flatten':
        return DirectFlattenProjector(num_prototypes, embedding_dim, **kwargs)

    elif projector_type == 'attention':
        return AttentionWeightedProjector(num_prototypes, embedding_dim, **kwargs)

    elif projector_type == 'hierarchical':
        return HierarchicalPoolingProjector(num_prototypes, embedding_dim, **kwargs)

    else:
        raise ValueError(
            f"Unknown projector type: {projector_type}. "
            f"Supported: 'spatial_cnn', 'direct_flatten', 'attention', 'hierarchical'"
        )
