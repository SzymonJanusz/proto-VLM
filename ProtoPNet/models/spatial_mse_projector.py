"""
Spatial MSE Projector: Maps spatial prototype similarities to CLIP spatial features using MSE loss.

This projector preserves 14×14 spatial structure throughout the projection and is trained
with MSE loss to match CLIP ResNet-50 layer3 spatial features (1024-dim per location).

Key differences from existing projectors:
- Preserves spatial resolution (no pooling until after projection)
- Outputs 1024-dim per spatial location (vs 512-dim global)
- Trained with per-location MSE loss (vs contrastive loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMSEProjector(nn.Module):
    """
    Spatial MSE Projector - Preserves 14×14 structure for MSE training.

    Maps spatial prototype similarities to CLIP ResNet-50 layer3 spatial features
    using 3-layer CNN without any pooling operations.

    Architecture:
        Input: (B, 200, 14, 14) - Spatial prototype similarities
        → Conv2D(200 → 512, kernel=3, padding=1)
        → BatchNorm2d(512)
        → ReLU
        → Dropout2d(0.3)
        → Conv2D(512 → 1024, kernel=3, padding=1)
        → BatchNorm2d(1024)
        → ReLU
        → Dropout2d(0.3)
        → Conv2D(1024 → 1024, kernel=3, padding=1)
        → BatchNorm2d(1024)
        Output: (B, 1024, 14, 14) - Spatial projected features

    Training:
        - Trained with per-location normalized MSE loss
        - Target: CLIP ResNet-50 layer3 features (1024-dim, 14×14)
        - Optional: Sparse prototype selection via top_k before projection

    Usage:
        >>> projector = SpatialMSEProjector()
        >>> spatial_sims = torch.randn(32, 200, 14, 14)  # Prototype similarities
        >>> spatial_features = projector(spatial_sims)  # (32, 1024, 14, 14)
        >>> assert spatial_features.shape == (32, 1024, 14, 14)

        >>> # Can pool for global features if needed
        >>> global_features = F.adaptive_avg_pool2d(spatial_features, (1, 1))
        >>> global_features = global_features.squeeze()  # (32, 1024)

    Args:
        num_prototypes: Number of input prototypes (default: 200)
        output_dim: Output feature dimension per location (default: 1024 for CLIP layer3)
        hidden_channels: Number of channels in intermediate layers (default: 512)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        num_prototypes: int = 200,
        output_dim: int = 1024,
        hidden_channels: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # 3-layer CNN - NO POOLING (preserves spatial structure)
        self.projection = nn.Sequential(
            # Layer 1: 200 → 512
            nn.Conv2d(num_prototypes, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),

            # Layer 2: 512 → 1024
            nn.Conv2d(hidden_channels, output_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),

            # Layer 3: 1024 → 1024 (refinement)
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output_dim)
        )

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
        Forward pass: spatial similarities → spatial CLIP features.

        Args:
            spatial_similarities: (B, M, H, W) - Spatial prototype activations
                Expected: (B, 200, 14, 14)
                Can be sparse (if top_k selection applied beforehand)

        Returns:
            spatial_features: (B, output_dim, H, W) - Spatial projected features
                Default: (B, 1024, 14, 14)

        Note:
            - Spatial resolution is preserved (14×14 → 14×14)
            - No normalization applied here (done during loss computation)
            - For global features, pool output: F.adaptive_avg_pool2d(output, (1, 1))
        """
        # Input shape check
        B, M, H, W = spatial_similarities.shape
        assert M == self.num_prototypes, f"Expected {self.num_prototypes} prototypes, got {M}"
        assert H == W == 14, f"Expected 14×14 spatial resolution, got {H}×{W}"

        # Apply convolutional layers (preserves spatial dimensions)
        spatial_features = self.projection(spatial_similarities)  # (B, output_dim, H, W)

        return spatial_features

    def get_config(self) -> dict:
        """Get configuration dictionary for saving/loading."""
        return {
            'num_prototypes': self.num_prototypes,
            'output_dim': self.output_dim,
            'hidden_channels': self.hidden_channels,
            'dropout': self.dropout,
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
        print(f"Saved SpatialMSEProjector to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """
        Load projector from checkpoint.

        Args:
            path: Path to checkpoint (.pt file)
            device: Device to load model onto

        Returns:
            SpatialMSEProjector instance with loaded weights
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Verify projector type
        projector_type = checkpoint.get('projector_type', 'SpatialMSEProjector')
        if projector_type != 'SpatialMSEProjector':
            print(f"Warning: Loading checkpoint with projector_type={projector_type}")

        # Create instance from config
        config = checkpoint['config']
        projector = cls(**config)

        # Load weights
        projector.load_state_dict(checkpoint['state_dict'])
        projector.to(device)
        projector.eval()

        print(f"Loaded SpatialMSEProjector from: {path}")
        return projector

    def __repr__(self):
        return (f"SpatialMSEProjector("
                f"num_prototypes={self.num_prototypes}, "
                f"output_dim={self.output_dim}, "
                f"hidden_channels={self.hidden_channels}, "
                f"dropout={self.dropout})")
