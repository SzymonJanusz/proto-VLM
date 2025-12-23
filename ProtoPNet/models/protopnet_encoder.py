"""
ProtoPNet-based image encoder that outputs CLIP-compatible embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_features import ResNetFeatures
from .prototype_layer import PrototypeLayer, WeightedPooling


class ProtoPNetEncoder(nn.Module):
    """
    ProtoPNet image encoder for CLIP integration.

    Architecture:
        Image -> CNN Backbone -> Prototype Layer -> Weighted Pooling -> Projection -> Embedding

    Args:
        num_prototypes: Number of prototypes to learn
        backbone_arch: ResNet architecture ('resnet50', 'resnet34')
        embedding_dim: Output embedding dimension (512 for CLIP)
        pooling_mode: 'max' or 'attention'
        pretrained_backbone: Whether to use pretrained weights for backbone
        dropout_rate: Dropout probability in projection head (default: 0.5)
        projection_hidden_dim: Hidden dimension for projection head (default: 512)
    """

    def __init__(
        self,
        num_prototypes=200,
        backbone_arch='resnet50',
        embedding_dim=512,
        pooling_mode='max',
        pretrained_backbone=True,
        dropout_rate=0.5,
        projection_hidden_dim=512
    ):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # 1. CNN Backbone (feature extractor)
        self.backbone = ResNetFeatures(
            arch=backbone_arch,
            pretrained=pretrained_backbone,
            layer_name='layer3'  # Use layer3 for better spatial resolution
        )
        self.feature_dim = self.backbone.feature_dim

        # 2. Prototype Layer
        self.prototype_layer = PrototypeLayer(
            num_prototypes=num_prototypes,
            prototype_dim=self.feature_dim,
            init_mode='random'
        )

        # 3. Weighted Pooling
        self.pooling = WeightedPooling(pooling_mode=pooling_mode)

        # 4. Projection Head (prototypes -> CLIP embedding space)
        # Enhanced with batch normalization and configurable dropout
        # Reduced capacity to prevent overfitting
        self.projection_head = nn.Sequential(
            nn.Linear(num_prototypes, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(projection_hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Initialize projection head
        self._initialize_projection()

    def _initialize_projection(self):
        """Initialize projection head weights"""
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, return_similarities=False):
        """
        Forward pass.

        Args:
            images: Input images (B, 3, 224, 224)
            return_similarities: Whether to return similarity maps for visualization

        Returns:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            similarities: (optional) Spatial similarity maps (B, M, H, W)
        """
        # 1. Extract features
        features = self.backbone(images)  # (B, feature_dim, H, W)

        # 2. Compute prototype similarities
        similarities = self.prototype_layer(features)  # (B, M, H, W)

        # 3. Pool to single similarity per prototype
        pooled_similarities = self.pooling(similarities)  # (B, M)

        # 4. Project to embedding space
        embeddings = self.projection_head(pooled_similarities)  # (B, embedding_dim)

        # 5. L2 normalize (critical for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if return_similarities:
            return embeddings, similarities
        else:
            return embeddings

    def get_prototypes(self):
        """Get prototype vectors for projection/visualization"""
        return self.prototype_layer.get_prototype_vectors()

    def set_prototypes(self, new_prototypes):
        """Update prototype vectors (used in projection step)"""
        self.prototype_layer.set_prototype_vectors(new_prototypes)
