"""
PiPNet-based image encoder that outputs CLIP-compatible embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_features import ResNetFeatures


# ---------------------------------------------------------
# PiPNet Prototype Layer
# ---------------------------------------------------------
class PiPPrototypeLayer(nn.Module):
    """
    PiPNet prototype presence layer.

    Each prototype is a 1x1 convolution filter producing a
    spatial presence probability map.
    """

    def __init__(self, num_prototypes, prototype_dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim

        # 1x1 conv acts as prototype detectors
        self.conv = nn.Conv2d(
            in_channels=prototype_dim,
            out_channels=num_prototypes,
            kernel_size=1,
            bias=True
        )

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W)

        Returns:
            presence_maps: (B, M, H, W) in [0, 1]
        """
        logits = self.conv(features)
        presence_maps = torch.sigmoid(logits)
        return presence_maps

    def get_prototype_vectors(self):
        """
        Return prototype vectors (M, C)
        """
        return self.conv.weight.squeeze(-1).squeeze(-1)

    def set_prototype_vectors(self, new_prototypes):
        """
        Update prototype vectors (M, C)
        """
        with torch.no_grad():
            self.conv.weight.copy_(new_prototypes.unsqueeze(-1).unsqueeze(-1))


# ---------------------------------------------------------
# Pooling (shared with ProtoPNet-style API)
# ---------------------------------------------------------
class PiPPooling(nn.Module):
    """
    Pool prototype presence maps into a single score per prototype.
    """

    def __init__(self, mode='max'):
        super().__init__()
        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, presence_maps):
        """
        Args:
            presence_maps: (B, M, H, W)

        Returns:
            pooled: (B, M)
        """
        if self.mode == 'max':
            return presence_maps.amax(dim=(2, 3))
        else:
            return presence_maps.mean(dim=(2, 3))


# ---------------------------------------------------------
# PiPNet Encoder
# ---------------------------------------------------------
class PiPNetEncoder(nn.Module):
    """
    PiPNet image encoder for CLIP integration.

    Architecture:
        Image → CNN Backbone → PiP Prototype Layer → Pooling →
        Projection → L2-normalized embedding

    Args:
        num_prototypes: Number of prototypes
        backbone_arch: ResNet architecture
        embedding_dim: Output embedding dimension (CLIP = 512)
        pooling_mode: 'max' or 'avg'
        pretrained_backbone: Whether to use pretrained weights
        dropout_rate: Dropout probability in projection head
    """

    def __init__(
        self,
        num_prototypes=200,
        backbone_arch='resnet50',
        embedding_dim=512,
        pooling_mode='max',
        pretrained_backbone=True,
        dropout_rate=0.5
    ):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # 1. CNN Backbone
        self.backbone = ResNetFeatures(
            arch=backbone_arch,
            pretrained=pretrained_backbone,
            layer_name='layer3'
        )
        self.feature_dim = self.backbone.feature_dim

        # 2. PiP Prototype Layer
        self.prototype_layer = PiPPrototypeLayer(
            num_prototypes=num_prototypes,
            prototype_dim=self.feature_dim
        )

        # 3. Pooling
        self.pooling = PiPPooling(mode=pooling_mode)

        # 4. Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(num_prototypes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._initialize_projection()

    def _initialize_projection(self):
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, return_presence=False):
        """
        Args:
            images: (B, 3, 224, 224)
            return_presence: Whether to return spatial presence maps

        Returns:
            embeddings: (B, embedding_dim), L2-normalized
            presence_maps: (optional) (B, M, H, W)
        """
        # 1. Backbone features
        features = self.backbone(images)

        # 2. Prototype presence maps
        presence_maps = self.prototype_layer(features)

        # 3. Pool
        pooled = self.pooling(presence_maps)

        # 4. Projection
        embeddings = self.projection_head(pooled)

        # 5. Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if return_presence:
            return embeddings, presence_maps
        return embeddings

    def get_prototypes(self):
        return self.prototype_layer.get_prototype_vectors()

    def set_prototypes(self, new_prototypes):
        self.prototype_layer.set_prototype_vectors(new_prototypes)
