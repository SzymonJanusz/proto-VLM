"""
Prototype layer that computes similarity between features and learned prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """
    Learnable prototype layer.

    Stores M prototype vectors and computes similarity between
    input feature patches and prototypes.

    Args:
        num_prototypes: Number of prototypes to learn (M)
        prototype_dim: Dimension of each prototype (should match feature channels)
        init_mode: How to initialize prototypes ('random', 'zeros')
    """

    def __init__(self, num_prototypes, prototype_dim, init_mode='random'):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim

        # Learnable prototype vectors
        self.prototype_vectors = nn.Parameter(
            torch.randn(num_prototypes, prototype_dim),
            requires_grad=True
        )

        if init_mode == 'zeros':
            nn.init.zeros_(self.prototype_vectors)
        elif init_mode == 'random':
            # Xavier initialization
            nn.init.xavier_normal_(self.prototype_vectors)

    def forward(self, features):
        """
        Compute similarity between features and prototypes.

        Args:
            features: Feature maps (B, C, H, W) where C = prototype_dim

        Returns:
            similarities: Similarity maps (B, M, H, W)
                         Using inverted L2 distance: -||f - p||^2
        """
        B, C, H, W = features.shape
        assert C == self.prototype_dim, \
            f"Feature dim {C} != prototype dim {self.prototype_dim}"

        # Reshape features: (B, C, H, W) -> (B, H*W, C)
        features_flat = features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Memory-efficient L2 distance computation using:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # This avoids creating a huge (B, H*W, M, C) intermediate tensor

        # Compute ||features||^2 for each patch: (B, H*W)
        features_norm_sq = torch.sum(features_flat ** 2, dim=2)  # (B, H*W)

        # Compute ||prototypes||^2 for each prototype: (M,)
        prototypes_norm_sq = torch.sum(self.prototype_vectors ** 2, dim=1)  # (M,)

        # Compute dot product: features @ prototypes.T -> (B, H*W, M)
        dot_product = torch.matmul(features_flat, self.prototype_vectors.t())  # (B, H*W, M)

        # L2 distance squared: ||a||^2 + ||b||^2 - 2<a, b>
        # Broadcasting: (B, H*W, 1) + (1, 1, M) - (B, H*W, M)
        distances = (
            features_norm_sq.unsqueeze(2) +  # (B, H*W, 1)
            prototypes_norm_sq.unsqueeze(0).unsqueeze(0) -  # (1, 1, M)
            2 * dot_product  # (B, H*W, M)
        )  # Result: (B, H*W, M)

        # Convert to similarity (inverted distance)
        # Note: negative distance so higher = more similar
        similarities = -distances  # (B, H*W, M)

        # Reshape back to spatial: (B, H*W, M) -> (B, M, H, W)
        similarities = similarities.permute(0, 2, 1).view(B, self.num_prototypes, H, W)

        return similarities

    def get_prototype_vectors(self):
        """Return prototype vectors for visualization/projection"""
        return self.prototype_vectors.detach()

    def set_prototype_vectors(self, new_prototypes):
        """Update prototype vectors (used during projection step)"""
        assert new_prototypes.shape == self.prototype_vectors.shape
        self.prototype_vectors.data = new_prototypes.data


class WeightedPooling(nn.Module):
    """
    Pool spatial prototype activations to single similarity score.

    Args:
        pooling_mode: 'max' (max pooling) or 'attention' (learnable attention)
    """

    def __init__(self, pooling_mode='max'):
        super().__init__()
        self.pooling_mode = pooling_mode

        if pooling_mode == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling_mode == 'attention':
            # Learnable attention per prototype
            # Will be applied per-prototype, so single channel input
            self.attention_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, similarities):
        """
        Args:
            similarities: (B, M, H, W) - spatial similarity maps

        Returns:
            pooled: (B, M) - single similarity per prototype
        """
        B, M, H, W = similarities.shape

        if self.pooling_mode == 'max':
            # Max pooling: find strongest activation
            pooled = self.pool(similarities)  # (B, M, 1, 1)
            return pooled.view(B, M)

        elif self.pooling_mode == 'attention':
            # Attention-weighted pooling
            pooled_list = []
            for m in range(M):
                sim_m = similarities[:, m:m+1, :, :]  # (B, 1, H, W)

                # Compute attention weights
                attn_logits = self.attention_conv(sim_m)  # (B, 1, H, W)
                attn_weights = F.softmax(attn_logits.view(B, 1, -1), dim=2)  # (B, 1, H*W)
                attn_weights = attn_weights.view(B, 1, H, W)

                # Weighted sum
                weighted_sim = (sim_m * attn_weights).sum(dim=[2, 3])  # (B, 1)
                pooled_list.append(weighted_sim)

            return torch.cat(pooled_list, dim=1)  # (B, M)

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")
