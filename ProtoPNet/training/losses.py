"""
Loss functions for training ProtoCLIP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for image-text matching.
    Same as used in CLIP.

    For batch of (image, text) pairs, treats diagonal as positive pairs,
    off-diagonal as negatives.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits_per_image, logits_per_text):
        """
        Args:
            logits_per_image: (B, B) similarity matrix from image view
            logits_per_text: (B, B) similarity matrix from text view

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = logits_per_image.shape[0]

        # Ground truth: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits_per_image.device)

        # Cross entropy loss from both directions
        loss_i = F.cross_entropy(logits_per_image, labels, reduction=self.reduction)
        loss_t = F.cross_entropy(logits_per_text, labels, reduction=self.reduction)

        # Average both directions
        loss = (loss_i + loss_t) / 2.0

        return loss


class PrototypeClusteringLoss(nn.Module):
    """
    Encourages prototypes to be diverse (not collapse to same pattern).
    Computes pairwise similarity between prototypes and penalizes high similarity.
    """

    def __init__(self, min_distance=0.5):
        super().__init__()
        self.min_distance = min_distance

    def forward(self, prototype_vectors):
        """
        Args:
            prototype_vectors: (M, D) prototype vectors

        Returns:
            loss: Scalar clustering loss
        """
        # Normalize prototypes
        prototypes_norm = F.normalize(prototype_vectors, p=2, dim=1)

        # Compute pairwise cosine similarity
        similarity_matrix = prototypes_norm @ prototypes_norm.t()  # (M, M)

        # Mask out diagonal (self-similarity)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        # Penalize high similarity (want prototypes to be different)
        # Loss is high when similarity > min_distance
        loss = F.relu(similarity_matrix - self.min_distance).mean()

        return loss


class PrototypeActivationLoss(nn.Module):
    """
    Encourages each prototype to activate strongly for at least some images.
    Prevents "dead" prototypes that never activate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prototype_similarities, pooled_similarities):
        """
        Args:
            prototype_similarities: (B, M, H, W) spatial similarity maps
            pooled_similarities: (B, M) max-pooled similarities

        Returns:
            loss: Scalar activation loss
        """
        # For each prototype, check if it has high activation somewhere
        # We want max activation across batch to be high
        max_activations = pooled_similarities.max(dim=0)[0]  # (M,)

        # Penalize low maximum activations (want each prototype to activate strongly somewhere)
        # Negate because we want high activations
        loss = -max_activations.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for training ProtoCLIP.

    Loss = λ_contrastive * L_contrastive
           + λ_clustering * L_clustering
           + λ_activation * L_activation
    """

    def __init__(
        self,
        lambda_contrastive=1.0,
        lambda_clustering=0.01,
        lambda_activation=0.01,
        use_clustering=True,
        use_activation=True,
        label_smoothing=0.0
    ):
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.lambda_clustering = lambda_clustering
        self.lambda_activation = lambda_activation
        self.use_clustering = use_clustering
        self.use_activation = use_activation
        self.label_smoothing = label_smoothing

        # Choose contrastive loss based on label smoothing
        if label_smoothing > 0:
            from .label_smoothing import LabelSmoothingContrastiveLoss
            self.contrastive_loss = LabelSmoothingContrastiveLoss(epsilon=label_smoothing)
        else:
            self.contrastive_loss = ContrastiveLoss()

        if use_clustering:
            self.clustering_loss = PrototypeClusteringLoss()
        if use_activation:
            self.activation_loss = PrototypeActivationLoss()

    def forward(
        self,
        logits_per_image,
        logits_per_text,
        prototype_vectors=None,
        prototype_similarities=None,
        pooled_similarities=None
    ):
        """
        Compute combined loss.

        Args:
            logits_per_image: (B, B) image-text similarity matrix
            logits_per_text: (B, B) text-image similarity matrix
            prototype_vectors: (M, D) prototype vectors (for clustering loss)
            prototype_similarities: (B, M, H, W) spatial similarities (for activation loss)
            pooled_similarities: (B, M) pooled similarities (for activation loss)

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses for logging
        """
        # Contrastive loss (always used)
        loss_contrastive = self.contrastive_loss(logits_per_image, logits_per_text)
        total_loss = self.lambda_contrastive * loss_contrastive

        loss_dict = {'contrastive': loss_contrastive.item()}

        # Clustering loss (optional)
        if self.use_clustering and prototype_vectors is not None:
            loss_clustering = self.clustering_loss(prototype_vectors)
            total_loss += self.lambda_clustering * loss_clustering
            loss_dict['clustering'] = loss_clustering.item()

        # Activation loss (optional)
        if self.use_activation and prototype_similarities is not None and pooled_similarities is not None:
            loss_activation = self.activation_loss(prototype_similarities, pooled_similarities)
            total_loss += self.lambda_activation * loss_activation
            loss_dict['activation'] = loss_activation.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class SpatialMSELoss(nn.Module):
    """
    Spatial MSE Loss for training projections with CLIP spatial features.

    Computes MSE loss between spatial feature maps with per-location L2 normalization.
    Used to train SpatialMSEProjector to match CLIP ResNet-50 layer3 spatial features.

    Key features:
    - Per-location normalization: Each spatial position is L2-normalized independently
    - Preserves spatial structure: Computes loss at all 14×14 locations
    - Normalized embeddings: Works with unit-norm features (CLIP-style)

    Args:
        reduction: Reduction method ('mean', 'sum', or 'none')
        normalize_per_location: If True, apply L2 norm per spatial location (default: True)

    Example:
        >>> loss_fn = SpatialMSELoss()
        >>> predicted = torch.randn(32, 1024, 14, 14)  # Projected features
        >>> target = torch.randn(32, 1024, 14, 14)  # CLIP layer3 features
        >>> loss = loss_fn(predicted, target)
        >>> assert loss.shape == ()  # Scalar loss
    """

    def __init__(self, reduction: str = 'mean', normalize_per_location: bool = True):
        super().__init__()
        self.reduction = reduction
        self.normalize_per_location = normalize_per_location

    def forward(self, predicted_spatial: torch.Tensor, target_spatial: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial MSE loss with optional per-location normalization.

        Args:
            predicted_spatial: (B, C, H, W) - Projected spatial features
                Expected: (B, 1024, 14, 14) from SpatialMSEProjector
            target_spatial: (B, C, H, W) - Target CLIP spatial features
                Expected: (B, 1024, 14, 14) from CLIP ResNet-50 layer3

        Returns:
            loss: Scalar MSE loss (if reduction='mean' or 'sum')
                  or (B, H, W) loss per location (if reduction='none')

        Algorithm:
            1. Reshape spatial features: (B, C, H, W) → (B, C, H*W)
            2. Normalize per spatial location: L2 norm along channel dimension
            3. Compute MSE between normalized features
            4. Aggregate loss (mean/sum)
        """
        B, C, H, W = predicted_spatial.shape

        # Verify shapes match
        assert predicted_spatial.shape == target_spatial.shape, \
            f"Shape mismatch: predicted {predicted_spatial.shape} vs target {target_spatial.shape}"

        if self.normalize_per_location:
            # Reshape to (B, C, H*W) for per-location normalization
            pred_flat = predicted_spatial.view(B, C, H * W)  # (B, C, 196)
            target_flat = target_spatial.view(B, C, H * W)

            # L2 normalize per spatial location (along channel dimension)
            # This ensures each location is a unit vector
            pred_norm = F.normalize(pred_flat, p=2, dim=1)  # (B, C, 196)
            target_norm = F.normalize(target_flat, p=2, dim=1)

            # Compute MSE
            mse = F.mse_loss(pred_norm, target_norm, reduction=self.reduction)

        else:
            # Direct MSE without normalization
            mse = F.mse_loss(predicted_spatial, target_spatial, reduction=self.reduction)

        return mse

    def __repr__(self):
        return (f"SpatialMSELoss("
                f"reduction={self.reduction}, "
                f"normalize_per_location={self.normalize_per_location})")
