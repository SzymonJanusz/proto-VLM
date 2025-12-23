"""
Label smoothing for contrastive learning.
Prevents overconfident predictions and improves generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingContrastiveLoss(nn.Module):
    """
    Contrastive loss with label smoothing.

    Instead of hard targets (0 or 1), uses soft targets:
    - Positive pairs: 1 - epsilon
    - Negative pairs: epsilon / (num_negatives)

    This prevents the model from becoming overconfident and improves
    generalization, especially helpful for preventing overfitting.

    Args:
        epsilon: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        reduction: 'mean' or 'sum'

    Example:
        >>> loss_fn = LabelSmoothingContrastiveLoss(epsilon=0.1)
        >>> logits_i = torch.randn(32, 32)  # Image-to-text similarities
        >>> logits_t = torch.randn(32, 32)  # Text-to-image similarities
        >>> loss = loss_fn(logits_i, logits_t)
    """

    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits_per_image, logits_per_text):
        """
        Compute label-smoothed contrastive loss.

        Args:
            logits_per_image: (B, B) similarity matrix from image view
            logits_per_text: (B, B) similarity matrix from text view

        Returns:
            loss: Scalar loss value
        """
        batch_size = logits_per_image.shape[0]

        # Create smoothed labels
        # Hard labels: diagonal = 1, off-diagonal = 0
        # Smooth labels: diagonal = 1-epsilon, off-diagonal = epsilon/(B-1)
        labels = torch.zeros_like(logits_per_image)
        labels.fill_(self.epsilon / (batch_size - 1))
        labels[range(batch_size), range(batch_size)] = 1.0 - self.epsilon

        # Cross entropy with soft labels
        log_probs_i = F.log_softmax(logits_per_image, dim=1)
        log_probs_t = F.log_softmax(logits_per_text, dim=1)

        loss_i = -(labels * log_probs_i).sum(dim=1)
        loss_t = -(labels * log_probs_t).sum(dim=1)

        if self.reduction == 'mean':
            loss = (loss_i.mean() + loss_t.mean()) / 2.0
        else:
            loss = (loss_i.sum() + loss_t.sum()) / 2.0

        return loss
