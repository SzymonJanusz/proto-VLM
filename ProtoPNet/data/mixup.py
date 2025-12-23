"""
CutMix augmentation for contrastive learning.
Better than Mixup for prototype learning as it preserves local structure.
"""

import torch
import numpy as np
import random


class CutMix:
    """
    CutMix augmentation - better than Mixup for prototype learning.

    Creates virtual training examples by cutting and pasting patches between images.
    Unlike Mixup which blends entire images, CutMix preserves local structure,
    making it more suitable for prototype-based models.

    Reference:
        CutMix: Regularization Strategy to Train Strong Classifiers
        with Localizable Features (Yun et al., ICCV 2019)

    Args:
        alpha: Beta distribution parameter (default: 1.0)
            - Higher alpha = more aggressive mixing
            - alpha=1.0 is standard, alpha=0.5 for conservative mixing
        prob: Probability of applying cutmix (default: 0.5)

    Example:
        >>> cutmix = CutMix(alpha=1.0, prob=0.5)
        >>> images = torch.randn(32, 3, 224, 224)
        >>> mixed_images, lam = cutmix(images)
    """

    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images):
        """
        Apply cutmix to a batch of images.

        Args:
            images: (B, C, H, W) tensor

        Returns:
            mixed_images: (B, C, H, W) tensor with patches mixed
            lam: Mixing coefficient (proportion of original image)
        """
        if random.random() > self.prob:
            return images, 1.0

        batch_size, _, H, W = images.size()

        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation for mixing
        index = torch.randperm(batch_size, device=images.device)

        # Sample bounding box
        # Box area proportional to (1 - lambda)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Mix images by replacing rectangular region
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda to match actual box size
        # (may differ from sampled lambda due to clipping)
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

        return mixed_images, lam


class Mixup:
    """
    Mixup augmentation for images.

    Creates virtual training examples by linearly interpolating pairs of images.
    Note: CutMix is generally preferred for prototype learning as it preserves
    local structure better.

    Reference:
        mixup: Beyond Empirical Risk Minimization (Zhang et al., ICLR 2018)

    Args:
        alpha: Beta distribution parameter (default: 0.2)
            - Higher alpha = more aggressive mixing
            - alpha=0.2 is conservative, good for prototype learning
        prob: Probability of applying mixup (default: 0.5)

    Example:
        >>> mixup = Mixup(alpha=0.2, prob=0.5)
        >>> images = torch.randn(32, 3, 224, 224)
        >>> mixed_images, lam = mixup(images)
    """

    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images):
        """
        Apply mixup to a batch of images.

        Args:
            images: (B, C, H, W) tensor

        Returns:
            mixed_images: (B, C, H, W) tensor with images linearly mixed
            lam: Mixing coefficient
        """
        if random.random() > self.prob:
            return images, 1.0

        batch_size = images.size(0)

        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size, device=images.device)

        # Mix images: mixed = lam * img1 + (1-lam) * img2
        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, lam
