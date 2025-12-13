"""
Data transforms for training and evaluation.

Provides standard transforms for ImageNet-style images.
"""

import torchvision.transforms as transforms


def get_train_transforms(image_size=224, augmentation_strength='medium'):
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size (default: 224)
        augmentation_strength: 'light', 'medium', or 'strong'

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if augmentation_strength == 'light':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

    elif augmentation_strength == 'medium':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    elif augmentation_strength == 'strong':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")

    return transform


def get_val_transforms(image_size=224):
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size (default: 224)

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform


def get_clip_style_transforms(image_size=224, is_train=True):
    """
    Get CLIP-style transforms (simpler, less augmentation).

    Args:
        image_size: Target image size (default: 224)
        is_train: Whether for training or validation

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
                std=[0.26862954, 0.26130258, 0.27577711]   # CLIP std
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    return transform
