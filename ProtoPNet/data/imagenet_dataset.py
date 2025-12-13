"""
ImageNet dataset with caption generation for vision-language training.

Supports standard ImageNet directory structure:
    imagenet/
    ├── train/
    │   ├── n01440764/  (class folder)
    │   │   ├── image1.JPEG
    │   │   ├── image2.JPEG
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .caption_generator import ImageNetCaptionGenerator
import random


class ImageNetWithCaptions(Dataset):
    """
    ImageNet dataset with generated captions for vision-language training.

    Args:
        root: Path to ImageNet root directory
        split: 'train' or 'val'
        transform: Image transformations
        caption_generator: ImageNetCaptionGenerator instance (optional)
        use_multiple_captions: If True, randomly sample from multiple caption templates
        class_mapping_file: Path to class index -> name mapping JSON
    """

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        caption_generator=None,
        use_multiple_captions=False,
        class_mapping_file=None
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_multiple_captions = use_multiple_captions

        # Setup caption generator
        if caption_generator is None:
            self.caption_generator = ImageNetCaptionGenerator(
                class_mapping_file=class_mapping_file,
                use_multiple=False  # We'll handle multiple captions ourselves
            )
        else:
            self.caption_generator = caption_generator

        # Build dataset
        self.samples = self._load_samples()
        self.class_to_idx = self._build_class_to_idx()

    def _build_class_to_idx(self):
        """Build mapping from class folder name to index"""
        split_dir = self.root / self.split
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def _load_samples(self):
        """
        Load all image paths and their class indices.

        Returns:
            list: [(image_path, class_idx), ...]
        """
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        samples = []

        # Iterate through class directories
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            # Get all images in this class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    # We'll determine class_idx properly after building class_to_idx
                    samples.append((str(img_path), class_name))

        # Convert class names to indices
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(
            sorted(set(cls_name for _, cls_name in samples))
        )}

        # Replace class names with indices
        samples = [(path, class_to_idx[cls_name]) for path, cls_name in samples]

        print(f"Loaded {len(samples)} images from {len(class_to_idx)} classes")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get an image-caption pair.

        Returns:
            tuple: (image, caption)
                - image: Transformed image tensor
                - caption: Generated text caption
        """
        img_path, class_idx = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Generate caption
        if self.use_multiple_captions:
            # Generate multiple captions and randomly pick one
            captions = self.caption_generator.generate_multiple_captions(class_idx)
            caption = random.choice(captions) if captions else f"class {class_idx}"
        else:
            caption = self.caption_generator.generate_caption(class_idx)

        return image, caption


class ImageNetSubset(ImageNetWithCaptions):
    """
    Subset of ImageNet for faster experimentation.

    Args:
        root: Path to ImageNet root directory
        split: 'train' or 'val'
        num_samples: Maximum number of samples to use
        num_classes: Maximum number of classes to use (optional)
        **kwargs: Additional arguments for ImageNetWithCaptions
    """

    def __init__(
        self,
        root,
        split='train',
        num_samples=10000,
        num_classes=None,
        **kwargs
    ):
        super().__init__(root, split, **kwargs)

        # Limit number of classes if specified
        if num_classes is not None:
            unique_classes = sorted(set(class_idx for _, class_idx in self.samples))
            selected_classes = unique_classes[:num_classes]
            self.samples = [
                (path, class_idx) for path, class_idx in self.samples
                if class_idx in selected_classes
            ]

        # Limit number of samples
        if num_samples < len(self.samples):
            # Random sampling
            indices = random.sample(range(len(self.samples)), num_samples)
            self.samples = [self.samples[i] for i in indices]

        print(f"Subset: {len(self.samples)} samples from {len(set(c for _, c in self.samples))} classes")


def create_imagenet_loaders(
    imagenet_root,
    batch_size=64,
    num_workers=4,
    train_transform=None,
    val_transform=None,
    use_subset=False,
    subset_samples=10000,
    caption_generator=None
):
    """
    Create ImageNet train and validation dataloaders.

    Args:
        imagenet_root: Path to ImageNet root directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_transform: Transform for training data (optional)
        val_transform: Transform for validation data (optional)
        use_subset: If True, use subset for faster experimentation
        subset_samples: Number of samples in subset
        caption_generator: Custom caption generator (optional)

    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    from .transforms import get_train_transforms, get_val_transforms

    # Use default transforms if not provided
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()

    # Create datasets
    if use_subset:
        train_dataset = ImageNetSubset(
            root=imagenet_root,
            split='train',
            num_samples=subset_samples,
            transform=train_transform,
            caption_generator=caption_generator,
            use_multiple_captions=True
        )
        val_dataset = ImageNetSubset(
            root=imagenet_root,
            split='val',
            num_samples=subset_samples // 10,  # Smaller validation set
            transform=val_transform,
            caption_generator=caption_generator,
            use_multiple_captions=False
        )
    else:
        train_dataset = ImageNetWithCaptions(
            root=imagenet_root,
            split='train',
            transform=train_transform,
            caption_generator=caption_generator,
            use_multiple_captions=True
        )
        val_dataset = ImageNetWithCaptions(
            root=imagenet_root,
            split='val',
            transform=val_transform,
            caption_generator=caption_generator,
            use_multiple_captions=False
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoader Statistics:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


if __name__ == '__main__':
    # Example usage
    print("ImageNet Dataset Example")
    print("=" * 60)

    # This is just a demonstration
    # Replace with your actual ImageNet path
    imagenet_root = "/path/to/imagenet"

    if not Path(imagenet_root).exists():
        print(f"ImageNet not found at {imagenet_root}")
        print("Please update the path or download ImageNet dataset")
    else:
        from .transforms import get_train_transforms

        # Create dataset
        dataset = ImageNetSubset(
            root=imagenet_root,
            split='train',
            num_samples=100,
            transform=get_train_transforms(),
            use_multiple_captions=True
        )

        # Get a sample
        image, caption = dataset[0]
        print(f"\nSample:")
        print(f"  Image shape: {image.shape}")
        print(f"  Caption: {caption}")
