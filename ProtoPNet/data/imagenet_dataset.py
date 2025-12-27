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
import json


def load_synset_to_idx_mapping(mapping_file='data/imagenet_class_index.json'):
    """
    Load synset ID to class index mapping from ImageNet class index file.

    Args:
        mapping_file: Path to ImageNet class index JSON file

    Returns:
        tuple: (synset_to_idx dict, synset_to_name dict)
            - synset_to_idx: {synset_id: class_idx}
            - synset_to_name: {synset_id: class_name}

    Example:
        >>> synset_to_idx, synset_to_name = load_synset_to_idx_mapping()
        >>> synset_to_idx['n02094433']
        187
        >>> synset_to_name['n02094433']
        'Yorkshire_terrier'
    """
    mapping_path = Path(mapping_file)

    if not mapping_path.exists():
        print(f"Warning: Synset mapping file not found at {mapping_file}")
        return None, None

    try:
        with open(mapping_path, 'r') as f:
            data = json.load(f)

        synset_to_idx = {}
        synset_to_name = {}

        # Parse the Keras format: {class_idx: [synset_id, class_name]}
        for class_idx_str, (synset_id, class_name) in data.items():
            class_idx = int(class_idx_str)
            synset_to_idx[synset_id] = class_idx
            synset_to_name[synset_id] = class_name

        print(f"Loaded synset mapping for {len(synset_to_idx)} ImageNet classes")
        return synset_to_idx, synset_to_name

    except Exception as e:
        print(f"Error loading synset mapping from {mapping_file}: {e}")
        return None, None


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
        class_mapping_file=None,
        synset_mapping_file='data/imagenet_class_index.json'
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_multiple_captions = use_multiple_captions

        # Load synset to class index mapping
        self.synset_to_idx, self.synset_to_name = load_synset_to_idx_mapping(synset_mapping_file)

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
        """Build mapping from synset ID (class folder name) to standard ImageNet class index"""
        split_dir = self.root / self.split

        if self.synset_to_idx is None:
            # Fallback to alphabetical ordering if no synset mapping available
            classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            return {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            # Use standard ImageNet class indices
            class_to_idx = {}
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    synset_id = class_dir.name
                    if synset_id in self.synset_to_idx:
                        class_to_idx[synset_id] = self.synset_to_idx[synset_id]
            return class_to_idx

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
        skipped_synsets = set()
        synset_counts = {}

        # Iterate through class directories
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            synset_id = class_dir.name  # e.g., 'n02094433'

            # Map synset to class index using standard ImageNet ordering
            if self.synset_to_idx is not None and synset_id in self.synset_to_idx:
                class_idx = self.synset_to_idx[synset_id]
            else:
                # Synset not in standard ImageNet 1000 classes
                skipped_synsets.add(synset_id)
                continue

            # Get all images in this class directory
            img_count = 0
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    samples.append((str(img_path), class_idx))
                    img_count += 1

            synset_counts[synset_id] = img_count

        # Report statistics
        print(f"Loaded {len(samples)} images from {len(synset_counts)} classes")
        if skipped_synsets:
            print(f"  Skipped {len(skipped_synsets)} synsets not in ImageNet-1K:")
            for synset in sorted(skipped_synsets)[:5]:  # Show first 5
                print(f"    - {synset}")
            if len(skipped_synsets) > 5:
                print(f"    ... and {len(skipped_synsets) - 5} more")

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
