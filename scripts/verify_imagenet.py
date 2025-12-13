"""
Verify ImageNet dataset structure and integrity.

This script checks:
1. Directory structure is correct
2. Expected number of classes
3. Images are readable
4. Class naming is correct

Usage:
    python scripts/verify_imagenet.py --imagenet_root ./imagenet
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys


def verify_imagenet(imagenet_root, quick_check=False):
    """
    Verify ImageNet dataset structure.

    Args:
        imagenet_root: Path to ImageNet root directory
        quick_check: If True, only check structure without loading images
    """
    root = Path(imagenet_root)

    print("=" * 70)
    print("ImageNet Dataset Verification")
    print("=" * 70)
    print(f"Root directory: {root.absolute()}")

    # Check 1: Root directory exists
    if not root.exists():
        print(f"\n✗ Error: Directory not found: {root}")
        return False

    print("\n✓ Root directory exists")

    # Check 2: Train and val directories exist
    train_dir = root / 'train'
    val_dir = root / 'val'

    if not train_dir.exists():
        print(f"✗ Error: Train directory not found: {train_dir}")
        return False

    if not val_dir.exists():
        print(f"⚠️  Warning: Val directory not found: {val_dir}")
        print("   (You can train without validation, but it's recommended)")
        val_dir = None
    else:
        print("✓ Train and val directories exist")

    # Check 3: Number of classes
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_train_classes = len(train_classes)

    print(f"\n✓ Found {num_train_classes} training classes")

    if val_dir:
        val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
        num_val_classes = len(val_classes)
        print(f"✓ Found {num_val_classes} validation classes")

        if num_train_classes != num_val_classes:
            print(f"⚠️  Warning: Train ({num_train_classes}) and val ({num_val_classes}) class counts differ")

    # Check 4: Class naming (should be WordNet IDs like n01440764)
    print("\nChecking class naming...")
    valid_naming = True
    for class_name in train_classes[:10]:  # Check first 10
        if not (class_name.startswith('n') and len(class_name) == 9):
            print(f"⚠️  Warning: Unexpected class name format: {class_name}")
            print("   Expected format: n01440764 (WordNet ID)")
            valid_naming = False
            break

    if valid_naming:
        print("✓ Class naming looks correct")

    # Check 5: Sample some images
    print("\nChecking sample images...")

    total_train_images = 0
    total_val_images = 0
    corrupted_images = []

    # Count train images
    print("Counting training images...")
    for class_dir in tqdm(train_classes, desc="Train classes"):
        class_path = train_dir / class_dir
        images = list(class_path.glob('*.JPEG')) + list(class_path.glob('*.jpg'))
        total_train_images += len(images)

        # Try loading a few images if not quick check
        if not quick_check and len(images) > 0:
            for img_path in images[:3]:  # Check first 3 images per class
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception as e:
                    corrupted_images.append((str(img_path), str(e)))

    print(f"✓ Total training images: {total_train_images:,}")

    # Count val images
    if val_dir:
        print("Counting validation images...")
        for class_dir in tqdm(val_classes, desc="Val classes"):
            class_path = val_dir / class_dir
            images = list(class_path.glob('*.JPEG')) + list(class_path.glob('*.jpg'))
            total_val_images += len(images)

        print(f"✓ Total validation images: {total_val_images:,}")

    # Check for corrupted images
    if corrupted_images:
        print(f"\n⚠️  Found {len(corrupted_images)} corrupted images:")
        for img_path, error in corrupted_images[:5]:  # Show first 5
            print(f"   {img_path}: {error}")
    elif not quick_check:
        print("\n✓ Sample images are readable")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    expected_classes = 1000  # ILSVRC2012 standard
    expected_train_images = 1281167  # Approximate
    expected_val_images = 50000

    # Class count
    if num_train_classes == expected_classes:
        print(f"✓ Class count: {num_train_classes} (standard ILSVRC2012)")
    elif num_train_classes < expected_classes:
        print(f"⚠️  Class count: {num_train_classes} (subset of ImageNet)")
    else:
        print(f"⚠️  Class count: {num_train_classes} (more than standard)")

    # Image count
    if total_train_images > 0:
        if abs(total_train_images - expected_train_images) < 1000:
            print(f"✓ Train images: {total_train_images:,} (full dataset)")
        else:
            print(f"⚠️  Train images: {total_train_images:,} (expected ~{expected_train_images:,})")
    else:
        print(f"✗ Train images: 0 (no images found!)")
        return False

    if val_dir and total_val_images > 0:
        if abs(total_val_images - expected_val_images) < 100:
            print(f"✓ Val images: {total_val_images:,} (full validation set)")
        else:
            print(f"⚠️  Val images: {total_val_images:,} (expected ~{expected_val_images:,})")

    # Final verdict
    print("\n" + "=" * 70)

    if total_train_images > 0 and len(corrupted_images) == 0:
        print("✓ Dataset verification PASSED")
        print("\nYou can now run training with:")
        print(f"  python scripts/train.py --imagenet_root {imagenet_root}")
        return True
    elif total_train_images > 0:
        print("⚠️  Dataset verification PASSED with warnings")
        print(f"\nFound {len(corrupted_images)} corrupted images, but dataset is usable")
        return True
    else:
        print("✗ Dataset verification FAILED")
        print("\nPlease check the dataset structure and image files")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify ImageNet dataset structure'
    )
    parser.add_argument('--imagenet_root', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick check (skip image loading)')

    args = parser.parse_args()

    success = verify_imagenet(args.imagenet_root, quick_check=args.quick)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
