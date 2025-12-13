"""
Download and organize Tiny-ImageNet dataset.

Tiny-ImageNet is a subset of ImageNet with:
- 200 classes
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
- 50 test images per class (10,000 total)
- Images are 64x64 pixels (will be resized to 224x224 during training)

Perfect for quick experimentation and testing!

Usage:
    python scripts/download_tiny_imagenet.py --output ./imagenet_tiny
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_with_progress(url, output_path):
    """Download file with progress bar"""

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def organize_tiny_imagenet(tiny_imagenet_dir, output_dir):
    """
    Organize Tiny-ImageNet to match standard ImageNet structure.

    Tiny-ImageNet has a different structure:
        train/
            n01443537/
                images/
                    n01443537_0.JPEG
        val/
            images/
                val_0.JPEG
            val_annotations.txt

    We need to reorganize to:
        train/
            n01443537/
                n01443537_0.JPEG
        val/
            n01443537/
                val_0.JPEG
    """
    tiny_path = Path(tiny_imagenet_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nOrganizing Tiny-ImageNet structure...")

    # Organize training set
    print("Organizing training set...")
    train_src = tiny_path / 'train'
    train_dst = output_path / 'train'
    train_dst.mkdir(exist_ok=True)

    for class_dir in tqdm(list(train_src.iterdir()), desc="Train classes"):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images_src = class_dir / 'images'

        if not images_src.exists():
            continue

        # Create class directory
        class_dst = train_dst / class_name
        class_dst.mkdir(exist_ok=True)

        # Copy images (remove 'images' subdirectory)
        for img in images_src.glob('*.JPEG'):
            shutil.copy2(img, class_dst / img.name)

    # Organize validation set
    print("Organizing validation set...")
    val_src = tiny_path / 'val'
    val_dst = output_path / 'val'
    val_dst.mkdir(exist_ok=True)

    # Parse val_annotations.txt to get image-to-class mapping
    val_annotations = val_src / 'val_annotations.txt'
    if not val_annotations.exists():
        print(f"⚠️  Warning: val_annotations.txt not found")
        return

    image_to_class = {}
    with open(val_annotations, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                image_name = parts[0]
                class_name = parts[1]
                image_to_class[image_name] = class_name

    # Move validation images to class folders
    val_images_src = val_src / 'images'
    if val_images_src.exists():
        for img in tqdm(list(val_images_src.glob('*.JPEG')), desc="Val images"):
            img_name = img.name
            if img_name in image_to_class:
                class_name = image_to_class[img_name]
                class_dst = val_dst / class_name
                class_dst.mkdir(exist_ok=True)
                shutil.copy2(img, class_dst / img_name)

    print(f"\n✓ Organized Tiny-ImageNet to: {output_path}")


def download_tiny_imagenet(output_dir):
    """Download and extract Tiny-ImageNet"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download URL
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = output_path / "tiny-imagenet-200.zip"
    extract_path = output_path / "tiny-imagenet-200"

    # Check if already downloaded
    if extract_path.exists():
        print(f"Tiny-ImageNet already exists at: {extract_path}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            return extract_path

    # Download
    print("\n" + "=" * 70)
    print("Downloading Tiny-ImageNet")
    print("=" * 70)
    print(f"URL: {url}")
    print(f"Destination: {zip_path}")
    print("\nThis will download ~250MB...")

    try:
        download_with_progress(url, zip_path)
        print(f"\n✓ Download complete!")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return None

    # Extract
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

    print(f"✓ Extracted to: {extract_path}")

    # Clean up zip file
    print("\nCleaning up...")
    zip_path.unlink()
    print("✓ Removed zip file")

    return extract_path


def main():
    parser = argparse.ArgumentParser(
        description='Download and organize Tiny-ImageNet dataset'
    )
    parser.add_argument('--output', type=str, default='./imagenet_tiny',
                        help='Output directory for organized dataset')
    parser.add_argument('--keep_original', action='store_true',
                        help='Keep original Tiny-ImageNet structure')

    args = parser.parse_args()

    print("=" * 70)
    print("Tiny-ImageNet Download and Setup")
    print("=" * 70)

    # Download
    tiny_imagenet_path = download_tiny_imagenet(args.output)

    if tiny_imagenet_path is None:
        print("\n✗ Failed to download Tiny-ImageNet")
        return

    # Organize to standard ImageNet structure
    organized_path = Path(args.output) / "organized"
    organize_tiny_imagenet(tiny_imagenet_path, organized_path)

    # Clean up original structure if not keeping
    if not args.keep_original:
        print("\nRemoving original structure...")
        shutil.rmtree(tiny_imagenet_path)
        print("✓ Removed original structure")

        # Move organized to main directory
        for item in organized_path.iterdir():
            shutil.move(str(item), str(Path(args.output) / item.name))
        organized_path.rmdir()

    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"Dataset location: {args.output}")
    print("\nDataset info:")
    print("  - 200 classes")
    print("  - 100,000 training images")
    print("  - 10,000 validation images")
    print("  - Images: 64x64 (will be resized to 224x224)")
    print("\nYou can now run training with:")
    print(f"  python scripts/train.py --imagenet_root {args.output} --use_subset")
    print("\nOr verify the dataset:")
    print(f"  python scripts/verify_imagenet.py --imagenet_root {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
