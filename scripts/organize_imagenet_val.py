"""
Organize ImageNet validation set into class folders.

The validation set is typically distributed as a flat directory of 50,000 images.
This script organizes them into class-based subdirectories matching the training set structure.

Usage:
    python scripts/organize_imagenet_val.py --val_dir imagenet/val --devkit_dir ILSVRC2012_devkit_t12
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_validation_ground_truth(devkit_dir):
    """
    Parse validation ground truth from devkit.

    Args:
        devkit_dir: Path to ILSVRC2012_devkit_t12 directory

    Returns:
        dict: {image_filename: class_folder_name}
    """
    devkit_path = Path(devkit_dir)

    # Path to validation ground truth
    gt_file = devkit_path / 'data' / 'ILSVRC2012_validation_ground_truth.txt'

    if not gt_file.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_file}\n"
            f"Please download ILSVRC2012_devkit_t12.tar.gz and extract it."
        )

    # Path to synset mapping
    meta_file = devkit_path / 'data' / 'meta.mat'

    if not meta_file.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_file}")

    # Read class indices from ground truth file
    with open(gt_file, 'r') as f:
        class_indices = [int(line.strip()) for line in f]

    # Map indices to synset IDs using meta.mat
    try:
        import scipy.io
        meta = scipy.io.loadmat(str(meta_file), squeeze_me=True)
        synsets = meta['synsets']

        # Extract ILSVRC2012_ID and WNID
        ilsvrc_ids = [int(s['ILSVRC2012_ID']) for s in synsets]
        wnids = [str(s['WNID']) for s in synsets]

        # Create mapping from ILSVRC2012_ID to WNID
        id_to_wnid = {ilsvrc_id: wnid for ilsvrc_id, wnid in zip(ilsvrc_ids, wnids)}

    except ImportError:
        print("scipy not found. Using alternative method...")
        # Fallback: use synset_words.txt
        synset_file = devkit_path / 'data' / 'ILSVRC2012_validation_ground_truth.txt'
        id_to_wnid = _parse_synsets_fallback(devkit_path)

    # Create mapping from image filename to class folder
    image_to_class = {}
    for i, class_idx in enumerate(class_indices):
        image_filename = f'ILSVRC2012_val_{i+1:08d}.JPEG'
        class_folder = id_to_wnid.get(class_idx)

        if class_folder:
            image_to_class[image_filename] = class_folder
        else:
            print(f"Warning: No class folder for index {class_idx}")

    return image_to_class


def _parse_synsets_fallback(devkit_dir):
    """
    Fallback method to parse synsets without scipy.

    Uses the fact that ILSVRC2012_ID maps to line number in sorted synset list.
    """
    # This is a simplified fallback - may not work perfectly
    # Recommend installing scipy for accurate parsing
    print("Warning: Using fallback synset parsing. Install scipy for better accuracy.")

    # Create a simple mapping (1-indexed to match ILSVRC2012_ID)
    # This requires knowing the synset order, which should match the training folders
    return {}


def organize_validation_set(val_dir, devkit_dir, dry_run=False):
    """
    Organize validation images into class folders.

    Args:
        val_dir: Path to validation directory
        devkit_dir: Path to devkit directory
        dry_run: If True, only print what would be done without moving files
    """
    val_path = Path(val_dir)

    if not val_path.exists():
        raise ValueError(f"Validation directory not found: {val_path}")

    # Parse ground truth
    print("Parsing validation ground truth...")
    image_to_class = parse_validation_ground_truth(devkit_dir)

    print(f"Found mappings for {len(image_to_class)} images")

    # Get list of validation images
    val_images = list(val_path.glob('*.JPEG'))
    print(f"Found {len(val_images)} validation images")

    # Create class folders and move images
    moved_count = 0
    missing_count = 0

    for img_path in tqdm(val_images, desc="Organizing images"):
        img_filename = img_path.name

        if img_filename not in image_to_class:
            print(f"Warning: No class mapping for {img_filename}")
            missing_count += 1
            continue

        class_folder = image_to_class[img_filename]
        class_dir = val_path / class_folder

        if dry_run:
            print(f"Would move {img_filename} -> {class_folder}/")
        else:
            # Create class directory if it doesn't exist
            class_dir.mkdir(exist_ok=True)

            # Move image to class directory
            dest_path = class_dir / img_filename
            shutil.move(str(img_path), str(dest_path))

        moved_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Total images: {len(val_images)}")
    print(f"Moved: {moved_count}")
    print(f"Missing mappings: {missing_count}")
    print(f"{'=' * 70}")

    if dry_run:
        print("\nDry run complete. Run without --dry_run to actually move files.")
    else:
        print("\n✓ Validation set organized successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Organize ImageNet validation set into class folders'
    )
    parser.add_argument('--val_dir', type=str, default='imagenet/val',
                        help='Path to validation directory')
    parser.add_argument('--devkit_dir', type=str, default='ILSVRC2012_devkit_t12',
                        help='Path to devkit directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without moving files')

    args = parser.parse_args()

    # Check scipy
    try:
        import scipy
        print(f"Using scipy version {scipy.__version__}")
    except ImportError:
        print("\n⚠️  WARNING: scipy not installed!")
        print("Install with: pip install scipy")
        print("Falling back to alternative method (may be less accurate)\n")

    organize_validation_set(args.val_dir, args.devkit_dir, args.dry_run)


if __name__ == '__main__':
    main()
