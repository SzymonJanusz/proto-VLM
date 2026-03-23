"""
Download and prepare the Caltech-101 dataset for training.

Downloads caltech-101.zip from Caltech data repository, extracts it,
and creates standard train/val/test splits organized by class folders.

Usage:
    python scripts/download_caltech101.py
    python scripts/download_caltech101.py --output_dir caltech101 --train_per_class 30 --val_per_class 10
"""

import argparse
import json
import random
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
SKIP_CLASSES = {"BACKGROUND_Google"}  # Not a real category


def download_with_progress(url, dest_path):
    """Download a file with a simple progress indicator."""
    print(f"Downloading {url}")
    print(f"  -> {dest_path}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  {pct:.1f}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=report)
    print()  # newline after progress


def find_categories_dir(extract_root):
    """
    Locate the 101_ObjectCategories folder after extraction.

    The Caltech ZIP contains a nested tarball:
        caltech-101.zip
          └── caltech-101/101_ObjectCategories.tar.gz
                └── 101_ObjectCategories/
    This function handles that structure automatically.
    """
    # Direct match (already extracted)
    for candidate in [
        extract_root / "101_ObjectCategories",
        extract_root / "caltech-101" / "101_ObjectCategories",
    ]:
        if candidate.is_dir():
            return candidate

    # Look for the nested tarball and extract it
    tarball = None
    for pattern in ["**/101_ObjectCategories.tar.gz", "**/101_ObjectCategories.tar"]:
        matches = list(extract_root.glob(pattern))
        if matches:
            tarball = matches[0]
            break

    if tarball:
        print(f"Extracting nested tarball: {tarball.name} ...")
        with tarfile.open(tarball, "r:gz" if tarball.suffix == ".gz" else "r:") as tf:
            tf.extractall(tarball.parent)
        candidate = tarball.parent / "101_ObjectCategories"
        if candidate.is_dir():
            return candidate

    raise RuntimeError(
        f"Could not find '101_ObjectCategories' folder under {extract_root}. "
        "Check the ZIP structure manually."
    )


def make_splits(class_dir, train_n, val_n, seed):
    """Return (train_files, val_files, test_files) for a single class directory."""
    images = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    random.seed(seed)
    random.shuffle(images)

    train_files = images[:train_n]
    val_files = images[train_n: train_n + val_n]
    test_files = images[train_n + val_n:]
    return train_files, val_files, test_files


def copy_split(files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def build_class_mapping(class_names, save_path):
    """Write {idx: class_name} JSON mapping for 101 classes."""
    mapping = {i: name for i, name in enumerate(sorted(class_names))}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved class mapping -> {save_path}")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Download and split Caltech-101 dataset")
    parser.add_argument("--output_dir", type=str, default="caltech101",
                        help="Where to write train/val/test splits (default: caltech101/)")
    parser.add_argument("--train_per_class", type=int, default=30,
                        help="Images per class for training split (default: 30)")
    parser.add_argument("--val_per_class", type=int, default=10,
                        help="Images per class for validation split (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Keep the downloaded ZIP file after extraction")
    parser.add_argument("--class_mapping_out", type=str, default="data/caltech101_classes.json",
                        help="Path to write the class-index JSON mapping")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = Path("caltech-101.zip")
    extract_tmp = Path("caltech101_tmp")

    # ------------------------------------------------------------------ #
    # 1. Download                                                          #
    # ------------------------------------------------------------------ #
    if not zip_path.exists():
        download_with_progress(DOWNLOAD_URL, zip_path)
    else:
        print(f"ZIP already exists at {zip_path}, skipping download.")

    # ------------------------------------------------------------------ #
    # 2. Extract                                                           #
    # ------------------------------------------------------------------ #
    print(f"\nExtracting to {extract_tmp} ...")
    extract_tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    categories_dir = find_categories_dir(extract_tmp)
    print(f"Found categories at: {categories_dir}")

    # ------------------------------------------------------------------ #
    # 3. Collect class names                                               #
    # ------------------------------------------------------------------ #
    class_dirs = sorted(
        d for d in categories_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_CLASSES
    )
    class_names = [d.name for d in class_dirs]
    print(f"\nFound {len(class_names)} categories (excluding BACKGROUND_Google)")

    # ------------------------------------------------------------------ #
    # 4. Create splits                                                     #
    # ------------------------------------------------------------------ #
    stats = {"train": 0, "val": 0, "test": 0}
    skipped = []

    print(f"\nSplitting: {args.train_per_class} train / {args.val_per_class} val / rest test per class")
    for class_dir in class_dirs:
        train_files, val_files, test_files = make_splits(
            class_dir, args.train_per_class, args.val_per_class, args.seed
        )
        total = len(train_files) + len(val_files) + len(test_files)
        if total == 0:
            skipped.append(class_dir.name)
            continue

        copy_split(train_files, output_dir / "train" / class_dir.name)
        copy_split(val_files,   output_dir / "val"   / class_dir.name)
        copy_split(test_files,  output_dir / "test"  / class_dir.name)

        stats["train"] += len(train_files)
        stats["val"]   += len(val_files)
        stats["test"]  += len(test_files)

    # ------------------------------------------------------------------ #
    # 5. Class mapping JSON                                                #
    # ------------------------------------------------------------------ #
    build_class_mapping(class_names, Path(args.class_mapping_out))

    # ------------------------------------------------------------------ #
    # 6. Cleanup                                                           #
    # ------------------------------------------------------------------ #
    shutil.rmtree(extract_tmp)
    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path}")

    # ------------------------------------------------------------------ #
    # 7. Summary                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 50)
    print("Caltech-101 setup complete")
    print("=" * 50)
    print(f"  Classes  : {len(class_names)}")
    print(f"  Train    : {stats['train']} images  ->  {output_dir}/train/")
    print(f"  Val      : {stats['val']} images  ->  {output_dir}/val/")
    print(f"  Test     : {stats['test']} images  ->  {output_dir}/test/")
    if skipped:
        print(f"  Skipped  : {skipped}")
    print(f"  Mapping  : {args.class_mapping_out}")
    print("\nNext steps:")
    print(f"  python scripts/train.py --dataset caltech101 --caltech_root {output_dir} ...")


if __name__ == "__main__":
    main()
