"""
Download COCO 2017 and convert to ImageFolder format for MCPNet.

Each image is assigned to the single category whose bounding boxes cover
the largest total area in that image (dominant-class assignment).
Images with no annotations are skipped.

Usage:
    python prepare_coco.py --out_dir /path/to/coco_dataset
"""

import argparse
import json
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

URLS = {
    "train2017.zip":                  "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip":                    "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip":   "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)


def extract(zip_path: Path, dest: Path):
    marker = dest / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  already extracted: {zip_path.name}")
        return
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.touch()


def build_imagefolder(raw_dir: Path, out_dir: Path, split: str, ann_filename: str):
    ann_path = raw_dir / "annotations" / ann_filename
    img_src = raw_dir / f"{split}2017"
    split_out = out_dir / split

    print(f"\nBuilding {split} split ...")
    with open(ann_path) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"].replace(" ", "_") for c in data["categories"]}
    img_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Sum bbox areas per (image, category)
    area_by_img_cat = defaultdict(lambda: defaultdict(float))
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        w, h = ann["bbox"][2], ann["bbox"][3]
        area_by_img_cat[ann["image_id"]][ann["category_id"]] += w * h

    skipped = 0
    assigned = 0
    for img_id, cat_areas in area_by_img_cat.items():
        best_cat = max(cat_areas, key=cat_areas.get)
        cat_name = cat_map[best_cat]
        filename = img_filename[img_id]
        src = img_src / filename
        dst = split_out / cat_name / filename
        if not src.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        assigned += 1

    n_classes = len(list(split_out.iterdir()))
    print(f"  {assigned} images assigned across {n_classes} classes, {skipped} skipped (missing source)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Root output directory for ImageFolder data")
    parser.add_argument("--raw_dir", default=None, help="Where to store raw downloads (default: out_dir/raw)")
    parser.add_argument("--keep_zips", action="store_true", help="Keep zip files after extraction")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading COCO 2017 ===")
    for filename, url in URLS.items():
        download(url, raw_dir / filename)

    print("\n=== Extracting ===")
    for filename in URLS:
        extract(raw_dir / filename, raw_dir)
        if not args.keep_zips:
            zip_path = raw_dir / filename
            if zip_path.exists():
                zip_path.unlink()
                print(f"  removed {filename}")

    print("\n=== Converting to ImageFolder ===")
    build_imagefolder(raw_dir, out_dir, "train", "instances_train2017.json")
    build_imagefolder(raw_dir, out_dir, "val",   "instances_val2017.json")

    print("\nDone. ImageFolder structure written to:", out_dir)
    print("  train/  ->", str(out_dir / "train"))
    print("  val/    ->", str(out_dir / "val"))


if __name__ == "__main__":
    main()
