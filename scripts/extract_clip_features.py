#!/usr/bin/env python

"""
Extract CLIP Features and Cache to HDF5

Pre-computes CLIP final embeddings for all ImageNet images and saves
to HDF5 files for fast loading during training.

Usage:
    python scripts/extract_clip_features.py \
        --data_root imagenet_tiny \
        --output_dir cached_features/clip_features \
        --batch_size 256 \
        --device cuda \
        --extract_final
"""

import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.clip_spatial_extractor import CLIPSpatialExtractor
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract and cache CLIP features to HDF5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated list of splits to process')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save cached features')
    parser.add_argument('--compression', type=str, default='gzip',
                        choices=['gzip', 'lzf', 'none'],
                        help='HDF5 compression algorithm')
    parser.add_argument('--compression_level', type=int, default=4,
                        help='Compression level (0-9 for gzip)')

    # CLIP extraction
    parser.add_argument('--extract_final', action='store_true',
                        help='Extract final embeddings (1024,) instead of layer3 spatial (1024, 14, 14)')
    parser.add_argument('--layer', type=str, default='layer3',
                        choices=['layer3', 'layer4'],
                        help='Which layer to extract spatial features from')

    # Extraction
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for extraction (CLIP is faster, can use larger batches)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    return parser.parse_args()


def create_hdf5_file(output_path, num_samples, feature_shape, compression, compression_level):
    """
    Create HDF5 file with pre-allocated datasets.

    Args:
        output_path: Path to output HDF5 file
        num_samples: Total number of samples
        feature_shape: Shape of feature tensor (e.g., (1024,) or (1024, 14, 14))
        compression: Compression algorithm
        compression_level: Compression level

    Returns:
        h5_file: Opened HDF5 file object
    """
    h5_file = h5py.File(output_path, 'w')

    # Create datasets with optimal chunking
    chunk_size = min(512, num_samples)  # Larger chunks for smaller features

    # Main feature array
    comp = None if compression == 'none' else compression
    comp_opts = compression_level if compression == 'gzip' else None

    h5_file.create_dataset(
        'features',
        shape=(num_samples, *feature_shape),
        dtype='float32',
        chunks=(chunk_size, *feature_shape),
        compression=comp,
        compression_opts=comp_opts
    )

    # Metadata arrays (variable length strings)
    dt = h5py.string_dtype(encoding='utf-8')
    h5_file.create_dataset(
        'image_paths',
        shape=(num_samples,),
        dtype=dt,
        chunks=(chunk_size,)
    )

    h5_file.create_dataset(
        'synsets',
        shape=(num_samples,),
        dtype=dt,
        chunks=(chunk_size,)
    )

    # Class indices (int)
    h5_file.create_dataset(
        'class_indices',
        shape=(num_samples,),
        dtype='int32',
        chunks=(chunk_size,)
    )

    return h5_file


def extract_and_save_features(
    clip_extractor,
    dataloader,
    h5_file,
    device,
    split_name
):
    """
    Extract CLIP features and save to HDF5.

    Args:
        clip_extractor: CLIPSpatialExtractor model
        dataloader: DataLoader for images
        h5_file: Opened HDF5 file
        device: Device
        split_name: Name of split (for progress bar)
    """
    clip_extractor.eval()

    # Track extraction statistics
    total_samples = 0
    pbar = tqdm(total=len(dataloader.dataset), desc=f'Extracting {split_name}')

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        batch_size = images.shape[0]

        # Extract CLIP features
        clip_features = clip_extractor(images)  # (B, 1024) or (B, 1024, 14, 14)
        features_to_save = clip_features.cpu().numpy()

        # Get metadata for this batch
        start_idx = batch_idx * dataloader.batch_size
        end_idx = start_idx + batch_size

        # Save to HDF5
        h5_file['features'][start_idx:end_idx] = features_to_save

        # Save metadata (paths and class indices from dataset)
        for i, sample_idx in enumerate(range(start_idx, end_idx)):
            if sample_idx < len(dataloader.dataset.samples):
                img_path, class_idx = dataloader.dataset.samples[sample_idx]

                # Extract synset from path
                # Path format: .../train/n01440764/image.JPEG
                synset = Path(img_path).parent.name

                h5_file['image_paths'][sample_idx] = img_path
                h5_file['class_indices'][sample_idx] = class_idx
                h5_file['synsets'][sample_idx] = synset

        total_samples += batch_size
        pbar.update(batch_size)

        # Clear GPU cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    print(f"  Extracted {total_samples} samples for {split_name}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse splits
    splits = args.splits.split(',')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize CLIP extractor
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    if args.extract_final:
        print(f"Initializing CLIP final embedding extractor (ResNet-50 -> final)...")
        feature_shape = (1024,)
    else:
        print(f"Initializing CLIP spatial feature extractor (ResNet-50 {args.layer})...")
        if args.layer == 'layer3':
            feature_shape = (1024, 14, 14)
        else:  # layer4
            feature_shape = (2048, 7, 7)

    clip_extractor = CLIPSpatialExtractor(
        layer=args.layer,
        device=device,
        normalize_input=True,
        extract_final=args.extract_final
    )
    clip_extractor.eval()
    print(f"  [OK] CLIP extractor initialized")
    print(f"  Feature shape: {feature_shape}")

    # Metadata to save
    metadata = {
        "model": {
            "clip_model": "RN50",
            "extract_final": args.extract_final,
            "layer": args.layer
        },
        "extraction": {
            "timestamp": datetime.now().isoformat(),
            "batch_size": args.batch_size,
            "device": str(device),
            "compression": args.compression,
            "compression_level": args.compression_level if args.compression == 'gzip' else None
        },
        "dataset": {
            "data_root": args.data_root,
            "splits": splits
        },
        "feature_shape": {
            "clip_final_embeddings" if args.extract_final else f"clip_{args.layer}_spatial": list(feature_shape)
        },
        "normalization": {
            "input": "ImageNet",
            "converted_to": "CLIP"
        }
    }

    # Process each split
    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING SPLIT: {split}")
        print("=" * 60)

        # Create dataset (use val transforms - no augmentation)
        dataset = ImageNetWithCaptions(
            root=args.data_root,
            split=split,
            transform=get_val_transforms()
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Keep order for consistent indexing
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False  # Keep all samples
        )

        print(f"  Dataset: {len(dataset)} images")
        print(f"  Batches: {len(dataloader)}")

        # Create HDF5 file
        output_path = os.path.join(args.output_dir, f'{split}.h5')
        print(f"  Creating HDF5: {output_path}")

        h5_file = create_hdf5_file(
            output_path,
            num_samples=len(dataset),
            feature_shape=feature_shape,
            compression=args.compression,
            compression_level=args.compression_level
        )

        # Extract and save features
        extract_and_save_features(
            clip_extractor=clip_extractor,
            dataloader=dataloader,
            h5_file=h5_file,
            device=device,
            split_name=split
        )

        # Close HDF5 file
        h5_file.close()
        print(f"  [OK] Saved features to: {output_path}")

        # Update metadata with split info
        metadata["dataset"][f"{split}_samples"] = len(dataset)

    # Save metadata JSON
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Saved metadata to: {metadata_path}")

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    for split in splits:
        h5_path = os.path.join(args.output_dir, f'{split}.h5')
        size_mb = os.path.getsize(h5_path) / (1024 * 1024)
        print(f"  {split}.h5: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
