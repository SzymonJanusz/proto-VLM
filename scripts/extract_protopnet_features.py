#!/usr/bin/env python

"""
Extract ProtoPNet Features and Cache to HDF5

Pre-computes prototype similarities for all ImageNet images and saves
to HDF5 files for fast loading during training.

Usage:
    python scripts/extract_protopnet_features.py \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --data_root imagenet_tiny \
        --output_dir cached_features/protopnet_features \
        --batch_size 128 \
        --device cuda
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

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract and cache ProtoPNet features to HDF5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                        help='Path to trained ProtoPNet checkpoint')

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

    # Extraction
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Optionally save sparse top-k features (default: None, saves all)')

    return parser.parse_args()


def load_protopnet_model(checkpoint_path, device):
    """Load ProtoCLIP model from checkpoint."""
    print(f"Loading ProtoPNet from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect projection_hidden_dim from checkpoint
    state_dict = checkpoint['model_state_dict']
    projection_weight = state_dict['image_encoder.projection_head.0.weight']
    projection_hidden_dim = projection_weight.shape[0]
    print(f"  Detected projection_hidden_dim: {projection_hidden_dim}")

    # Create model
    model = ProtoCLIP(
        num_prototypes=200,
        image_backbone='resnet50',
        embedding_dim=512,
        pooling_mode='max',
        freeze_text_encoder=True,
        projection_hidden_dim=projection_hidden_dim
    ).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print("  ProtoPNet loaded and frozen")
    return model


def extract_prototype_similarities(images, protopnet_model, device):
    """
    Extract spatial prototype similarities from ProtoPNet.

    Args:
        images: (B, 3, 224, 224) - ImageNet normalized images
        protopnet_model: ProtoPNet model (frozen)
        device: Device

    Returns:
        prototype_sims: (B, 200, 14, 14) - Spatial similarity maps
    """
    with torch.no_grad():
        # Forward through ProtoPNet to get similarities
        if hasattr(protopnet_model, 'image_encoder'):
            # Get prototype similarities (before projection head)
            features = protopnet_model.image_encoder.backbone(images)  # (B, 1024, 14, 14)
            prototype_sims = protopnet_model.image_encoder.prototype_layer(features)  # (B, 200, 14, 14)
        else:
            # Direct ProtoPNetEncoder model
            features = protopnet_model.backbone(images)
            prototype_sims = protopnet_model.prototype_layer(features)

    return prototype_sims


def create_hdf5_file(output_path, num_samples, feature_shape, compression, compression_level):
    """
    Create HDF5 file with pre-allocated datasets.

    Args:
        output_path: Path to output HDF5 file
        num_samples: Total number of samples
        feature_shape: Shape of feature tensor (e.g., (200, 14, 14))
        compression: Compression algorithm
        compression_level: Compression level

    Returns:
        h5_file: Opened HDF5 file object
    """
    h5_file = h5py.File(output_path, 'w')

    # Create datasets with optimal chunking
    chunk_size = min(256, num_samples)  # Chunk by batches

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
    model,
    dataloader,
    h5_file,
    device,
    split_name,
    top_k=None
):
    """
    Extract features and save to HDF5.

    Args:
        model: ProtoPNet model
        dataloader: DataLoader for images
        h5_file: Opened HDF5 file
        device: Device
        split_name: Name of split (for progress bar)
        top_k: Optional top-k sparse selection
    """
    model.eval()

    # Track extraction statistics
    total_samples = 0
    pbar = tqdm(total=len(dataloader.dataset), desc=f'Extracting {split_name}')

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        batch_size = images.shape[0]

        # Extract prototype similarities
        prototype_sims = extract_prototype_similarities(images, model, device)  # (B, 200, 14, 14)

        # Optional: Apply sparse selection
        if top_k is not None:
            from ProtoPNet.utils.prototype_selection import select_top_k_prototypes
            sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)
            features_to_save = sparse_sims.cpu().numpy()
        else:
            features_to_save = prototype_sims.cpu().numpy()

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

    # Load ProtoPNet model
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    protopnet_model = load_protopnet_model(args.backbone_checkpoint, device)

    # Metadata to save
    metadata = {
        "model": {
            "checkpoint_path": args.backbone_checkpoint,
            "num_prototypes": 200
        },
        "extraction": {
            "timestamp": datetime.now().isoformat(),
            "batch_size": args.batch_size,
            "device": str(device),
            "top_k": args.top_k,
            "compression": args.compression,
            "compression_level": args.compression_level if args.compression == 'gzip' else None
        },
        "dataset": {
            "data_root": args.data_root,
            "splits": splits
        },
        "feature_shape": {
            "prototype_similarities": [200, 14, 14]
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
            feature_shape=(200, 14, 14),
            compression=args.compression,
            compression_level=args.compression_level
        )

        # Extract and save features
        extract_and_save_features(
            model=protopnet_model,
            dataloader=dataloader,
            h5_file=h5_file,
            device=device,
            split_name=split,
            top_k=args.top_k
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
