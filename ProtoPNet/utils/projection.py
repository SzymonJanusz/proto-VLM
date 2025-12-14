"""
Prototype projection utilities.

This module implements prototype projection - the process of replacing
learned prototype vectors with actual feature vectors from training images.
This grounds prototypes in real image patches for interpretability.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from pathlib import Path
import numpy as np


class PrototypeProjector:
    """
    Projects learned prototypes to nearest training image patches.

    Uses streaming algorithm to handle large datasets efficiently:
    - Process training data in batches
    - Track best matching patch per prototype
    - Update candidates incrementally

    Args:
        model: ProtoCLIP model with trained prototypes
        device: Computation device ('cuda' or 'cpu')
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.num_prototypes = model.image_encoder.num_prototypes
        self.feature_dim = model.image_encoder.feature_dim

    def project_prototypes(self, train_loader, num_batches=None):
        """
        Project prototypes to nearest training patches.

        Args:
            train_loader: DataLoader for training images
            num_batches: Optional limit on number of batches to process
                        (None = process all batches)

        Returns:
            new_prototypes: Tensor of shape (M, feature_dim) with projected prototypes
            projection_info: Dictionary containing projection metadata
        """
        print(f"\nProjecting {self.num_prototypes} prototypes...")

        # Set model to eval mode
        self.model.eval()

        # Get current prototypes
        current_prototypes = self.model.get_prototypes()  # (M, feature_dim)

        # Initialize best candidates for each prototype
        best_distances = torch.full(
            (self.num_prototypes,),
            float('inf'),
            device=self.device
        )
        best_patches = torch.zeros(
            (self.num_prototypes, self.feature_dim),
            device=self.device
        )
        best_metadata = {
            'image_idx': [-1] * self.num_prototypes,
            'batch_idx': [-1] * self.num_prototypes,
            'patch_h': [-1] * self.num_prototypes,
            'patch_w': [-1] * self.num_prototypes,
            'image_path': [None] * self.num_prototypes,
            'class_id': [-1] * self.num_prototypes,
        }

        # Access dataset from loader
        dataset = train_loader.dataset

        # Determine number of batches to process
        total_batches = len(train_loader) if num_batches is None else min(num_batches, len(train_loader))

        # Progress tracking
        pbar = tqdm(enumerate(train_loader), total=total_batches, desc="Projecting prototypes")
        total_patches_searched = 0

        with torch.no_grad():
            for batch_idx, batch_data in pbar:
                # Stop if reached batch limit
                if num_batches is not None and batch_idx >= num_batches:
                    break

                # Unpack batch (images, captions) or just images
                if isinstance(batch_data, (tuple, list)):
                    images = batch_data[0]
                else:
                    images = batch_data

                images = images.to(self.device)
                B = images.shape[0]

                # Extract features
                features = self._extract_features_batch(images)  # (B, feature_dim, H, W)
                _, _, H, W = features.shape

                # Reshape to patches: (B, feature_dim, H, W) -> (B*H*W, feature_dim)
                patches = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)  # (B*H*W, feature_dim)
                num_patches = patches.shape[0]
                total_patches_searched += num_patches

                # Compute L2 distances from all patches to all prototypes
                distances = self._compute_l2_distances(patches, current_prototypes)  # (B*H*W, M)

                # For each prototype, find if this batch has closer matches
                for proto_idx in range(self.num_prototypes):
                    proto_distances = distances[:, proto_idx]  # (B*H*W,)
                    min_dist, min_patch_idx = proto_distances.min(dim=0)

                    # Update if this is the best match so far
                    if min_dist < best_distances[proto_idx]:
                        best_distances[proto_idx] = min_dist
                        best_patches[proto_idx] = patches[min_patch_idx]

                        # Decode patch index to batch, h, w coordinates
                        patch_idx_val = min_patch_idx.item()
                        img_idx = patch_idx_val // (H * W)
                        spatial_idx = patch_idx_val % (H * W)
                        patch_h = spatial_idx // W
                        patch_w = spatial_idx % W

                        # Store metadata
                        global_img_idx = batch_idx * train_loader.batch_size + img_idx
                        best_metadata['image_idx'][proto_idx] = global_img_idx
                        best_metadata['batch_idx'][proto_idx] = batch_idx
                        best_metadata['patch_h'][proto_idx] = patch_h
                        best_metadata['patch_w'][proto_idx] = patch_w

                        # Get image path and class from dataset
                        if hasattr(dataset, 'samples') and global_img_idx < len(dataset.samples):
                            img_path, class_id = dataset.samples[global_img_idx]
                            best_metadata['image_path'][proto_idx] = str(img_path)
                            best_metadata['class_id'][proto_idx] = class_id

                # Update progress bar with current best distance
                avg_dist = best_distances[best_distances != float('inf')].mean().item()
                pbar.set_postfix({'avg_dist': f'{avg_dist:.4f}'})

        # Compute statistics
        final_distances = best_distances.cpu().numpy()
        projection_stats = {
            'total_batches': total_batches,
            'total_patches': total_patches_searched,
            'mean_distance': float(np.mean(final_distances)),
            'median_distance': float(np.median(final_distances)),
            'min_distance': float(np.min(final_distances)),
            'max_distance': float(np.max(final_distances)),
        }

        # Validate projection
        self._validate_projection(current_prototypes, best_patches, final_distances)

        # Build projection info dictionary
        projection_info = {
            'prototype_id': list(range(self.num_prototypes)),
            'image_path': best_metadata['image_path'],
            'image_idx': best_metadata['image_idx'],
            'batch_idx': best_metadata['batch_idx'],
            'patch_h': best_metadata['patch_h'],
            'patch_w': best_metadata['patch_w'],
            'distance': final_distances.tolist(),
            'class_id': best_metadata['class_id'],
            'receptive_field': {
                'patch_size': 16,  # Approximate receptive field size
                'stride': 16,
                'feature_map_size': (H, W),
            },
            'projection_stats': projection_stats,
        }

        print(f"\n✓ Projection complete!")
        print(f"  Patches searched: {total_patches_searched:,}")
        print(f"  Mean distance: {projection_stats['mean_distance']:.4f}")
        print(f"  Median distance: {projection_stats['median_distance']:.4f}")
        print(f"  Distance range: [{projection_stats['min_distance']:.4f}, {projection_stats['max_distance']:.4f}]")

        return best_patches.cpu(), projection_info

    def _extract_features_batch(self, images):
        """
        Extract features from images using the backbone.

        Args:
            images: Batch of images (B, 3, 224, 224)

        Returns:
            features: Feature maps (B, feature_dim, H, W)
        """
        return self.model.image_encoder.backbone(images)

    def _compute_l2_distances(self, patches, prototypes):
        """
        Compute L2 distances between patches and prototypes efficiently.

        Uses the formula: ||a - b||² = ||a||² + ||b||² - 2<a, b>
        This matches the distance computation in prototype_layer.py

        Args:
            patches: Patch features (N, D)
            prototypes: Prototype vectors (M, D)

        Returns:
            distances: L2 distances (N, M)
        """
        # Compute ||patches||² for each patch: (N,)
        patches_norm_sq = torch.sum(patches ** 2, dim=1)  # (N,)

        # Compute ||prototypes||² for each prototype: (M,)
        prototypes_norm_sq = torch.sum(prototypes ** 2, dim=1)  # (M,)

        # Compute dot product: patches @ prototypes.T -> (N, M)
        dot_product = torch.matmul(patches, prototypes.t())  # (N, M)

        # L2 distance squared: ||a||² + ||b||² - 2<a, b>
        # Broadcasting: (N, 1) + (1, M) - (N, M)
        distances = (
            patches_norm_sq.unsqueeze(1) +       # (N, 1)
            prototypes_norm_sq.unsqueeze(0) -    # (1, M)
            2 * dot_product                      # (N, M)
        )

        # Take square root to get L2 distance (not squared)
        # Add small epsilon for numerical stability
        distances = torch.sqrt(distances + 1e-8)

        return distances

    def _validate_projection(self, old_prototypes, new_prototypes, distances, threshold=1.0):
        """
        Validate projection quality and warn about potential issues.

        Args:
            old_prototypes: Original learned prototypes (M, D)
            new_prototypes: Projected prototypes (M, D)
            distances: Distances to nearest patches (M,)
            threshold: Warning threshold for mean distance
        """
        # Check for high distances (poor projection)
        mean_dist = np.mean(distances)
        if mean_dist > threshold:
            print(f"\n⚠️  WARNING: Mean projection distance ({mean_dist:.4f}) > threshold ({threshold})")
            print(f"   This may indicate poor prototype learning in Stage 1.")
            print(f"   Consider training for more epochs or adjusting hyperparameters.")

        # Check for duplicate projections (multiple prototypes map to same patch)
        # This is approximate - we check if prototypes are very close
        new_prototypes_np = new_prototypes.cpu().numpy()
        similarity_matrix = np.dot(new_prototypes_np, new_prototypes_np.T)
        np.fill_diagonal(similarity_matrix, 0)
        max_similarity = np.max(similarity_matrix)

        if max_similarity > 0.99:
            num_duplicates = np.sum(similarity_matrix > 0.99) // 2
            print(f"\n⚠️  WARNING: Found {num_duplicates} near-duplicate prototype projections")
            print(f"   Some prototypes mapped to very similar or identical patches.")

    def save_projection(self, save_dir, projection_info):
        """
        Save projection results to disk.

        Args:
            save_dir: Directory to save results
            projection_info: Projection metadata dictionary
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save projection info as pickle
        info_path = save_dir / 'projection_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(projection_info, f)
        print(f"\n✓ Saved projection info to: {info_path}")

        # Save human-readable summary
        summary_path = save_dir / 'projection_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PROTOTYPE PROJECTION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            stats = projection_info['projection_stats']
            f.write(f"Total batches processed: {stats['total_batches']}\n")
            f.write(f"Total patches searched: {stats['total_patches']:,}\n")
            f.write(f"Number of prototypes: {len(projection_info['prototype_id'])}\n\n")

            f.write(f"Distance Statistics:\n")
            f.write(f"  Mean: {stats['mean_distance']:.4f}\n")
            f.write(f"  Median: {stats['median_distance']:.4f}\n")
            f.write(f"  Min: {stats['min_distance']:.4f}\n")
            f.write(f"  Max: {stats['max_distance']:.4f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("TOP 10 PROTOTYPES (by distance)\n")
            f.write("=" * 70 + "\n\n")

            # Sort by distance
            sorted_indices = np.argsort(projection_info['distance'])[:10]
            for rank, idx in enumerate(sorted_indices, 1):
                f.write(f"{rank}. Prototype {idx}:\n")
                f.write(f"   Distance: {projection_info['distance'][idx]:.4f}\n")
                f.write(f"   Patch: ({projection_info['patch_h'][idx]}, {projection_info['patch_w'][idx]})\n")
                f.write(f"   Class: {projection_info['class_id'][idx]}\n")
                if projection_info['image_path'][idx]:
                    img_path = Path(projection_info['image_path'][idx])
                    f.write(f"   Image: .../{img_path.parent.name}/{img_path.name}\n")
                f.write("\n")

        print(f"✓ Saved projection summary to: {summary_path}")


def get_patch_bounding_box(patch_h, patch_w, img_size=224, feature_size=14):
    """
    Convert feature map patch coordinates to pixel bounding box.

    Args:
        patch_h: Patch row index (0 to feature_size-1)
        patch_w: Patch column index (0 to feature_size-1)
        img_size: Original image size (default: 224)
        feature_size: Feature map spatial size (default: 14)

    Returns:
        Tuple of (top, left, bottom, right) pixel coordinates
    """
    stride = img_size / feature_size

    top = int(patch_h * stride)
    left = int(patch_w * stride)
    bottom = int((patch_h + 1) * stride)
    right = int((patch_w + 1) * stride)

    return top, left, bottom, right
