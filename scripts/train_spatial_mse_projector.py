#!/usr/bin/env python

"""
Train Spatial MSE Projector

Trains a spatial projection that maps ProtoPNet 14×14 prototype similarities to
CLIP ResNet-50 layer3 spatial features (1024-dim, 14×14) using per-location MSE loss.

Key Features:
- Preserves spatial structure (14×14 resolution maintained)
- MSE loss with per-location L2 normalization
- Configurable top-k sparse prototype selection
- Targets CLIP spatial features (not global or text embeddings)

Usage:
  # Sparse (top-5 prototypes)
  python scripts/train_spatial_mse_projector.py \
    --backbone_checkpoint checkpoints/protopnet/best.pt \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_k5 \
    --epochs 50 \
    --batch_size 64 \
    --top_k 5

  # No sparsity (all 200 prototypes)
  python scripts/train_spatial_mse_projector.py \
    --backbone_checkpoint checkpoints/protopnet/best.pt \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_full \
    --epochs 50 \
    --batch_size 64 \
    --top_k 200
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.spatial_mse_projector import SpatialMSEProjector
from ProtoPNet.models.clip_spatial_extractor import CLIPSpatialExtractor
from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.training.losses import SpatialMSELoss
from ProtoPNet.utils.prototype_selection import select_top_k_prototypes, compute_sparsity_statistics
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_train_transforms, get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Spatial MSE Projector for CLIP layer3 features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model & checkpoints
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                        help='Path to trained ProtoPNet checkpoint (frozen)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trained projector checkpoints')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--split_train', type=str, default='train',
                        help='Training split name')
    parser.add_argument('--split_val', type=str, default='val',
                        help='Validation split name')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (lower than contrastive due to spatial features)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Prototype selection (NEW: configurable sparsity)
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k prototypes to select (1=sparsest, 200=all)')

    # Model architecture
    parser.add_argument('--num_prototypes', type=int, default=200,
                        help='Number of prototypes')
    parser.add_argument('--output_dim', type=int, default=1024,
                        help='Output dimension (CLIP layer3 = 1024)')
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Hidden channels in projector')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def load_protopnet_model(checkpoint_path, device):
    """Load frozen ProtoCLIP model."""
    print(f"Loading ProtoCLIP from: {checkpoint_path}")
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

    print("  ProtoCLIP loaded and frozen")
    return model


def extract_prototype_similarities(images, protopnet_model, device):
    """
    Extract spatial prototype similarities from ProtoPNet.

    Args:
        images: (B, 3, 224, 224) - ImageNet normalized images
        protopnet_model: Frozen ProtoPNet model
        device: Device

    Returns:
        prototype_sims: (B, 200, 14, 14) - Spatial similarity maps
    """
    with torch.no_grad():
        # Forward through ProtoPNet to get similarities
        # ProtoCLIP model structure: has .image_encoder which is ProtoPNetEncoder
        if hasattr(protopnet_model, 'image_encoder'):
            # Get prototype similarities (before projection head)
            features = protopnet_model.image_encoder.backbone(images)  # (B, 1024, 14, 14)
            prototype_sims = protopnet_model.image_encoder.prototype_layer(features)  # (B, 200, 14, 14)
        else:
            # Direct ProtoPNetEncoder model
            features = protopnet_model.backbone(images)
            prototype_sims = protopnet_model.prototype_layer(features)

    return prototype_sims


def train_epoch(
        projector,
        protopnet_model,
        clip_extractor,
        dataloader,
        optimizer,
        loss_fn,
        top_k,
        num_prototypes,
        device
):
    """Train for one epoch."""
    projector.train()
    protopnet_model.eval()  # Frozen
    clip_extractor.eval()  # Frozen

    epoch_loss = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    # Track sparsity statistics
    sparsity_stats_list = []

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch[0].to(device)  # (B, 3, 224, 224)
        batch_size = images.shape[0]

        # 1. Extract CLIP spatial features (frozen, target)
        clip_spatial_features = clip_extractor(images)  # (B, 1024, 14, 14)
        clip_spatial_features = clip_spatial_features.float()  # FIX: Ensure float32

        # 2. Extract prototype similarities (frozen)
        prototype_sims = extract_prototype_similarities(images, protopnet_model, device)  # (B, 200, 14, 14)
        prototype_sims = prototype_sims.float()  # FIX: Ensure float32

        # 3. Apply sparse selection (configurable)
        if top_k < num_prototypes:
            sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)  # (B, 200, 14, 14) sparse

            # Compute sparsity statistics
            if batch_idx % 10 == 0:  # Don't compute every batch (expensive)
                stats = compute_sparsity_statistics(prototype_sims, sparse_sims)
                sparsity_stats_list.append(stats)
        else:
            sparse_sims = prototype_sims  # No sparsity

        # 4. Project sparse prototypes (trainable)
        projected_spatial = projector(sparse_sims)  # (B, 1024, 14, 14)

        # 5. Compute spatial MSE loss (per-location normalized)
        loss = loss_fn(projected_spatial, clip_spatial_features)

        # 6. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute additional metrics
        with torch.no_grad():
            # Cosine similarity (flattened)
            pred_flat = projected_spatial.view(batch_size, 1024, -1)  # (B, 1024, 196)
            target_flat = clip_spatial_features.view(batch_size, 1024, -1)

            # Normalize per location
            pred_norm = F.normalize(pred_flat, p=2, dim=1)
            target_norm = F.normalize(target_flat, p=2, dim=1)

            # Cosine similarity per location, then average
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()  # Mean over batch and locations

        # Update metrics
        epoch_loss += loss.item()
        epoch_cosine_sim += cosine_sim.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cosine': f'{cosine_sim.item():.4f}'
        })

    # Average metrics
    avg_loss = epoch_loss / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    # Average sparsity statistics
    if sparsity_stats_list:
        avg_sparsity_stats = {
            'mean_sparsity': np.mean([s['mean_sparsity'] for s in sparsity_stats_list]),
            'mean_num_selected': np.mean([s['mean_num_selected'] for s in sparsity_stats_list]),
        }
    else:
        avg_sparsity_stats = None

    return avg_loss, avg_cosine, avg_sparsity_stats


def validate_epoch(
        projector,
        protopnet_model,
        clip_extractor,
        dataloader,
        loss_fn,
        top_k,
        num_prototypes,
        device
):
    """Validate for one epoch."""
    projector.eval()
    protopnet_model.eval()
    clip_extractor.eval()

    epoch_loss = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.shape[0]

            # Extract features
            clip_spatial_features = clip_extractor(images)
            clip_spatial_features = clip_spatial_features.float()  # FIX: Ensure float32

            prototype_sims = extract_prototype_similarities(images, protopnet_model, device)
            prototype_sims = prototype_sims.float()  # FIX: Ensure float32

            # Apply sparse selection
            if top_k < num_prototypes:
                sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)
            else:
                sparse_sims = prototype_sims

            # Project
            projected_spatial = projector(sparse_sims)

            # Loss
            loss = loss_fn(projected_spatial, clip_spatial_features)

            # Cosine similarity
            pred_flat = projected_spatial.view(batch_size, 1024, -1)
            target_flat = clip_spatial_features.view(batch_size, 1024, -1)
            pred_norm = F.normalize(pred_flat, p=2, dim=1)
            target_norm = F.normalize(target_flat, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()

            epoch_loss += loss.item()
            epoch_cosine_sim += cosine_sim.item()
            num_batches += 1

            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_cosine': f'{cosine_sim.item():.4f}'
            })

    avg_loss = epoch_loss / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    return avg_loss, avg_cosine


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load frozen ProtoPNet model
    print(f"\nLoading ProtoPNet from: {args.backbone_checkpoint}")
    protopnet_model = load_protopnet_model(args.backbone_checkpoint, device)
    print(f"✓ ProtoPNet loaded and frozen")

    # Create CLIP spatial feature extractor
    print(f"\nInitializing CLIP spatial feature extractor (ResNet-50 layer3)...")
    clip_extractor = CLIPSpatialExtractor(layer='layer3', device=device, normalize_input=True)
    clip_extractor.eval()
    print(f"✓ CLIP extractor initialized")

    # Create trainable spatial MSE projector
    print(f"\nCreating SpatialMSEProjector...")
    projector = SpatialMSEProjector(
        num_prototypes=args.num_prototypes,
        output_dim=args.output_dim,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout
    ).to(device)
    print(f"✓ Projector created: {projector}")
    print(f"  Trainable parameters: {sum(p.numel() for p in projector.parameters() if p.requires_grad):,}")

    # Sparsity configuration
    print(f"\n{'=' * 60}")
    print(f"SPARSITY CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Top-k prototypes: {args.top_k} / {args.num_prototypes}")
    if args.top_k < args.num_prototypes:
        sparsity_ratio = (args.num_prototypes - args.top_k) / args.num_prototypes
        print(f"Sparsity ratio: {sparsity_ratio:.1%} (keeping {(1 - sparsity_ratio):.1%} of prototypes)")
    else:
        print(f"No sparsity (using all prototypes)")
    print(f"{'=' * 60}\n")

    # Create loss function
    loss_fn = SpatialMSELoss(reduction='mean', normalize_per_location=True)
    print(f"✓ Loss function: {loss_fn}")

    # Optimizer and scheduler
    optimizer = Adam(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"✓ Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"✓ Scheduler: CosineAnnealingLR")

    # Data loaders
    print(f"\nLoading datasets from: {args.data_root}")
    train_dataset = ImageNetWithCaptions(
        root=args.data_root,
        split=args.split_train,
        transform=get_train_transforms()
    )
    val_dataset = ImageNetWithCaptions(
        root=args.data_root,
        split=args.split_val,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"✓ Train dataset: {len(train_dataset)} images")
    print(f"✓ Val dataset: {len(val_dataset)} images")
    print(f"✓ Batch size: {args.batch_size}")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"STARTING TRAINING")
    print(f"{'=' * 60}\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_cosines = []
    val_cosines = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss, train_cosine, sparsity_stats = train_epoch(
            projector, protopnet_model, clip_extractor,
            train_loader, optimizer, loss_fn,
            args.top_k, args.num_prototypes, device
        )

        # Validate
        val_loss, val_cosine = validate_epoch(
            projector, protopnet_model, clip_extractor,
            val_loader, loss_fn,
            args.top_k, args.num_prototypes, device
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train Cosine: {train_cosine:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")
        if sparsity_stats:
            print(f"Sparsity: {sparsity_stats['mean_sparsity']:.1%} | "
                  f"Selected: {sparsity_stats['mean_num_selected']:.1f} prototypes")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_cosines.append(train_cosine)
        val_cosines.append(val_cosine)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, 'best.pt')
            projector.save(save_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
            projector.save(save_path)

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, 'final.pt')
    projector.save(final_path)

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves - Spatial MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_cosines, label='Train Cosine Sim')
    plt.plot(val_cosines, label='Val Cosine Sim')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Training Curves - Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    print(f"\n✓ Saved training curves")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == '__main__':
    main()
