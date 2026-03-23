#!/usr/bin/env python

"""
Train Spatial MSE Projector from Cached Features

Fast training using pre-computed ProtoPNet and CLIP features.
No feature extraction during training - loads from HDF5 cache.

Usage:
    python scripts/train_projector_from_cache.py \
        --cache_root cached_features \
        --output_dir checkpoints/projector_exp1 \
        --epochs 50 \
        --batch_size 256 \
        --lr 1e-4 \
        --top_k 1
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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.spatial_mse_projector import SpatialMSEProjector
from ProtoPNet.models.improved_spatial_projector import ImprovedSpatialMSEProjector
from ProtoPNet.utils.prototype_selection import select_top_k_prototypes
from ProtoPNet.data.cached_feature_dataset import CachedFeatureDataset, load_cached_train_dataset, load_cached_val_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Spatial MSE Projector from cached features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Cached features
    parser.add_argument('--cache_root', type=str, required=True,
                        help='Root directory with cached_features/')
    parser.add_argument('--clip_cache_root', type=str, default=None,
                        help='Separate cache root for CLIP features. Defaults to --cache_root. '
                             'Use to point at spatial (1024,14,14) vs global (1024,) CLIP cache.')
    parser.add_argument('--in_memory', action='store_true',
                        help='Load entire dataset into RAM (faster but requires ~17GB)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trained projector checkpoints')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (can be larger since no feature extraction!)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Prototype selection
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top-k prototypes to select (1=sparsest, 200=all)')

    # Model architecture
    parser.add_argument('--projector_type', type=str, default='baseline',
                        choices=['baseline', 'improved'],
                        help='Projector architecture: baseline (SpatialMSEProjector) or improved (ImprovedSpatialMSEProjector)')
    parser.add_argument('--num_prototypes', type=int, default=200,
                        help='Number of prototypes')
    parser.add_argument('--output_dim', type=int, default=1024,
                        help='Output dimension (CLIP final embeddings = 1024)')
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Hidden channels in projector (baseline) or hidden_dim (improved)')
    parser.add_argument('--encoder_dim', type=int, default=256,
                        help='Encoder output channels (improved architecture only)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--use_attention', action='store_true',
                        help='Enable spatial attention (improved architecture only)')

    # Advanced training (for improved architecture)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (improved architecture)')
    parser.add_argument('--encoder_lr_mult', type=float, default=2.0,
                        help='Learning rate multiplier for sparse encoder (improved architecture)')
    parser.add_argument('--refinement_lr_mult', type=float, default=0.5,
                        help='Learning rate multiplier for refinement (improved architecture)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable prototype dropout augmentation (ONLY dropout, no spatial augmentation!)')
    parser.add_argument('--aug_zero_prob', type=float, default=0.1,
                        help='Probability of prototype dropout augmentation (input noise)')
    parser.add_argument('--ortho_weight', type=float, default=0.0,
                        help='Orthogonality regularization weight (0=disabled)')
    parser.add_argument('--mse_weight', type=float, default=0.0,
                        help='MSE loss weight in combined loss (default: 0.0 = cosine only)')
    parser.add_argument('--cosine_weight', type=float, default=1.0,
                        help='Cosine loss weight in combined loss (default: 1.0)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0=main process, recommended for HDF5 on Windows)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def augment_features(prototype_sims, p_zero=0.1):
    """
    Augment prototype similarities during training.

    CRITICAL: NO spatial augmentation (horizontal flip) with cached features!
    - Cached CLIP targets are global (1024-dim) from original images
    - Flipping spatial prototypes creates wrong mapping: flipped input → unflipped target
    - Only prototype dropout is valid (input noise/regularization)

    Args:
        prototype_sims: (B, M, H, W) - Prototype similarities
        p_zero: Probability of zeroing additional prototypes (dropout)

    Returns:
        augmented: (B, M, H, W) - Augmented prototype similarities
    """
    B, M, H, W = prototype_sims.shape

    # REMOVED: Horizontal flip - INVALID with cached global CLIP targets!
    # Would create: flipped prototypes → unflipped CLIP embedding (WRONG!)

    # VALID: Random prototype dropout (force robustness to missing prototypes)
    # Same image should produce same CLIP embedding even with fewer prototypes
    if torch.rand(1).item() < p_zero:
        # For each sample in batch, zero out additional prototypes
        for b in range(B):
            # Find currently active prototypes
            active_mask = (prototype_sims[b].abs().sum(dim=(1, 2)) > 0)
            active_indices = torch.nonzero(active_mask, as_tuple=True)[0]

            if len(active_indices) > 1:  # Only drop if more than 1 active
                # Drop 10% of active prototypes
                num_to_zero = max(1, int(len(active_indices) * 0.1))
                drop_indices = active_indices[torch.randperm(len(active_indices))[:num_to_zero]]
                prototype_sims[b, drop_indices] = 0

    return prototype_sims


def train_epoch(projector, dataloader, optimizer, device, top_k=None, augment=False, aug_zero_prob=0.1,
                ortho_weight=0.0, mse_weight=0.0, cosine_weight=1.0):
    """Train for one epoch with weighted combination of MSE and cosine loss."""
    projector.train()

    epoch_loss = 0.0
    epoch_loss_mse = 0.0
    epoch_loss_cosine = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (prototype_sims, clip_embeddings, _, _) in enumerate(pbar):
        # Move to device
        prototype_sims = prototype_sims.to(device)  # (B, 200, 14, 14)
        clip_embeddings = clip_embeddings.to(device)  # (B, 1024)
        batch_size = prototype_sims.shape[0]

        # Apply top-k sparse selection if specified
        if top_k is not None:
            prototype_sims = select_top_k_prototypes(prototype_sims, k=top_k)

        # Apply feature augmentation (if enabled)
        # ONLY prototype dropout - NO spatial augmentation with cached features!
        if augment:
            prototype_sims = augment_features(prototype_sims, p_zero=aug_zero_prob)

        # Project prototypes (trainable)
        projected_spatial = projector(prototype_sims)  # (B, 1024, 14, 14)

        # Normalize and compute losses — branch on target shape
        if clip_embeddings.dim() == 4:
            # Spatial target (B, 1024, 14, 14): compare per-location, no pooling
            projected_norm = F.normalize(projected_spatial, p=2, dim=1)
            target_norm    = F.normalize(clip_embeddings,   p=2, dim=1)
            loss_mse    = F.mse_loss(projected_norm, target_norm)
            loss_cosine = 1.0 - (projected_norm * target_norm).sum(dim=1).mean()
        else:
            # Global target (B, 1024): pool then compare
            projected_global = F.adaptive_avg_pool2d(projected_spatial, (1, 1)).squeeze(-1).squeeze(-1)
            projected_norm   = F.normalize(projected_global,  p=2, dim=1)
            target_norm      = F.normalize(clip_embeddings,   p=2, dim=1)
            loss_mse    = F.mse_loss(projected_norm, target_norm)
            loss_cosine = 1.0 - F.cosine_similarity(projected_norm, target_norm, dim=1).mean()

        # Combined loss with adjustable weights
        loss = mse_weight * loss_mse + cosine_weight * loss_cosine

        # Optional: Orthogonality regularization
        if ortho_weight > 0:
            pred_centered = projected_norm - projected_norm.mean(dim=0, keepdim=True)
            correlation = torch.mm(pred_centered.t(), pred_centered) / batch_size
            identity = torch.eye(correlation.shape[0], device=device)
            loss_ortho = F.mse_loss(correlation, identity)
            loss = loss + ortho_weight * loss_ortho
        else:
            loss_ortho = torch.tensor(0.0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute cosine similarity metric
        with torch.no_grad():
            if clip_embeddings.dim() == 4:
                cosine_sim = (projected_norm * target_norm).sum(dim=1).mean()
            else:
                cosine_sim = F.cosine_similarity(projected_norm, target_norm, dim=1).mean()

        # Update metrics
        epoch_loss += loss.item()
        epoch_loss_mse += loss_mse.item()
        epoch_loss_cosine += loss_cosine.item()
        epoch_cosine_sim += cosine_sim.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{loss_mse.item():.4f}',
            'cosine_loss': f'{loss_cosine.item():.4f}',
            'cosine_sim': f'{cosine_sim.item():.4f}'
        })

    # Average metrics
    avg_loss = epoch_loss / num_batches
    avg_loss_mse = epoch_loss_mse / num_batches
    avg_loss_cosine = epoch_loss_cosine / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    return avg_loss, avg_loss_mse, avg_loss_cosine, avg_cosine


def validate_epoch(projector, dataloader, device, top_k=None, ortho_weight=0.0, mse_weight=0.0, cosine_weight=1.0):
    """Validate for one epoch with weighted combination of MSE and cosine loss."""
    projector.eval()

    epoch_loss = 0.0
    epoch_loss_mse = 0.0
    epoch_loss_cosine = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for prototype_sims, clip_embeddings, _, _ in pbar:
            prototype_sims = prototype_sims.to(device)
            clip_embeddings = clip_embeddings.to(device)
            batch_size = prototype_sims.shape[0]

            # Apply top-k sparse selection if specified
            if top_k is not None:
                prototype_sims = select_top_k_prototypes(prototype_sims, k=top_k)

            # Project
            projected_spatial = projector(prototype_sims)

            # Normalize and compute losses — branch on target shape
            if clip_embeddings.dim() == 4:
                # Spatial target (B, 1024, 14, 14): compare per-location, no pooling
                projected_norm = F.normalize(projected_spatial, p=2, dim=1)
                target_norm    = F.normalize(clip_embeddings,   p=2, dim=1)
                loss_mse    = F.mse_loss(projected_norm, target_norm)
                loss_cosine = 1.0 - (projected_norm * target_norm).sum(dim=1).mean()
            else:
                # Global target (B, 1024): pool then compare
                projected_global = F.adaptive_avg_pool2d(projected_spatial, (1, 1)).squeeze(-1).squeeze(-1)
                projected_norm   = F.normalize(projected_global,  p=2, dim=1)
                target_norm      = F.normalize(clip_embeddings,   p=2, dim=1)
                loss_mse    = F.mse_loss(projected_norm, target_norm)
                loss_cosine = 1.0 - F.cosine_similarity(projected_norm, target_norm, dim=1).mean()

            # Combined loss with adjustable weights
            loss = mse_weight * loss_mse + cosine_weight * loss_cosine

            # Optional: Orthogonality regularization
            if ortho_weight > 0:
                pred_centered = projected_norm - projected_norm.mean(dim=0, keepdim=True)
                correlation = torch.mm(pred_centered.t(), pred_centered) / batch_size
                identity = torch.eye(correlation.shape[0], device=device)
                loss_ortho = F.mse_loss(correlation, identity)
                loss = loss + ortho_weight * loss_ortho

            # Cosine similarity
            if clip_embeddings.dim() == 4:
                cosine_sim = (projected_norm * target_norm).sum(dim=1).mean()
            else:
                cosine_sim = F.cosine_similarity(projected_norm, target_norm, dim=1).mean()

            epoch_loss += loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_cosine += loss_cosine.item()
            epoch_cosine_sim += cosine_sim.item()
            num_batches += 1

            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_cosine': f'{cosine_sim.item():.4f}'
            })

    avg_loss = epoch_loss / num_batches
    avg_loss_mse = epoch_loss_mse / num_batches
    avg_loss_cosine = epoch_loss_cosine / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    return avg_loss, avg_loss_mse, avg_loss_cosine, avg_cosine


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

    # Load cached datasets
    print("\n" + "=" * 60)
    print("LOADING CACHED FEATURES")
    print("=" * 60)

    print(f"Cache root: {args.cache_root}")
    print(f"In-memory loading: {args.in_memory}")

    clip_cache_root = args.clip_cache_root  # None → falls back to cache_root

    train_dataset = load_cached_train_dataset(
        cache_root=args.cache_root,
        clip_cache_root=clip_cache_root,
        top_k=None,             # raw features; top_k applied in training loop only
        in_memory=args.in_memory
    )

    val_dataset = load_cached_val_dataset(
        cache_root=args.cache_root,
        clip_cache_root=clip_cache_root,
        top_k=None,
        in_memory=args.in_memory
    )

    print(f"[OK] Train dataset: {len(train_dataset)} samples")
    print(f"[OK] Val dataset: {len(val_dataset)} samples")
    print(f"  Feature shapes: {train_dataset.get_feature_shapes()}")

    # Create dataloaders
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

    print(f"[OK] Batch size: {args.batch_size}")
    print("=" * 60)

    # Calculate sparsity ratio for initialization (improved architecture)
    sparsity_ratio = 1.0 - (args.top_k / args.num_prototypes) if args.top_k else 0.0

    # Create projector based on architecture type
    print(f"\nCreating {args.projector_type} projector...")
    if args.projector_type == 'improved':
        projector = ImprovedSpatialMSEProjector(
            num_prototypes=args.num_prototypes,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_channels,
            encoder_dim=args.encoder_dim,
            dropout=args.dropout,
            use_attention=args.use_attention,
            sparsity_ratio=sparsity_ratio
        ).to(device)
        print(f"[OK] ImprovedSpatialMSEProjector created")
        print(f"  Architecture features:")
        print(f"    - Sparse-aware encoding (sparsity={sparsity_ratio:.3f})")
        print(f"    - Multi-scale residual blocks")
        print(f"    - Spatial attention: {args.use_attention}")
    else:
        projector = SpatialMSEProjector(
            num_prototypes=args.num_prototypes,
            output_dim=args.output_dim,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout
        ).to(device)
        print(f"[OK] SpatialMSEProjector (baseline) created")

    print(f"  Trainable parameters: {sum(p.numel() for p in projector.parameters() if p.requires_grad):,}")

    # Optimizer and scheduler
    if args.projector_type == 'improved':
        # Layer-wise learning rates for improved architecture
        param_groups = projector.get_named_parameter_groups()
        optimizer = AdamW([
            {
                'params': param_groups['sparse_encoder'],
                'lr': args.lr * args.encoder_lr_mult,
                'name': 'sparse_encoder'
            },
            {
                'params': param_groups['blocks'],
                'lr': args.lr,
                'name': 'blocks'
            },
            {
                'params': param_groups['refinement'],
                'lr': args.lr * args.refinement_lr_mult,
                'name': 'refinement'
            }
        ], weight_decay=args.weight_decay)

        # Warmup + cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

        print(f"[OK] Optimizer: AdamW with layer-wise learning rates")
        print(f"    - Sparse encoder LR: {args.lr * args.encoder_lr_mult:.6f} ({args.encoder_lr_mult}× base)")
        print(f"    - Blocks LR: {args.lr:.6f} (1.0× base)")
        print(f"    - Refinement LR: {args.lr * args.refinement_lr_mult:.6f} ({args.refinement_lr_mult}× base)")
        print(f"    - Weight decay: {args.weight_decay}")
        print(f"[OK] Scheduler: Warmup ({args.warmup_epochs} epochs) + CosineAnnealing")
    else:
        # Standard Adam + cosine annealing for baseline
        optimizer = Adam(
            projector.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        print(f"[OK] Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
        print(f"[OK] Scheduler: CosineAnnealingLR")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"STARTING TRAINING")
    print(f"{'=' * 60}")
    print(f"Loss configuration:")
    print(f"  MSE weight: {args.mse_weight}")
    print(f"  Cosine weight: {args.cosine_weight}")
    if args.ortho_weight > 0:
        print(f"  Orthogonality weight: {args.ortho_weight}")
    print()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_cosines = []
    val_cosines = []
    learning_rates = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        learning_rates.append(current_lr)

        # Train
        train_loss, train_loss_mse, train_loss_cosine, train_cosine = train_epoch(
            projector, train_loader, optimizer, device,
            top_k=args.top_k,
            augment=args.augment,
            aug_zero_prob=args.aug_zero_prob,
            ortho_weight=args.ortho_weight,
            mse_weight=args.mse_weight,
            cosine_weight=args.cosine_weight
        )

        # Validate
        val_loss, val_loss_mse, val_loss_cosine, val_cosine = validate_epoch(
            projector, val_loader, device,
            top_k=args.top_k,
            ortho_weight=args.ortho_weight,
            mse_weight=args.mse_weight,
            cosine_weight=args.cosine_weight
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f} (MSE: {train_loss_mse:.4f}, Cosine: {train_loss_cosine:.4f}) | Cosine Sim: {train_cosine:.4f}")
        print(f"Val Loss: {val_loss:.4f} (MSE: {val_loss_mse:.4f}, Cosine: {val_loss_cosine:.4f}) | Cosine Sim: {val_cosine:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_cosines.append(train_cosine)
        val_cosines.append(val_cosine)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, 'best.pt')
            projector.save(save_path)
            print(f"[OK] Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
            projector.save(save_path)

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, 'final.pt')
    projector.save(final_path)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(train_losses, label='Train', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Combined Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cosine Similarity
    axes[0, 1].plot(train_cosines, label='Train', linewidth=2)
    axes[0, 1].plot(val_cosines, label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Cosine Similarity (Higher = Better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(learning_rates, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Loss (log scale)
    axes[1, 1].semilogy(train_losses, label='Train', linewidth=2)
    axes[1, 1].semilogy(val_losses, label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Training Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Cached Feature Training (top_k={args.top_k}, lr={args.lr})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved training curves")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best.pt')}")

    # Close datasets (close HDF5 files)
    train_dataset.close()
    val_dataset.close()


if __name__ == '__main__':
    main()
