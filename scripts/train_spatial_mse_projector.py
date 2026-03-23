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
- Supports both pretrained backbone and training from scratch

Usage:
  # Train from scratch with random initialization
  python scripts/train_spatial_mse_projector.py \
    --train_from_scratch \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_k1_scratch \
    --epochs 50 \
    --batch_size 64 \
    --top_k 1

  # Train from scratch with pretrained Proto-CLIP initialization (NEW)
  python scripts/train_spatial_mse_projector.py \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --sampling_method kmeans \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_k1_protoclip \
    --epochs 50 \
    --batch_size 64 \
    --top_k 1

  # With local checkpoint from train.py (frozen)
  python scripts/train_spatial_mse_projector.py \
    --backbone_checkpoint checkpoints/finetune_best.pt \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_k1 \
    --epochs 50 \
    --batch_size 64 \
    --top_k 1 \
    --freeze_backbone

  # Sparse (top-5 prototypes) with pretrained backbone
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
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                        help='Path to trained ProtoPNet checkpoint (frozen). If not provided, trains from scratch.')
    parser.add_argument('--pretrained_protoclip', type=str, default=None,
                        help='Path to pretrained Proto-CLIP checkpoints (16k prototypes, subsampled to 200)')
    parser.add_argument('--sampling_method', type=str, default='kmeans',
                        choices=['kmeans', 'random', 'first'],
                        help='Prototype sampling method for pretrained Proto-CLIP init')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trained projector checkpoints')
    parser.add_argument('--train_from_scratch', action='store_true',
                        help='Train from scratch (random initialization) instead of loading checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze ProtoPNet backbone (only applicable when loading checkpoint)')
    parser.add_argument('--projection_hidden_dim', type=int, default=1024,
                        help='Hidden dimension for projection head (only used when training from scratch)')

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


def create_fresh_protopnet_model(projection_hidden_dim, device, freeze=False,
                                  pretrained_protoclip_path=None, sampling_method='kmeans'):
    """Create a ProtoCLIP model with random or pretrained initialization."""
    if pretrained_protoclip_path:
        print(f"Creating ProtoCLIP model with pretrained Proto-CLIP initialization...")
        print(f"  Pretrained path: {pretrained_protoclip_path}")
        print(f"  Sampling method: {sampling_method}")
    else:
        print(f"Creating fresh ProtoCLIP model (random initialization)...")
    print(f"  projection_hidden_dim: {projection_hidden_dim}")

    model = ProtoCLIP(
        num_prototypes=200,
        image_backbone='resnet50',
        embedding_dim=512,
        pooling_mode='max',
        freeze_text_encoder=True,
        projection_hidden_dim=projection_hidden_dim,
        pretrained_protoclip_path=pretrained_protoclip_path,
        protoclip_sampling_method=sampling_method
    ).to(device)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("  ProtoCLIP created and frozen")
    else:
        model.train()
        print("  ProtoCLIP created (trainable)")

    return model


def load_protopnet_model(checkpoint_path, device, freeze=True):
    """Load ProtoCLIP model from checkpoint."""
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

    if freeze:
        model.eval()
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        print("  ProtoCLIP loaded and frozen")
    else:
        model.train()
        print("  ProtoCLIP loaded (trainable)")

    return model


def extract_prototype_similarities(images, protopnet_model, device, freeze=True):
    """
    Extract spatial prototype similarities from ProtoPNet.

    Args:
        images: (B, 3, 224, 224) - ImageNet normalized images
        protopnet_model: ProtoPNet model (frozen or trainable)
        device: Device
        freeze: Whether to extract without gradients

    Returns:
        prototype_sims: (B, 200, 14, 14) - Spatial similarity maps
    """
    if freeze:
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
    else:
        # Trainable mode - keep gradients
        if hasattr(protopnet_model, 'image_encoder'):
            features = protopnet_model.image_encoder.backbone(images)
            prototype_sims = protopnet_model.image_encoder.prototype_layer(features)
        else:
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
        device,
        freeze_protopnet=True
):
    """Train for one epoch."""
    projector.train()
    if freeze_protopnet:
        protopnet_model.eval()  # Frozen
    else:
        protopnet_model.train()  # Trainable
    clip_extractor.eval()  # Always frozen

    epoch_loss = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0

    # Track sparsity statistics
    sparsity_stats_list = []

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch[0].to(device)  # (B, 3, 224, 224)
        batch_size = images.shape[0]

        # 1. Extract CLIP final embeddings (frozen, target)
        clip_final_embeddings = clip_extractor(images)  # (B, 1024) - final embeddings
        clip_final_embeddings = clip_final_embeddings.float()  # FIX: Ensure float32

        # 2. Extract prototype similarities
        prototype_sims = extract_prototype_similarities(images, protopnet_model, device, freeze=freeze_protopnet)  # (B, 200, 14, 14)
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

        # 5. Pool projected spatial features to global embeddings
        projected_global = F.adaptive_avg_pool2d(projected_spatial, (1, 1))  # (B, 1024, 1, 1)
        projected_global = projected_global.squeeze(-1).squeeze(-1)  # (B, 1024)

        # 6. Compute global embedding loss (MSE + Cosine)
        # L2 normalize both embeddings
        projected_norm = F.normalize(projected_global, p=2, dim=1)
        target_norm = F.normalize(clip_final_embeddings, p=2, dim=1)

        # Combined loss: MSE + Cosine
        loss_mse = F.mse_loss(projected_norm, target_norm)
        loss_cosine = 1.0 - F.cosine_similarity(projected_norm, target_norm, dim=1).mean()
        loss = 0.5 * loss_mse + 0.5 * loss_cosine  # Equal weighting

        # 7. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute additional metrics
        with torch.no_grad():
            # Cosine similarity (global embeddings)
            pred_norm = F.normalize(projected_global, p=2, dim=1)  # (B, 1024)
            target_norm = F.normalize(clip_final_embeddings, p=2, dim=1)  # (B, 1024)

            # Cosine similarity averaged over batch
            cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()

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
        device,
        freeze_protopnet=True
):
    """Validate for one epoch."""
    projector.eval()
    protopnet_model.eval()  # Always eval mode for validation
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
            clip_final_embeddings = clip_extractor(images)  # (B, 1024) - final embeddings
            clip_final_embeddings = clip_final_embeddings.float()  # FIX: Ensure float32

            prototype_sims = extract_prototype_similarities(images, protopnet_model, device, freeze=True)  # Always freeze in validation
            prototype_sims = prototype_sims.float()  # FIX: Ensure float32

            # Apply sparse selection
            if top_k < num_prototypes:
                sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)
            else:
                sparse_sims = prototype_sims

            # Project
            projected_spatial = projector(sparse_sims)  # (B, 1024, 14, 14)

            # Pool to global embeddings
            projected_global = F.adaptive_avg_pool2d(projected_spatial, (1, 1))  # (B, 1024, 1, 1)
            projected_global = projected_global.squeeze(-1).squeeze(-1)  # (B, 1024)

            # Compute global embedding loss (MSE + Cosine)
            projected_norm = F.normalize(projected_global, p=2, dim=1)
            target_norm = F.normalize(clip_final_embeddings, p=2, dim=1)

            loss_mse = F.mse_loss(projected_norm, target_norm)
            loss_cosine = 1.0 - F.cosine_similarity(projected_norm, target_norm, dim=1).mean()
            loss = 0.5 * loss_mse + 0.5 * loss_cosine

            # Cosine similarity metric
            cosine_sim = F.cosine_similarity(projected_norm, target_norm, dim=1).mean()

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

    # Create or load ProtoPNet model
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    if args.backbone_checkpoint is not None:
        # Load from local checkpoint (trained via train.py)
        print("Mode: Using pretrained backbone from local checkpoint")
        freeze_mode = args.freeze_backbone
        protopnet_model = load_protopnet_model(
            checkpoint_path=args.backbone_checkpoint,
            device=device,
            freeze=freeze_mode
        )
    elif args.pretrained_protoclip is not None:
        # Initialize from pretrained Proto-CLIP repository checkpoints
        print("Mode: Training from scratch with pretrained Proto-CLIP initialization")
        protopnet_model = create_fresh_protopnet_model(
            projection_hidden_dim=args.projection_hidden_dim,
            device=device,
            freeze=False,  # Keep trainable when training from scratch
            pretrained_protoclip_path=args.pretrained_protoclip,
            sampling_method=args.sampling_method
        )
        freeze_mode = False
    else:
        # Random initialization
        print("Mode: Training from scratch (random initialization)")
        protopnet_model = create_fresh_protopnet_model(
            projection_hidden_dim=args.projection_hidden_dim,
            device=device,
            freeze=False  # Keep trainable when training from scratch
        )
        freeze_mode = False

    print(f"[OK] ProtoPNet initialized (frozen={freeze_mode})")
    print("=" * 60)

    # Create CLIP feature extractor (final embeddings for retrieval)
    print(f"\nInitializing CLIP final embedding extractor (ResNet-50 -> final)...")
    clip_extractor = CLIPSpatialExtractor(
        layer='layer3',
        device=device,
        normalize_input=True,
        extract_final=True  # Extract final embeddings (1024-dim) for retrieval
    )
    clip_extractor.eval()
    print(f"[OK] CLIP extractor initialized (extracting final embeddings)")

    # Create trainable spatial MSE projector
    print(f"\nCreating SpatialMSEProjector...")
    projector = SpatialMSEProjector(
        num_prototypes=args.num_prototypes,
        output_dim=args.output_dim,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout
    ).to(device)
    print(f"[OK] Projector created: {projector}")
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
    print(f"[OK] Loss function: {loss_fn}")

    # Optimizer and scheduler
    print("\n" + "=" * 60)
    print("OPTIMIZER CONFIGURATION")
    print("=" * 60)

    if freeze_mode:
        # Only train the projector
        optimizer = Adam(
            projector.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"Training: Projector only (ProtoPNet frozen)")
        print(f"  Projector params: {sum(p.numel() for p in projector.parameters() if p.requires_grad):,}")
    else:
        # Train both ProtoPNet and projector
        optimizer = Adam([
            {
                'params': protopnet_model.image_encoder.backbone.parameters(),
                'lr': args.lr * 0.1  # Lower LR for backbone
            },
            {
                'params': protopnet_model.image_encoder.prototype_layer.parameters(),
                'lr': args.lr
            },
            {
                'params': protopnet_model.image_encoder.projection_head.parameters(),
                'lr': args.lr
            },
            {
                'params': projector.parameters(),
                'lr': args.lr
            }
        ], weight_decay=args.weight_decay)
        print(f"Training: ProtoPNet + Projector (from scratch)")
        print(f"  Backbone params: {sum(p.numel() for p in protopnet_model.image_encoder.backbone.parameters() if p.requires_grad):,}")
        print(f"  Prototype params: {sum(p.numel() for p in protopnet_model.image_encoder.prototype_layer.parameters() if p.requires_grad):,}")
        print(f"  Projection head params: {sum(p.numel() for p in protopnet_model.image_encoder.projection_head.parameters() if p.requires_grad):,}")
        print(f"  Spatial projector params: {sum(p.numel() for p in projector.parameters() if p.requires_grad):,}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"[OK] Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"[OK] Scheduler: CosineAnnealingLR")
    print("=" * 60)

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

    print(f"[OK] Train dataset: {len(train_dataset)} images")
    print(f"[OK] Val dataset: {len(val_dataset)} images")
    print(f"[OK] Batch size: {args.batch_size}")

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"STARTING TRAINING")
    print(f"{'=' * 60}\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_cosines = []
    val_cosines = []
    learning_rates = []
    sparsity_means = []
    num_selected_prototypes = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        learning_rates.append(current_lr)

        # Train
        train_loss, train_cosine, sparsity_stats = train_epoch(
            projector, protopnet_model, clip_extractor,
            train_loader, optimizer, loss_fn,
            args.top_k, args.num_prototypes, device,
            freeze_protopnet=freeze_mode
        )

        # Validate
        val_loss, val_cosine = validate_epoch(
            projector, protopnet_model, clip_extractor,
            val_loader, loss_fn,
            args.top_k, args.num_prototypes, device,
            freeze_protopnet=freeze_mode
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train Cosine: {train_cosine:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")
        if sparsity_stats:
            print(f"Sparsity: {sparsity_stats['mean_sparsity']:.1%} | "
                  f"Selected: {sparsity_stats['mean_num_selected']:.1f} prototypes")
            sparsity_means.append(sparsity_stats['mean_sparsity'])
            num_selected_prototypes.append(sparsity_stats['mean_num_selected'])
        else:
            sparsity_means.append(1.0)  # No sparsity when top_k = num_prototypes
            num_selected_prototypes.append(args.num_prototypes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_cosines.append(train_cosine)
        val_cosines.append(val_cosine)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, 'best.pt')
            projector.save(save_path)

            # Also save ProtoPNet if training from scratch
            if not freeze_mode:
                protopnet_save_path = os.path.join(args.output_dir, 'protopnet_best.pt')
                torch.save({
                    'model_state_dict': protopnet_model.state_dict(),
                    'epoch': epoch,
                }, protopnet_save_path)
                print(f"[OK] Saved best models (val_loss={val_loss:.4f})")
            else:
                print(f"[OK] Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
            projector.save(save_path)
            if not freeze_mode:
                protopnet_save_path = os.path.join(args.output_dir, f'protopnet_epoch_{epoch + 1}.pt')
                torch.save({
                    'model_state_dict': protopnet_model.state_dict(),
                    'epoch': epoch,
                }, protopnet_save_path)

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, 'final.pt')
    projector.save(final_path)

    if not freeze_mode:
        protopnet_final_path = os.path.join(args.output_dir, 'protopnet_final.pt')
        torch.save({
            'model_state_dict': protopnet_model.state_dict(),
        }, protopnet_final_path)
        print(f"[OK] Saved final checkpoints (projector + protopnet)")

    # Plot training curves
    fig = plt.figure(figsize=(16, 10))

    # 1. MSE Loss
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('MSE Loss', fontsize=10)
    plt.title('Spatial MSE Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 2. Cosine Similarity
    plt.subplot(2, 3, 2)
    plt.plot(train_cosines, label='Train Cosine', linewidth=2)
    plt.plot(val_cosines, label='Val Cosine', linewidth=2)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Cosine Similarity', fontsize=10)
    plt.title('Cosine Similarity (Higher is Better)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 3. Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(learning_rates, linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 4. Loss comparison (log scale)
    plt.subplot(2, 3, 4)
    plt.semilogy(train_losses, label='Train Loss', linewidth=2)
    plt.semilogy(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('MSE Loss (log scale)', fontsize=10)
    plt.title('MSE Loss (Log Scale)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 5. Sparsity statistics
    plt.subplot(2, 3, 5)
    plt.plot(sparsity_means, linewidth=2, color='purple')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Sparsity Ratio', fontsize=10)
    plt.title(f'Prototype Sparsity (top_k={args.top_k})', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    # 6. Number of selected prototypes
    plt.subplot(2, 3, 6)
    plt.plot(num_selected_prototypes, linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Num Selected Prototypes', fontsize=10)
    plt.title('Active Prototypes per Image', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, args.num_prototypes + 5])

    plt.suptitle(f'Spatial MSE Projector Training (top_k={args.top_k}, lr={args.lr})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved comprehensive training curves with all metrics")

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == '__main__':
    main()
