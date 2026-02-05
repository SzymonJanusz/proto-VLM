#!/usr/bin/env python
"""
Train CNN Feature Projector

Trains a projection head that maps CNN features (1024-dim) from a frozen
ResNet backbone to CLIP embedding space (512-dim) using contrastive learning.

This allows us to generate CLIP-compatible embeddings from abstract CNN features
(not prototype similarities) for use with text decoders like ClipCap.

Usage:
    python scripts/train_cnn_projector.py \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --data_root imagenet_tiny \
        --output_dir checkpoints/cnn_projector \
        --epochs 10 \
        --batch_size 128
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.cnn_feature_projector import CNNFeatureProjector
from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_train_transforms, get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN Feature Projector for CLIP space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model & checkpoints
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                       help='Path to trained ProtoPNet checkpoint (for frozen backbone)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save trained projector checkpoints')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to ImageNet root directory')
    parser.add_argument('--class_mapping', type=str, default='data/imagenet_classes.json',
                       help='Path to ImageNet class mapping JSON')
    parser.add_argument('--split_train', type=str, default='train',
                       help='Training split name')
    parser.add_argument('--split_val', type=str, default='val',
                       help='Validation split name')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')

    # Model architecture
    parser.add_argument('--feature_dim', type=int, default=1024,
                       help='CNN feature dimension (ResNet50 layer3)')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='CLIP embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden layer dimension')
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


def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Compute CLIP-style contrastive loss.

    Args:
        image_embeds: (B, D) - L2-normalized image embeddings
        text_embeds: (B, D) - L2-normalized text embeddings
        temperature: Temperature parameter for scaling logits

    Returns:
        loss: Scalar contrastive loss
    """
    batch_size = image_embeds.shape[0]

    # Compute similarity matrix (cosine similarity since embeddings are normalized)
    logits = (image_embeds @ text_embeds.T) / temperature  # (B, B)

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=image_embeds.device)

    # Cross entropy loss in both directions
    loss_i2t = F.cross_entropy(logits, labels)  # Image-to-text
    loss_t2i = F.cross_entropy(logits.T, labels)  # Text-to-image

    # Average both directions
    loss = (loss_i2t + loss_t2i) / 2.0

    return loss


def extract_cnn_features(images, backbone):
    """
    Extract CNN features from images using frozen backbone.

    Args:
        images: (B, 3, 224, 224) - Batch of images
        backbone: Frozen ResNet backbone

    Returns:
        features: (B, feature_dim) - Pooled CNN features
    """
    with torch.no_grad():
        # Extract features
        cnn_features = backbone(images)  # (B, 1024, 14, 14)

        # Global average pooling
        pooled = F.adaptive_avg_pool2d(cnn_features, (1, 1))  # (B, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 1024)

    return pooled


def train_epoch(projector, backbone, text_encoder, dataloader, optimizer, temperature, device):
    """Train for one epoch."""
    projector.train()
    backbone.eval()  # Keep backbone frozen
    text_encoder.eval()  # Keep text encoder frozen

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Dataset returns (image, caption) tuple
        images, captions = batch
        images = images.to(device)

        # Extract CNN features (frozen backbone)
        cnn_features = extract_cnn_features(images, backbone)

        # Project to CLIP space
        image_embeds = projector(cnn_features)  # (B, 512)

        # Get text embeddings (frozen text encoder)
        with torch.no_grad():
            text_embeds = text_encoder.encode_text(captions)  # (B, 512)

        # Compute contrastive loss
        loss = clip_contrastive_loss(image_embeds, text_embeds, temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(projector, backbone, text_encoder, dataloader, temperature, device):
    """Validate on validation set."""
    projector.eval()
    backbone.eval()
    text_encoder.eval()

    total_loss = 0.0
    correct_i2t = 0
    correct_t2i = 0
    total = 0

    pbar = tqdm(dataloader, desc='Validating')
    for batch in pbar:
        # Dataset returns (image, caption) tuple
        images, captions = batch
        images = images.to(device)

        # Extract CNN features
        cnn_features = extract_cnn_features(images, backbone)

        # Project to CLIP space
        image_embeds = projector(cnn_features)  # (B, 512)

        # Get text embeddings
        text_embeds = text_encoder.encode_text(captions)  # (B, 512)

        # Compute loss
        loss = clip_contrastive_loss(image_embeds, text_embeds, temperature)
        total_loss += loss.item()

        # Compute retrieval accuracy
        batch_size = image_embeds.shape[0]
        logits = (image_embeds @ text_embeds.T) / temperature  # (B, B)

        # Image-to-text accuracy
        pred_i2t = logits.argmax(dim=1)
        correct_i2t += (pred_i2t == torch.arange(batch_size, device=device)).sum().item()

        # Text-to-image accuracy
        pred_t2i = logits.T.argmax(dim=1)
        correct_t2i += (pred_t2i == torch.arange(batch_size, device=device)).sum().item()

        total += batch_size

    avg_loss = total_loss / len(dataloader)
    acc_i2t = 100.0 * correct_i2t / total
    acc_t2i = 100.0 * correct_t2i / total

    return avg_loss, acc_i2t, acc_t2i


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Training CNN Feature Projector")
    print("=" * 80)
    print(f"Backbone checkpoint: {args.backbone_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data root: {args.data_root}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load ProtoCLIP model (for frozen backbone and text encoder)
    print("Loading ProtoCLIP model...")

    # Load checkpoint
    checkpoint = torch.load(args.backbone_checkpoint, map_location=args.device, weights_only=False)

    # Infer projection_hidden_dim from checkpoint
    # projection_head.0.weight has shape (projection_hidden_dim, 200)
    state_dict = checkpoint['model_state_dict']
    projection_first_layer_weight = state_dict['image_encoder.projection_head.0.weight']
    projection_hidden_dim = projection_first_layer_weight.shape[0]  # Should be 128

    print(f"  Detected projection_hidden_dim from checkpoint: {projection_hidden_dim}")

    # Create model with correct config
    model = ProtoCLIP(
        num_prototypes=200,
        image_backbone='resnet50',
        embedding_dim=512,
        pooling_mode='max',
        freeze_text_encoder=True,
        projection_hidden_dim=projection_hidden_dim  # Use detected value
    ).to(args.device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract frozen components
    backbone = model.image_encoder.backbone
    text_encoder = model.text_encoder

    print(f"  Loaded backbone: {backbone.__class__.__name__}")
    print(f"  Loaded text encoder: {text_encoder.__class__.__name__}")

    # Freeze backbone and text encoder
    for param in backbone.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Create CNN Feature Projector
    print("\nCreating CNN Feature Projector...")
    projector = CNNFeatureProjector(
        feature_dim=args.feature_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(args.device)

    print(f"  Feature dim: {args.feature_dim}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in projector.parameters()):,}")

    # Create datasets
    print("\nLoading datasets...")
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset = ImageNetWithCaptions(
        root=args.data_root,
        split=args.split_train,
        transform=train_transforms
    )
    val_dataset = ImageNetWithCaptions(
        root=args.data_root,
        split=args.split_val,
        transform=val_transforms
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

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

    # Create optimizer and scheduler
    optimizer = Adam(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            projector, backbone, text_encoder,
            train_loader, optimizer, args.temperature, args.device
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_acc_i2t, val_acc_t2i = validate(
            projector, backbone, text_encoder,
            val_loader, args.temperature, args.device
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val I2T Acc: {val_acc_i2t:.2f}%")
        print(f"Val T2I Acc: {val_acc_t2i:.2f}%")

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'epoch_{epoch}.pt')
        projector.save(checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best.pt')
            projector.save(best_path)
            print(f"Saved best model: {best_path}")

        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc_i2t': val_acc_i2t,
            'val_acc_t2i': val_acc_t2i,
            'lr': current_lr
        })

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best.pt')}")
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
