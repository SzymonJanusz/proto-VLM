#!/usr/bin/env python
"""
Train Prototype Projector

Trains a projection head that maps sparse spatial prototype similarities (200, 14, 14)
from ProtoPNet to CLIP embedding space (512-dim) using contrastive learning.

This allows generating CLIP-compatible embeddings from prototype activations
for use with text decoders like ClipCap.

Key Features:
- 100 epochs training (10x longer than CNN projector)
- Sparse top-k prototype selection (configurable k)
- CLIP ViT baseline comparison during validation
- ClipCap caption generation every 25 epochs
- Training curve visualization
- Comprehensive metrics tracking

Usage:
    python scripts/train_prototype_projector.py \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --data_root imagenet_tiny \
        --output_dir checkpoints/prototype_projector \
        --top_k 1 \
        --epochs 100 \
        --batch_size 128 \
        --evaluate_captions \
        --clipcap_model pretrained_checkpoints/clipcap/clipcap_coco.pt
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
import clip

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.prototype_projector import (
    SpatialCNNProjector,
    HierarchicalPoolingProjector,
    create_projector
)
from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.utils.prototype_selection import select_top_k_prototypes, compute_sparsity_statistics
from ProtoPNet.utils.text_decoders import ClipCapDecoder
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_train_transforms, get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Prototype Projector for CLIP space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model & checkpoints
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                       help='Path to trained ProtoPNet checkpoint (for frozen ProtoPNet)')
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
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')

    # Prototype selection
    parser.add_argument('--top_k', type=int, default=1,
                       help='Number of top prototypes to select (sparse activation)')

    # Model architecture
    parser.add_argument('--projector_type', type=str, default='spatial_cnn',
                       choices=['spatial_cnn', 'hierarchical', 'direct_flatten', 'attention'],
                       help='Type of projector architecture (spatial_cnn=CNN, hierarchical=Pooling)')
    parser.add_argument('--num_prototypes', type=int, default=200,
                       help='Number of prototypes')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='CLIP embedding dimension')
    parser.add_argument('--hidden_channels', type=int, default=512,
                       help='Hidden channels in spatial CNN (for spatial_cnn)')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden dimension in MLP (for hierarchical)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')

    # Caption evaluation
    parser.add_argument('--evaluate_captions', action='store_true',
                       help='Generate ClipCap captions every 25 epochs')
    parser.add_argument('--clipcap_model', type=str,
                       default='pretrained_checkpoints/clipcap/clipcap_coco.pt',
                       help='Path to pretrained ClipCap model')
    parser.add_argument('--num_caption_samples', type=int, default=10,
                       help='Number of validation images for caption generation')
    parser.add_argument('--captions_per_image', type=int, default=5,
                       help='Number of captions to generate per image')

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


def extract_prototype_similarities(images, protopnet_model, top_k=1):
    """
    Extract sparse spatial prototype similarities.

    Args:
        images: (B, 3, 224, 224) - Batch of images
        protopnet_model: ProtoCLIP model (frozen)
        top_k: Number of top prototypes to select

    Returns:
        sparse_similarities: (B, 200, 14, 14) - Sparse spatial similarities
    """
    with torch.no_grad():
        # Get spatial similarities from ProtoPNet
        # ProtoCLIP's image_encoder is ProtoPNetEncoder
        _, spatial_sims = protopnet_model.image_encoder.forward(
            images,
            return_similarities=True
        )  # (B, 200, 14, 14)

        # Apply top-k selection
        sparse_sims = select_top_k_prototypes(spatial_sims, k=top_k)

    return sparse_sims


def train_epoch(projector, protopnet_model, text_encoder, dataloader, optimizer, temperature, top_k, device):
    """Train for one epoch."""
    projector.train()
    protopnet_model.eval()  # Keep ProtoPNet frozen
    text_encoder.eval()  # Keep text encoder frozen

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Dataset returns (image, caption) tuple
        images, captions = batch
        images = images.to(device)

        # Extract prototype similarities (frozen ProtoPNet)
        prototype_sims = extract_prototype_similarities(images, protopnet_model, top_k=top_k)

        # Project to CLIP space
        image_embeds = projector(prototype_sims)  # (B, 512)

        # Get text embeddings (use Hugging Face CLIP - same as ProtoPNet training)
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
def validate_with_baseline(
    projector,
    protopnet_model,
    text_encoder,
    clip_vit_model,
    dataloader,
    temperature,
    top_k,
    device
):
    """
    Validate prototype projector AND CLIP ViT baseline.

    Returns:
        results: dict with validation metrics for both methods
    """
    projector.eval()
    protopnet_model.eval()
    text_encoder.eval()
    clip_vit_model.eval()

    # Initialize accumulators
    prototype_total_loss = 0.0
    prototype_correct_i2t = 0
    prototype_correct_t2i = 0

    clip_vit_total_loss = 0.0
    clip_vit_correct_i2t = 0
    clip_vit_correct_t2i = 0

    total = 0
    all_proto_embeds = []
    all_clip_embeds = []

    # Preprocessing constants for CLIP ViT (compute once)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    pbar = tqdm(dataloader, desc='Validating')
    for batch in pbar:
        # Dataset returns (image, caption) tuple
        images, captions = batch
        images = images.to(device)

        # === Preprocessing for CLIP ViT ===
        # Renormalize images from ImageNet to CLIP normalization
        # Denormalize from ImageNet
        images_denorm = images * imagenet_std + imagenet_mean
        # Renormalize for CLIP
        clip_images = (images_denorm - clip_mean) / clip_std

        # === Prototype Projector Path ===
        # Extract prototype similarities
        prototype_sims = extract_prototype_similarities(images, protopnet_model, top_k=top_k)

        # Project to CLIP space
        prototype_embeds = projector(prototype_sims)  # (B, 512)

        # === CLIP ViT Baseline Path ===
        # Extract CLIP ViT embeddings with proper CLIP preprocessing
        clip_embeds = clip_vit_model.encode_image(clip_images)  # (B, 512)
        clip_embeds = F.normalize(clip_embeds.float(), p=2, dim=-1)  # Ensure float32 and normalized

        # === Text Embeddings (separate for each method) ===
        # Prototype method: use Hugging Face CLIP (same as ProtoPNet training)
        protoclip_text_embeds = text_encoder.encode_text(captions)  # (B, 512)

        # CLIP ViT baseline: use OpenAI CLIP (for comparison)
        clip_text_tokens = clip.tokenize(captions, truncate=True).to(device)
        clip_text_embeds = clip_vit_model.encode_text(clip_text_tokens)  # (B, 512)
        clip_text_embeds = F.normalize(clip_text_embeds.float(), p=2, dim=-1)

        # === Compute losses ===
        # Prototype method uses Hugging Face CLIP text
        prototype_loss = clip_contrastive_loss(prototype_embeds, protoclip_text_embeds, temperature)
        # CLIP ViT baseline uses OpenAI CLIP text
        clip_vit_loss = clip_contrastive_loss(clip_embeds, clip_text_embeds, temperature)

        prototype_total_loss += prototype_loss.item()
        clip_vit_total_loss += clip_vit_loss.item()

        # === Compute retrieval accuracy ===
        batch_size = prototype_embeds.shape[0]

        # Prototype projector accuracy (use Hugging Face CLIP text)
        proto_logits = (prototype_embeds @ protoclip_text_embeds.T) / temperature  # (B, B)
        pred_i2t = proto_logits.argmax(dim=1)
        pred_t2i = proto_logits.T.argmax(dim=1)
        prototype_correct_i2t += (pred_i2t == torch.arange(batch_size, device=device)).sum().item()
        prototype_correct_t2i += (pred_t2i == torch.arange(batch_size, device=device)).sum().item()

        # CLIP ViT accuracy (use OpenAI CLIP text)
        clip_logits = (clip_embeds @ clip_text_embeds.T) / temperature  # (B, B)
        pred_i2t = clip_logits.argmax(dim=1)
        pred_t2i = clip_logits.T.argmax(dim=1)
        clip_vit_correct_i2t += (pred_i2t == torch.arange(batch_size, device=device)).sum().item()
        clip_vit_correct_t2i += (pred_t2i == torch.arange(batch_size, device=device)).sum().item()

        total += batch_size

        # Collect embeddings for similarity analysis
        all_proto_embeds.append(prototype_embeds.cpu())
        all_clip_embeds.append(clip_embeds.cpu())

    # Compute averages
    num_batches = len(dataloader)
    prototype_avg_loss = prototype_total_loss / num_batches
    prototype_acc_i2t = 100.0 * prototype_correct_i2t / total
    prototype_acc_t2i = 100.0 * prototype_correct_t2i / total

    clip_vit_avg_loss = clip_vit_total_loss / num_batches
    clip_vit_acc_i2t = 100.0 * clip_vit_correct_i2t / total
    clip_vit_acc_t2i = 100.0 * clip_vit_correct_t2i / total

    # === Embedding Similarity Analysis ===
    all_proto_embeds = torch.cat(all_proto_embeds, dim=0)  # (N, 512)
    all_clip_embeds = torch.cat(all_clip_embeds, dim=0)  # (N, 512)

    # Cosine similarity between corresponding embeddings
    cosine_sims = (all_proto_embeds * all_clip_embeds).sum(dim=1)  # (N,)

    results = {
        'prototype_loss': prototype_avg_loss,
        'prototype_i2t_acc': prototype_acc_i2t,
        'prototype_t2i_acc': prototype_acc_t2i,
        'clip_vit_loss': clip_vit_avg_loss,
        'clip_vit_i2t_acc': clip_vit_acc_i2t,
        'clip_vit_t2i_acc': clip_vit_acc_t2i,
        'embedding_cosine_mean': cosine_sims.mean().item(),
        'embedding_cosine_std': cosine_sims.std().item(),
        'embedding_cosine_min': cosine_sims.min().item(),
        'embedding_cosine_max': cosine_sims.max().item(),
    }

    return results


@torch.no_grad()
def generate_validation_captions(
    projector,
    protopnet_model,
    clip_vit_model,
    clipcap_decoder,
    val_dataset,
    num_samples=10,
    captions_per_image=5,
    top_k=1,
    device='cuda'
):
    """
    Generate captions for validation images using both methods.

    Args:
        projector: Prototype projector
        protopnet_model: ProtoCLIP model (frozen)
        clip_vit_model: CLIP ViT model (frozen)
        clipcap_decoder: ClipCap decoder
        val_dataset: Validation dataset
        num_samples: Number of validation images to caption
        captions_per_image: Number of captions to generate per image
        top_k: Top-k prototype selection
        device: Device

    Returns:
        captions_dict: Dictionary with image paths and captions from both methods
    """
    projector.eval()
    protopnet_model.eval()
    clip_vit_model.eval()

    # Select random validation images
    indices = torch.randperm(len(val_dataset))[:num_samples].tolist()

    results = {
        'image_paths': [],
        'prototype_captions': [],
        'clip_vit_captions': [],
    }

    print(f"\nGenerating captions for {num_samples} validation images...")

    # Preprocessing constants for CLIP ViT
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    for idx in tqdm(indices, desc='Captioning'):
        image, caption = val_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # === Prototype Projector Path ===
        prototype_sims = extract_prototype_similarities(image_tensor, protopnet_model, top_k=top_k)
        proto_embed = projector(prototype_sims).squeeze(0)  # (512,)

        # === CLIP ViT Path (with proper CLIP preprocessing) ===
        # Renormalize for CLIP
        image_denorm = image_tensor * imagenet_std + imagenet_mean
        clip_image = (image_denorm - clip_mean) / clip_std

        clip_embed = clip_vit_model.encode_image(clip_image)
        clip_embed = F.normalize(clip_embed.float(), p=2, dim=-1).squeeze(0)  # (512,)

        # === Generate Captions ===
        proto_captions = clipcap_decoder.decode(proto_embed, num_captions=captions_per_image)
        clip_captions = clipcap_decoder.decode(clip_embed, num_captions=captions_per_image)

        # Store results
        results['image_paths'].append(str(idx))  # Store index as path
        results['prototype_captions'].append(proto_captions)
        results['clip_vit_captions'].append(clip_captions)

    return results


def plot_training_curves(history, output_dir):
    """
    Plot training curves after training completes.

    Creates 3 plots:
        1. Loss curves (train + val, both methods)
        2. Accuracy curves (I2T + T2I, both methods)
        3. Embedding similarity over epochs
    """
    epochs = [h['epoch'] for h in history]

    # Extract metrics
    train_loss = [h['train_loss'] for h in history]
    proto_val_loss = [h['prototype_val_loss'] for h in history]
    clip_val_loss = [h['clip_vit_val_loss'] for h in history]

    proto_i2t_acc = [h['prototype_i2t_acc'] for h in history]
    proto_t2i_acc = [h['prototype_t2i_acc'] for h in history]
    clip_i2t_acc = [h['clip_vit_i2t_acc'] for h in history]
    clip_t2i_acc = [h['clip_vit_t2i_acc'] for h in history]

    embed_cosine_mean = [h['embedding_cosine_mean'] for h in history]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss (Prototype)', linewidth=2)
    axes[0].plot(epochs, proto_val_loss, label='Val Loss (Prototype)', linewidth=2)
    axes[0].plot(epochs, clip_val_loss, label='Val Loss (CLIP ViT)', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[1].plot(epochs, proto_i2t_acc, label='Prototype I2T', linewidth=2)
    axes[1].plot(epochs, proto_t2i_acc, label='Prototype T2I', linewidth=2)
    axes[1].plot(epochs, clip_i2t_acc, label='CLIP ViT I2T', linewidth=2, linestyle='--')
    axes[1].plot(epochs, clip_t2i_acc, label='CLIP ViT T2I', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Retrieval Accuracy (%)', fontsize=12)
    axes[1].set_title('Retrieval Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Embedding similarity
    axes[2].plot(epochs, embed_cosine_mean, label='Mean Cosine Similarity', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2].set_title('Prototype vs CLIP ViT Embedding Similarity', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {os.path.join(output_dir, 'training_curves.png')}")
    plt.close()


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


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("PROTOTYPE PROJECTOR TRAINING")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Backbone checkpoint: {args.backbone_checkpoint}")
    print(f"Projector type: {args.projector_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Top-k prototypes: {args.top_k}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load models
    print("Loading models...")
    device = args.device

    # 1. Load frozen ProtoPNet
    protopnet_model = load_protopnet_model(args.backbone_checkpoint, device)

    # 2. Load frozen CLIP text encoder (from ProtoPNet)
    text_encoder = protopnet_model.text_encoder
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    # 3. Load CLIP ViT for baseline
    print("Loading CLIP ViT-B/32 for baseline...")
    clip_vit_model, _ = clip.load("ViT-B/32", device=device)
    clip_vit_model.eval()
    for param in clip_vit_model.parameters():
        param.requires_grad = False
    print("  CLIP ViT loaded and frozen")

    # 4. Create prototype projector (trainable)
    print(f"Creating {args.projector_type.upper()} Projector...")

    # Prepare kwargs based on projector type
    projector_kwargs = {
        'num_prototypes': args.num_prototypes,
        'embedding_dim': args.embedding_dim,
        'dropout': args.dropout
    }

    if args.projector_type == 'spatial_cnn':
        projector_kwargs['hidden_channels'] = args.hidden_channels
    elif args.projector_type == 'hierarchical':
        projector_kwargs['hidden_dim'] = args.hidden_dim

    projector = create_projector(args.projector_type, **projector_kwargs).to(device)
    print(f"  Projector created: {sum(p.numel() for p in projector.parameters())} parameters")

    # 5. Load ClipCap decoder (if evaluating captions)
    clipcap_decoder = None
    if args.evaluate_captions:
        print(f"Loading ClipCap from: {args.clipcap_model}")
        clipcap_decoder = ClipCapDecoder(
            model_path=args.clipcap_model,
            device=device
        )
        print("  ClipCap loaded")

    # Load datasets
    print("\nLoading datasets...")
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
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")

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

    # Optimizer and scheduler
    optimizer = Adam(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    history = []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            projector, protopnet_model, text_encoder,
            train_loader, optimizer, args.temperature, args.top_k, device
        )

        # Validate
        val_results = validate_with_baseline(
            projector, protopnet_model, text_encoder, clip_vit_model,
            val_loader, args.temperature, args.top_k, device
        )

        # Learning rate step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Prototype Val Loss: {val_results['prototype_loss']:.4f}")
        print(f"  CLIP ViT Val Loss: {val_results['clip_vit_loss']:.4f}")
        print(f"  Prototype I2T/T2I: {val_results['prototype_i2t_acc']:.2f}% / {val_results['prototype_t2i_acc']:.2f}%")
        print(f"  CLIP ViT I2T/T2I: {val_results['clip_vit_i2t_acc']:.2f}% / {val_results['clip_vit_t2i_acc']:.2f}%")
        print(f"  Embedding Similarity: {val_results['embedding_cosine_mean']:.4f} ± {val_results['embedding_cosine_std']:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            projector.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

            # Generate captions
            if args.evaluate_captions and clipcap_decoder is not None:
                captions = generate_validation_captions(
                    projector, protopnet_model, clip_vit_model, clipcap_decoder,
                    val_dataset, num_samples=args.num_caption_samples,
                    captions_per_image=args.captions_per_image,
                    top_k=args.top_k, device=device
                )
                caption_path = os.path.join(args.output_dir, f'captions_epoch_{epoch}.json')
                with open(caption_path, 'w') as f:
                    json.dump(captions, f, indent=2)
                print(f"  Saved captions: {caption_path}")

        # Save best model
        if val_results['prototype_loss'] < best_val_loss:
            best_val_loss = val_results['prototype_loss']
            best_path = os.path.join(args.output_dir, 'best.pt')
            projector.save(best_path)
            print(f"  ✓ New best model! Val loss: {best_val_loss:.4f}")

        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'prototype_val_loss': val_results['prototype_loss'],
            'prototype_i2t_acc': val_results['prototype_i2t_acc'],
            'prototype_t2i_acc': val_results['prototype_t2i_acc'],
            'clip_vit_val_loss': val_results['clip_vit_loss'],
            'clip_vit_i2t_acc': val_results['clip_vit_i2t_acc'],
            'clip_vit_t2i_acc': val_results['clip_vit_t2i_acc'],
            'embedding_cosine_mean': val_results['embedding_cosine_mean'],
            'embedding_cosine_std': val_results['embedding_cosine_std'],
            'learning_rate': current_lr
        })

        # Save history
        history_path = os.path.join(args.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final prototype val loss: {history[-1]['prototype_val_loss']:.4f}")
    print(f"Final CLIP ViT val loss: {history[-1]['clip_vit_val_loss']:.4f}")

    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(history, args.output_dir)

    print("\nAll outputs saved to:", args.output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
