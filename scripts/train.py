"""
Main training script for ProtoCLIP with 3-stage training.

This script implements the full training pipeline:
    Stage 1 (Warmup): Train backbone + prototypes (text encoder frozen)
    Stage 2 (Projection): Project prototypes to nearest training patches
    Stage 3 (Fine-tuning): Fine-tune projection head (backbone + prototypes frozen)

Usage:
    # Train with pretrained Proto-CLIP initialization
    python scripts/train.py \
        --imagenet_root /path/to/imagenet \
        --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
        --use_subset \
    --subset_samples 1000down0

    # Train from scratch (random initialization)
    python scripts/train.py \
        --imagenet_root /path/to/imagenet \
        --use_subset

    # Full ImageNet training with pretrained init
    python scripts/train.py \
        --imagenet_root /path/to/imagenet \
        --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
        --batch_size 128 \
        --num_workers 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.training.trainer import ProtoCLIPTrainer
from ProtoPNet.training.losses import CombinedLoss
from ProtoPNet.data.imagenet_dataset import create_imagenet_loaders
from ProtoPNet.data.caption_generator import (
    ImageNetCaptionGenerator,
    download_imagenet_class_mapping
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ProtoCLIP with 3-stage training'
    )

    # Data arguments
    parser.add_argument('--imagenet_root', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use subset for faster experimentation')
    parser.add_argument('--subset_samples', type=int, default=10000,
                        help='Number of samples in subset (if use_subset=True)')
    parser.add_argument('--class_mapping_file', type=str,
                        default='./data/imagenet_classes.json',
                        help='Path to ImageNet class mapping file')

    # Model arguments
    parser.add_argument('--num_prototypes', type=int, default=200,
                        help='Number of prototypes')
    parser.add_argument('--pretrained_protoclip', type=str, default=None,
                        help='Path to pretrained Proto-CLIP checkpoints')
    parser.add_argument('--sampling_method', type=str, default='kmeans',
                        choices=['kmeans', 'random', 'first'],
                        help='Prototype sampling method for pretrained init')
    parser.add_argument('--pooling_mode', type=str, default='max',
                        choices=['max', 'attention'],
                        help='Pooling mode for prototype activations')

    # Training arguments - Stage 1 (Warmup)
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (Stage 1)')
    parser.add_argument('--warmup_lr_backbone', type=float, default=None,
                        help='Learning rate for backbone in warmup (default: 1e-4 random, 5e-5 pretrained)')
    parser.add_argument('--warmup_lr_prototypes', type=float, default=None,
                        help='Learning rate for prototypes in warmup (default: 1e-3 random, 5e-4 pretrained)')

    # Training arguments - Stage 3 (Fine-tuning)
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs (Stage 3)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--unfreeze_text_encoder', action='store_true',
                        help='Unfreeze text encoder during fine-tuning (may hurt zero-shot)')

    # Loss weights
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--lambda_clustering', type=float, default=0.01,
                        help='Weight for clustering loss')
    parser.add_argument('--lambda_activation', type=float, default=0.01,
                        help='Weight for activation loss')

    # Optimization arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Skip stages (for resuming)
    parser.add_argument('--skip_warmup', action='store_true',
                        help='Skip Stage 1 (warmup)')
    parser.add_argument('--skip_projection', action='store_true',
                        help='Skip Stage 2 (projection)')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip Stage 3 (fine-tuning)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')

    args = parser.parse_args()

    # Auto-set learning rates based on initialization method
    if args.warmup_lr_backbone is None:
        args.warmup_lr_backbone = 5e-5 if args.pretrained_protoclip else 1e-4
    if args.warmup_lr_prototypes is None:
        args.warmup_lr_prototypes = 5e-4 if args.pretrained_protoclip else 1e-3

    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args):
    """Create ProtoCLIP model with optional pretrained initialization"""
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = ProtoCLIP(
        num_prototypes=args.num_prototypes,
        image_backbone='resnet50',
        text_model="openai/clip-vit-base-patch32",
        embedding_dim=512,
        freeze_text_encoder=True,
        temperature=0.07,
        pooling_mode=args.pooling_mode,
        pretrained_protoclip_path=args.pretrained_protoclip,
        protoclip_sampling_method=args.sampling_method
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    print(f"  Prototypes: {args.num_prototypes}")
    print(f"  Pooling mode: {args.pooling_mode}")
    print(f"  Pretrained init: {args.pretrained_protoclip is not None}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def create_data_loaders(args):
    """Create ImageNet train and validation data loaders"""
    print("\n" + "=" * 70)
    print("Creating Data Loaders")
    print("=" * 70)

    # Download class mapping if needed
    class_mapping_file = Path(args.class_mapping_file)
    if not class_mapping_file.exists():
        print(f"\nClass mapping not found at {class_mapping_file}")
        print("Downloading ImageNet class names...")
        download_imagenet_class_mapping(str(class_mapping_file))

    # Create caption generator
    caption_generator = ImageNetCaptionGenerator(
        class_mapping_file=str(class_mapping_file) if class_mapping_file.exists() else None
    )

    # Create data loaders
    train_loader, val_loader = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_subset=args.use_subset,
        subset_samples=args.subset_samples,
        caption_generator=caption_generator
    )

    return train_loader, val_loader


def stage1_warmup(model, train_loader, val_loader, args):
    """
    Stage 1: Warmup Training

    Train backbone + prototypes with frozen text encoder.
    Use auxiliary losses (clustering + activation).
    """
    print("\n" + "=" * 70)
    print("STAGE 1: WARMUP TRAINING")
    print("=" * 70)
    print(f"Epochs: {args.warmup_epochs}")
    print(f"LR backbone: {args.warmup_lr_backbone}")
    print(f"LR prototypes: {args.warmup_lr_prototypes}")

    # Create optimizer
    optimizer = AdamW([
        {
            'params': model.image_encoder.backbone.parameters(),
            'lr': args.warmup_lr_backbone
        },
        {
            'params': model.image_encoder.prototype_layer.parameters(),
            'lr': args.warmup_lr_prototypes
        },
        {
            'params': model.image_encoder.projection_head.parameters(),
            'lr': args.warmup_lr_prototypes
        }
    ])

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.warmup_epochs,
        eta_min=args.warmup_lr_backbone / 10
    )

    # Create loss function with auxiliary losses
    loss_fn = CombinedLoss(
        lambda_contrastive=args.lambda_contrastive,
        lambda_clustering=args.lambda_clustering,
        lambda_activation=args.lambda_activation
    )

    # Create trainer
    trainer = ProtoCLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.warmup_epochs):
        print(f"\n{'=' * 70}")
        print(f"Warmup Epoch {epoch + 1}/{args.warmup_epochs}")
        print(f"{'=' * 70}")

        # Train
        train_loss, _ = trainer.train_epoch(return_similarities=True)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss, _ = trainer.validate()
        print(f"Val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                'warmup_best.pt',
                epoch=epoch,
                stage='warmup'
            )

        # Save latest checkpoint
        trainer.save_checkpoint(
            'warmup_latest.pt',
            epoch=epoch,
            stage='warmup'
        )

    print(f"\n✓ Warmup complete! Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint
    trainer.load_checkpoint(f"{args.checkpoint_dir}/warmup_best.pt")

    return model


def stage2_projection(model, train_loader, args):
    """
    Stage 2: Prototype Projection

    Project each prototype to its nearest training patch.
    This grounds prototypes in real image regions for interpretability.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: PROTOTYPE PROJECTION")
    print("=" * 70)

    # TODO: Implement PrototypeProjector
    # For now, we'll skip this stage and add a warning
    print("\n⚠️  WARNING: Prototype projection not yet implemented!")
    print("Prototypes will remain as learned during warmup.")
    print("\nTo implement projection:")
    print("  1. Create ProtoPNet/utils/projection.py")
    print("  2. Implement PrototypeProjector class")
    print("  3. Find nearest training patches for each prototype")
    print("  4. Replace prototype vectors with patch features")

    # Placeholder for when projection.py is implemented:
    """
    from ProtoPNet.utils.projection import PrototypeProjector

    projector = PrototypeProjector(model, train_loader, device=args.device)
    new_prototypes, projection_info = projector.project_prototypes()

    # Replace prototypes with projected ones
    model.image_encoder.set_prototypes(new_prototypes)

    # Save projection info for visualization
    import pickle
    with open(f"{args.checkpoint_dir}/projection_info.pkl", 'wb') as f:
        pickle.dump(projection_info, f)

    print(f"✓ Projected {len(new_prototypes)} prototypes to training patches")
    """

    return model


def stage3_finetune(model, train_loader, val_loader, args):
    """
    Stage 3: Fine-tuning

    Fine-tune projection head with frozen backbone and prototypes.
    Optionally unfreeze text encoder for task-specific adaptation.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: FINE-TUNING")
    print("=" * 70)
    print(f"Epochs: {args.finetune_epochs}")
    print(f"LR: {args.finetune_lr}")
    print(f"Unfreeze text encoder: {args.unfreeze_text_encoder}")

    # Freeze backbone and prototypes
    for param in model.image_encoder.backbone.parameters():
        param.requires_grad = False
    for param in model.image_encoder.prototype_layer.parameters():
        param.requires_grad = False

    print("\n✓ Backbone frozen")
    print("✓ Prototypes frozen")

    # Optionally unfreeze text encoder
    if args.unfreeze_text_encoder:
        model.text_encoder.unfreeze()
        print("✓ Text encoder unfrozen")

    # Create optimizer (only projection head + optionally text encoder)
    params_to_optimize = [
        {'params': model.image_encoder.projection_head.parameters(), 'lr': args.finetune_lr}
    ]

    if args.unfreeze_text_encoder:
        params_to_optimize.append({
            'params': model.text_encoder.parameters(),
            'lr': args.finetune_lr / 10  # Lower LR for text encoder
        })

    optimizer = AdamW(params_to_optimize)

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.finetune_epochs,
        eta_min=args.finetune_lr / 10
    )

    # Create loss function (no auxiliary losses)
    loss_fn = CombinedLoss(
        lambda_contrastive=1.0,
        lambda_clustering=0.0,
        lambda_activation=0.0
    )

    # Create trainer
    trainer = ProtoCLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.finetune_epochs):
        print(f"\n{'=' * 70}")
        print(f"Fine-tune Epoch {epoch + 1}/{args.finetune_epochs}")
        print(f"{'=' * 70}")

        # Train
        train_loss, _ = trainer.train_epoch(return_similarities=False)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss, _ = trainer.validate()
        print(f"Val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                'finetune_best.pt',
                epoch=epoch,
                stage='finetune'
            )

        # Save latest checkpoint
        trainer.save_checkpoint(
            'finetune_latest.pt',
            epoch=epoch,
            stage='finetune'
        )

    print(f"\n✓ Fine-tuning complete! Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint
    trainer.load_checkpoint(f"{args.checkpoint_dir}/finetune_best.pt")

    return model


def main():
    """Main training pipeline"""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Print configuration
    print("=" * 70)
    print("ProtoCLIP Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"ImageNet root: {args.imagenet_root}")
    print(f"Pretrained init: {args.pretrained_protoclip or 'None (random)'}")
    print(f"Use subset: {args.use_subset}")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create model
    model = create_model(args)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Stage 1: Warmup
    if not args.skip_warmup:
        model = stage1_warmup(model, train_loader, val_loader, args)
    else:
        print("\n⏭️  Skipping Stage 1 (Warmup)")

    # Stage 2: Projection
    if not args.skip_projection:
        model = stage2_projection(model, train_loader, args)
    else:
        print("\n⏭️  Skipping Stage 2 (Projection)")

    # Stage 3: Fine-tuning
    if not args.skip_finetune:
        model = stage3_finetune(model, train_loader, val_loader, args)
    else:
        print("\n⏭️  Skipping Stage 3 (Fine-tuning)")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"\nFinal model checkpoints:")
    print(f"  - warmup_best.pt")
    print(f"  - finetune_best.pt")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on retrieval tasks: python scripts/evaluate.py")
    print(f"  2. Visualize prototypes: python scripts/visualize_prototypes.py")
    print(f"  3. Test zero-shot capabilities with custom queries")
    print("=" * 70)


if __name__ == '__main__':
    main()
