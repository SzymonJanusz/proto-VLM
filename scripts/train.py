"""
Main training script for ProtoCLIP with 3-stage (or optional 4-stage) training.

This script implements the full training pipeline:
    Stage 1 (Warmup): Train backbone + prototypes (text encoder frozen)
    Stage 2 (Projection): Project prototypes to nearest training patches
    Stage 3 (Fine-tuning): Fine-tune projection head (backbone + prototypes frozen)
    Stage 4 (Spatial Projector, OPTIONAL): Train spatial MSE projector for CLIP spatial features

Usage:
    # Standard 3-stage training with pretrained Proto-CLIP initialization
    python scripts/train.py \
        --imagenet_root /path/to/imagenet \
        --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
        --use_subset \
        --subset_samples 10000

    # Full 4-stage training with spatial projector (sparse, top-5)
    python scripts/train.py \
        --imagenet_root /path/to/imagenet \
        --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
        --train_spatial_projector \
        --spatial_top_k 5 \
        --spatial_freeze_backbone \
        --batch_size 64

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.training.trainer import ProtoCLIPTrainer
from ProtoPNet.training.losses import CombinedLoss
from ProtoPNet.data.imagenet_dataset import create_imagenet_loaders
from ProtoPNet.data.caltech101_dataset import create_caltech101_loaders
from ProtoPNet.data.caption_generator import (
    ImageNetCaptionGenerator,
    download_imagenet_class_mapping
)
from ProtoPNet.utils.early_stopping import EarlyStopping


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ProtoCLIP with 3-stage training'
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'caltech101'],
                        help='Dataset to use for training (default: imagenet)')
    parser.add_argument('--imagenet_root', type=str, default=None,
                        help='Path to ImageNet root directory (required for --dataset imagenet)')
    parser.add_argument('--caltech_root', type=str, default='caltech101',
                        help='Path to Caltech-101 root directory (for --dataset caltech101)')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use subset for faster experimentation (ImageNet only)')
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
    parser.add_argument('--warmup_weight_decay', type=float, default=0.01,
                        help='Weight decay for warmup stage')
    parser.add_argument('--warmup_patience', type=int, default=3,
                        help='Early stopping patience for warmup (0 to disable)')
    parser.add_argument('--warmup_min_delta', type=float, default=0.0,
                        help='Minimum validation loss improvement for warmup')

    # Training arguments - Stage 3 (Fine-tuning)
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs (Stage 3)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_weight_decay', type=float, default=0.01,
                        help='Weight decay for fine-tuning (L2 regularization)')
    parser.add_argument('--finetune_dropout', type=float, default=0.5,
                        help='Dropout rate for projection head during fine-tuning')
    parser.add_argument('--projection_hidden_dim', type=int, default=512,
                        help='Hidden dimension for projection head (default: 512, reduces overfitting)')
    parser.add_argument('--finetune_warmup_epochs', type=int, default=2,
                        help='LR warmup epochs for fine-tuning (0 to disable)')
    parser.add_argument('--finetune_patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--finetune_min_delta', type=float, default=0.0,
                        help='Minimum validation loss improvement to reset patience')
    parser.add_argument('--finetune_lambda_clustering', type=float, default=0.005,
                        help='Clustering loss weight during fine-tuning (0 to disable)')
    parser.add_argument('--finetune_lambda_activation', type=float, default=0.005,
                        help='Activation loss weight during fine-tuning (0 to disable)')
    parser.add_argument('--finetune_label_smoothing', type=float, default=0.1,
                        help='Label smoothing for fine-tuning (0 to disable)')
    parser.add_argument('--unfreeze_text_encoder', action='store_true',
                        help='Unfreeze text encoder during fine-tuning (may hurt zero-shot)')

    # Training arguments - Stage 2 (Projection)
    parser.add_argument('--projection_max_batches', type=int, default=None,
                        help='Max batches for projection (None = use all)')
    parser.add_argument('--projection_visualize', action='store_true',
                        help='Generate prototype visualizations after projection')
    parser.add_argument('--projection_num_viz', type=int, default=20,
                        help='Number of prototypes to visualize')

    # Loss weights
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--lambda_clustering', type=float, default=0.1,
                        help='Weight for clustering loss')
    parser.add_argument('--lambda_activation', type=float, default=0.1,
                        help='Weight for activation loss')

    # Data Augmentation arguments
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                        choices=['light', 'medium', 'strong', 'strong_v2'],
                        help='Augmentation strength for training')
    parser.add_argument('--use_cutmix', action='store_true',
                        help='Use CutMix augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help='Probability of applying cutmix')

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

    # Stage 4: Spatial MSE Projector arguments
    parser.add_argument('--train_spatial_projector', action='store_true',
                        help='Enable Stage 4: Train spatial MSE projector')
    parser.add_argument('--spatial_epochs', type=int, default=50,
                        help='Number of epochs for spatial projector training (Stage 4)')
    parser.add_argument('--spatial_lr', type=float, default=1e-4,
                        help='Learning rate for spatial projector')
    parser.add_argument('--spatial_weight_decay', type=float, default=1e-4,
                        help='Weight decay for spatial projector')
    parser.add_argument('--spatial_top_k', type=int, default=200,
                        help='Top-k prototypes for sparse selection (1-200, 200=no sparsity)')
    parser.add_argument('--spatial_hidden_channels', type=int, default=512,
                        help='Hidden channels in spatial projector')
    parser.add_argument('--spatial_dropout', type=float, default=0.3,
                        help='Dropout for spatial projector')
    parser.add_argument('--spatial_freeze_backbone', action='store_true',
                        help='Freeze ProtoCLIP during spatial training (recommended)')
    parser.add_argument('--spatial_patience', type=int, default=10,
                        help='Early stopping patience for spatial training')
    parser.add_argument('--skip_spatial', action='store_true',
                        help='Skip Stage 4 (spatial projector training)')

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
        protoclip_sampling_method=args.sampling_method,
        dropout_rate=args.finetune_dropout,
        projection_hidden_dim=args.projection_hidden_dim
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
    """Create train and validation data loaders for the selected dataset."""
    print("\n" + "=" * 70)
    print("Creating Data Loaders")
    print("=" * 70)

    from ProtoPNet.data.transforms import get_train_transforms, get_val_transforms
    train_transform = get_train_transforms(augmentation_strength=args.augmentation_strength)
    val_transform = get_val_transforms()

    if args.dataset == 'caltech101':
        train_loader, val_loader = create_caltech101_loaders(
            caltech_root=args.caltech_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            class_mapping_file='data/caltech101_classes.json',
        )
        return train_loader, val_loader

    # --- ImageNet (default) ---
    if args.imagenet_root is None:
        raise ValueError("--imagenet_root is required when --dataset imagenet")

    # Download class mapping if needed
    class_mapping_file = Path(args.class_mapping_file)
    if not class_mapping_file.exists():
        print(f"\nClass mapping not found at {class_mapping_file}")
        print("Downloading ImageNet class names...")
        download_imagenet_class_mapping(str(class_mapping_file))

    caption_generator = ImageNetCaptionGenerator(
        class_mapping_file=str(class_mapping_file) if class_mapping_file.exists() else None
    )

    train_loader, val_loader = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        use_subset=args.use_subset,
        subset_samples=args.subset_samples,
        caption_generator=caption_generator
    )

    return train_loader, val_loader


def stage1_warmup(model, train_loader, val_loader, args, writer=None):
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
            'lr': args.warmup_lr_backbone,
            'weight_decay': args.warmup_weight_decay
        },
        {
            'params': model.image_encoder.prototype_layer.parameters(),
            'lr': args.warmup_lr_prototypes,
            'weight_decay': args.warmup_weight_decay
        },
        {
            'params': model.image_encoder.projection_head.parameters(),
            'lr': args.warmup_lr_prototypes,
            'weight_decay': args.warmup_weight_decay
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
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_writer=writer,
        stage_name='warmup'
    )

    # Create early stopping if enabled
    early_stopping = None
    if args.warmup_patience > 0:
        from ProtoPNet.utils.early_stopping import EarlyStopping
        early_stopping = EarlyStopping(
            patience=args.warmup_patience,
            min_delta=args.warmup_min_delta,
            verbose=True
        )

    # Create CutMix augmenter if enabled
    augmenter = None
    if args.use_cutmix:
        from ProtoPNet.data.mixup import CutMix
        augmenter = CutMix(alpha=args.cutmix_alpha, prob=args.mixup_prob)
        print(f"✓ CutMix enabled (alpha={args.cutmix_alpha}, prob={args.mixup_prob})")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.warmup_epochs):
        print(f"\n{'=' * 70}")
        print(f"Warmup Epoch {epoch + 1}/{args.warmup_epochs}")
        print(f"{'=' * 70}")

        # Train
        train_loss, _ = trainer.train_epoch(return_similarities=True, augmenter=augmenter)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss, _ = trainer.validate()
        print(f"Val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR: {current_lr:.6f}")

        # Log learning rate to TensorBoard
        if writer is not None:
            writer.add_scalar('warmup/learning_rate', current_lr, epoch)

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

        # Early stopping check
        if early_stopping is not None and early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
            break

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

    from ProtoPNet.utils.projection import PrototypeProjector
    from ProtoPNet.utils.visualization import visualize_prototypes, save_prototype_grid
    from ProtoPNet.data.transforms import get_val_transforms
    from ProtoPNet.data.imagenet_dataset import create_imagenet_loaders

    # Create projection-specific data loader with validation transforms (no augmentation)
    print("\nCreating projection data loader (using validation transforms)...")
    projection_loader, _ = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=get_val_transforms(),  # Use val transforms for stable features
        val_transform=get_val_transforms(),
        use_subset=args.use_subset,
        subset_samples=args.subset_samples,
        caption_generator=None  # Don't need captions for projection
    )

    # Create projector
    projector = PrototypeProjector(
        model=model,
        device=args.device
    )

    # Project prototypes
    print(f"\nProjecting {model.image_encoder.num_prototypes} prototypes...")
    new_prototypes, projection_info = projector.project_prototypes(
        projection_loader,
        num_batches=args.projection_max_batches
    )

    # Update model with projected prototypes
    model.set_prototypes(new_prototypes)
    print(f"\n✓ Prototypes replaced with training patches")

    # Save projection metadata
    projector.save_projection(
        save_dir=args.checkpoint_dir,
        projection_info=projection_info
    )

    # Print summary statistics
    stats = projection_info['projection_stats']
    print(f"\nProjection Statistics:")
    print(f"  Batches processed: {stats['total_batches']}")
    print(f"  Patches searched: {stats['total_patches']:,}")
    print(f"  Mean distance: {stats['mean_distance']:.4f}")
    print(f"  Median distance: {stats['median_distance']:.4f}")
    print(f"  Distance range: [{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]")

    # Generate visualizations if requested
    if args.projection_visualize:
        print(f"\nGenerating visualizations for {args.projection_num_viz} prototypes...")
        try:
            visualize_prototypes(
                projection_info=projection_info,
                output_dir=args.checkpoint_dir,
                num_prototypes=args.projection_num_viz,
                class_mapping_file=args.class_mapping_file
            )

            # Also create a grid image
            save_prototype_grid(
                projection_info=projection_info,
                output_dir=args.checkpoint_dir,
                grid_size=(5, 4)  # 5 rows x 4 cols = 20 prototypes
            )
        except Exception as e:
            print(f"⚠️  Warning: Visualization failed with error: {e}")
            print("   Continuing without visualizations...")

    return model


def stage3_finetune(model, train_loader, val_loader, args, writer=None):
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
    print(f"Weight decay: {args.finetune_weight_decay}")
    print(f"Dropout: {args.finetune_dropout}")
    print(f"Warmup epochs: {args.finetune_warmup_epochs}")
    print(f"Early stopping patience: {args.finetune_patience}")
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

    # Create optimizer with weight decay for regularization
    params_to_optimize = [
        {
            'params': model.image_encoder.projection_head.parameters(),
            'lr': args.finetune_lr,
            'weight_decay': args.finetune_weight_decay
        }
    ]

    if args.unfreeze_text_encoder:
        params_to_optimize.append({
            'params': model.text_encoder.parameters(),
            'lr': args.finetune_lr / 10,  # Lower LR for text encoder
            'weight_decay': args.finetune_weight_decay * 0.1  # Lower weight decay for pretrained encoder
        })

    optimizer = AdamW(params_to_optimize)

    # Create learning rate scheduler with warmup
    if args.finetune_warmup_epochs > 0:
        # Warmup scheduler: gradually increase LR from 0 to target
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of target LR
            end_factor=1.0,
            total_iters=args.finetune_warmup_epochs
        )

        # Main scheduler: cosine decay after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.finetune_epochs - args.finetune_warmup_epochs,
            eta_min=args.finetune_lr / 10
        )

        # Combine warmup + cosine decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.finetune_warmup_epochs]
        )
    else:
        # No warmup, just cosine decay
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.finetune_epochs,
            eta_min=args.finetune_lr / 10
        )

    # Create loss function with optional auxiliary losses for regularization
    loss_fn = CombinedLoss(
        lambda_contrastive=1.0,
        lambda_clustering=args.finetune_lambda_clustering,
        lambda_activation=args.finetune_lambda_activation,
        label_smoothing=args.finetune_label_smoothing
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
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_writer=writer,
        stage_name='finetune'
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(
        patience=args.finetune_patience,
        min_delta=args.finetune_min_delta,
        verbose=True
    )

    # Create CutMix augmenter if enabled
    augmenter = None
    if args.use_cutmix:
        from ProtoPNet.data.mixup import CutMix
        augmenter = CutMix(alpha=args.cutmix_alpha, prob=args.mixup_prob)
        print(f"✓ CutMix enabled (alpha={args.cutmix_alpha}, prob={args.mixup_prob})")

    for epoch in range(args.finetune_epochs):
        print(f"\n{'=' * 70}")
        print(f"Fine-tune Epoch {epoch + 1}/{args.finetune_epochs}")
        print(f"{'=' * 70}")

        # Train (compute similarities if auxiliary losses are enabled)
        use_aux_losses = (args.finetune_lambda_clustering > 0 or args.finetune_lambda_activation > 0)
        train_loss, _ = trainer.train_epoch(return_similarities=use_aux_losses, augmenter=augmenter)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss, _ = trainer.validate()
        print(f"Val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR: {current_lr:.6f}")

        # Log learning rate to TensorBoard
        if writer is not None:
            writer.add_scalar('finetune/learning_rate', current_lr, epoch)

        # Save checkpoint if best
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

        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
            break

    print(f"\n✓ Fine-tuning complete! Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint
    trainer.load_checkpoint(f"{args.checkpoint_dir}/finetune_best.pt")

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
            if hasattr(protopnet_model, 'image_encoder'):
                features = protopnet_model.image_encoder.backbone(images)
                prototype_sims = protopnet_model.image_encoder.prototype_layer(features)
            else:
                features = protopnet_model.backbone(images)
                prototype_sims = protopnet_model.prototype_layer(features)
    else:
        if hasattr(protopnet_model, 'image_encoder'):
            features = protopnet_model.image_encoder.backbone(images)
            prototype_sims = protopnet_model.image_encoder.prototype_layer(features)
        else:
            features = protopnet_model.backbone(images)
            prototype_sims = protopnet_model.prototype_layer(features)

    return prototype_sims


def train_spatial_epoch(
    spatial_projector,
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
    """Train spatial projector for one epoch."""
    from ProtoPNet.utils.prototype_selection import select_top_k_prototypes, compute_sparsity_statistics
    import torch.nn.functional as F
    from tqdm import tqdm

    spatial_projector.train()
    if freeze_protopnet:
        protopnet_model.eval()
    else:
        protopnet_model.train()
    clip_extractor.eval()

    epoch_loss = 0.0
    epoch_cosine_sim = 0.0
    num_batches = 0
    sparsity_stats_list = []

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch[0].to(device)
        batch_size = images.shape[0]

        # Extract CLIP spatial features (target)
        clip_spatial_features = clip_extractor(images)
        clip_spatial_features = clip_spatial_features.float()

        # Extract prototype similarities
        prototype_sims = extract_prototype_similarities(images, protopnet_model, device, freeze=freeze_protopnet)
        prototype_sims = prototype_sims.float()

        # Apply sparse selection
        if top_k < num_prototypes:
            sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)

            if batch_idx % 10 == 0:
                stats = compute_sparsity_statistics(prototype_sims, sparse_sims)
                sparsity_stats_list.append(stats)
        else:
            sparse_sims = prototype_sims

        # Project sparse prototypes
        projected_spatial = spatial_projector(sparse_sims)

        # Compute spatial MSE loss
        loss = loss_fn(projected_spatial, clip_spatial_features)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            pred_flat = projected_spatial.view(batch_size, 1024, -1)
            target_flat = clip_spatial_features.view(batch_size, 1024, -1)
            pred_norm = F.normalize(pred_flat, p=2, dim=1)
            target_norm = F.normalize(target_flat, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()

        epoch_loss += loss.item()
        epoch_cosine_sim += cosine_sim.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cosine': f'{cosine_sim.item():.4f}'})

    avg_loss = epoch_loss / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    if sparsity_stats_list:
        import numpy as np
        avg_sparsity_stats = {
            'mean_sparsity': np.mean([s['mean_sparsity'] for s in sparsity_stats_list]),
            'mean_num_selected': np.mean([s['mean_num_selected'] for s in sparsity_stats_list]),
        }
    else:
        avg_sparsity_stats = None

    return avg_loss, avg_cosine, avg_sparsity_stats


def validate_spatial_epoch(
    spatial_projector,
    protopnet_model,
    clip_extractor,
    dataloader,
    loss_fn,
    top_k,
    num_prototypes,
    device
):
    """Validate spatial projector for one epoch."""
    from ProtoPNet.utils.prototype_selection import select_top_k_prototypes
    import torch.nn.functional as F
    from tqdm import tqdm

    spatial_projector.eval()
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
            clip_spatial_features = clip_spatial_features.float()

            prototype_sims = extract_prototype_similarities(images, protopnet_model, device, freeze=True)
            prototype_sims = prototype_sims.float()

            # Apply sparse selection
            if top_k < num_prototypes:
                sparse_sims = select_top_k_prototypes(prototype_sims, k=top_k)
            else:
                sparse_sims = prototype_sims

            # Project
            projected_spatial = spatial_projector(sparse_sims)

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

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'val_cosine': f'{cosine_sim.item():.4f}'})

    avg_loss = epoch_loss / num_batches
    avg_cosine = epoch_cosine_sim / num_batches

    return avg_loss, avg_cosine


def save_spatial_checkpoint(spatial_projector, protopnet_model, path, epoch, freeze_backbone):
    """Save spatial projector checkpoint."""
    checkpoint = {
        'spatial_projector_state_dict': spatial_projector.state_dict(),
        'spatial_projector_config': {
            'num_prototypes': spatial_projector.num_prototypes,
            'output_dim': spatial_projector.output_dim,
            'hidden_channels': spatial_projector.hidden_channels,
            'dropout': spatial_projector.dropout
        },
        'epoch': epoch,
        'freeze_backbone': freeze_backbone
    }

    if not freeze_backbone:
        checkpoint['protopnet_state_dict'] = protopnet_model.state_dict()

    torch.save(checkpoint, path)
    print(f"✓ Saved spatial checkpoint: {path}")


def stage4_spatial_projector(model, train_loader, val_loader, args, writer=None):
    """
    Stage 4: Spatial MSE Projector Training

    Trains a spatial projection that maps ProtoPNet 14×14 prototype similarities
    to CLIP ResNet-50 layer3 spatial features using per-location MSE loss.

    Key Features:
    - Preserves spatial structure (14×14 resolution)
    - MSE loss with per-location L2 normalization
    - Optional top-k sparse prototype selection
    - Can train with frozen or trainable ProtoCLIP backbone

    Args:
        model: ProtoCLIP model (from Stage 3)
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Command-line arguments
        writer: TensorBoard writer

    Returns:
        spatial_projector: Trained SpatialMSEProjector
        model: ProtoCLIP model (optionally updated if backbone trained)
    """
    print("\n" + "=" * 70)
    print("STAGE 4: SPATIAL MSE PROJECTOR TRAINING")
    print("=" * 70)
    print(f"Epochs: {args.spatial_epochs}")
    print(f"LR: {args.spatial_lr}")
    print(f"Top-k: {args.spatial_top_k} / {args.num_prototypes}")
    print(f"Freeze backbone: {args.spatial_freeze_backbone}")

    # Import spatial components
    from ProtoPNet.models.spatial_mse_projector import SpatialMSEProjector
    from ProtoPNet.models.clip_spatial_extractor import CLIPSpatialExtractor
    from ProtoPNet.training.losses import SpatialMSELoss
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Create CLIP spatial feature extractor (frozen)
    clip_extractor = CLIPSpatialExtractor(
        layer='layer3',
        device=args.device,
        normalize_input=True
    )
    clip_extractor.eval()
    print(f"✓ CLIP extractor initialized (layer3)")

    # Create trainable spatial MSE projector
    spatial_projector = SpatialMSEProjector(
        num_prototypes=args.num_prototypes,
        output_dim=1024,  # CLIP layer3 dimension
        hidden_channels=args.spatial_hidden_channels,
        dropout=args.spatial_dropout
    ).to(args.device)
    print(f"✓ Spatial projector created")
    print(f"  Trainable parameters: {sum(p.numel() for p in spatial_projector.parameters() if p.requires_grad):,}")

    # Freeze or unfreeze ProtoCLIP backbone
    if args.spatial_freeze_backbone:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("✓ ProtoCLIP frozen")
    else:
        model.train()
        print("✓ ProtoCLIP trainable (joint training)")

    # Create optimizer
    if args.spatial_freeze_backbone:
        optimizer = Adam(
            spatial_projector.parameters(),
            lr=args.spatial_lr,
            weight_decay=args.spatial_weight_decay
        )
    else:
        # Joint training: backbone + prototypes + spatial projector
        optimizer = Adam([
            {
                'params': model.image_encoder.backbone.parameters(),
                'lr': args.spatial_lr * 0.1  # Lower LR for backbone
            },
            {
                'params': model.image_encoder.prototype_layer.parameters(),
                'lr': args.spatial_lr
            },
            {
                'params': spatial_projector.parameters(),
                'lr': args.spatial_lr
            }
        ], weight_decay=args.spatial_weight_decay)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.spatial_epochs,
        eta_min=args.spatial_lr / 10
    )

    # Loss function
    loss_fn = SpatialMSELoss(reduction='mean', normalize_per_location=True)
    print(f"✓ Loss function: SpatialMSELoss (per-location normalized)")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.spatial_patience,
        min_delta=0.0,
        verbose=True
    )

    # Sparsity info
    if args.spatial_top_k < args.num_prototypes:
        sparsity_ratio = (args.num_prototypes - args.spatial_top_k) / args.num_prototypes
        print(f"✓ Sparse selection: top-{args.spatial_top_k} ({(1 - sparsity_ratio):.1%} of prototypes)")
    else:
        print(f"✓ No sparsity (using all prototypes)")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.spatial_epochs):
        print(f"\n{'=' * 70}")
        print(f"Spatial Epoch {epoch + 1}/{args.spatial_epochs}")
        print(f"{'=' * 70}")

        # Train epoch
        train_loss, train_cosine, sparsity_stats = train_spatial_epoch(
            spatial_projector, model, clip_extractor,
            train_loader, optimizer, loss_fn,
            args.spatial_top_k, args.num_prototypes,
            args.device, args.spatial_freeze_backbone
        )

        # Validate
        val_loss, val_cosine = validate_spatial_epoch(
            spatial_projector, model, clip_extractor,
            val_loader, loss_fn,
            args.spatial_top_k, args.num_prototypes,
            args.device
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train Cosine: {train_cosine:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")
        print(f"LR: {current_lr:.6f}")
        if sparsity_stats:
            print(f"Sparsity: {sparsity_stats['mean_sparsity']:.1%} | "
                  f"Selected: {sparsity_stats['mean_num_selected']:.1f} prototypes")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('spatial/train_loss', train_loss, epoch)
            writer.add_scalar('spatial/val_loss', val_loss, epoch)
            writer.add_scalar('spatial/train_cosine', train_cosine, epoch)
            writer.add_scalar('spatial/val_cosine', val_cosine, epoch)
            writer.add_scalar('spatial/learning_rate', current_lr, epoch)
            if sparsity_stats:
                writer.add_scalar('spatial/sparsity', sparsity_stats['mean_sparsity'], epoch)

        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_spatial_checkpoint(
                spatial_projector, model,
                f"{args.checkpoint_dir}/spatial_projector_best.pt",
                epoch, args.spatial_freeze_backbone
            )
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_spatial_checkpoint(
                spatial_projector, model,
                f"{args.checkpoint_dir}/spatial_projector_epoch_{epoch + 1}.pt",
                epoch, args.spatial_freeze_backbone
            )

        # Early stopping
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {early_stopping.best_epoch + 1}")
            break

    print(f"\n✓ Spatial projector training complete! Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint
    checkpoint = torch.load(f"{args.checkpoint_dir}/spatial_projector_best.pt", weights_only=False)
    spatial_projector.load_state_dict(checkpoint['spatial_projector_state_dict'])

    return spatial_projector, model


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

    # Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.checkpoint_dir) / 'tensorboard' / f'run_{timestamp}'
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"\n✓ TensorBoard logging enabled")
    print(f"  Log directory: {log_dir}")
    print(f"  To view: tensorboard --logdir {Path(args.checkpoint_dir) / 'tensorboard'}")

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

    # Move model to device
    model = model.to(args.device)

    try:
        # Stage 1: Warmup
        if not args.skip_warmup:
            model = stage1_warmup(model, train_loader, val_loader, args, writer)
        else:
            print("\n⏭️  Skipping Stage 1 (Warmup)")

        # Stage 2: Projection
        if not args.skip_projection:
            model = stage2_projection(model, train_loader, args)
        else:
            print("\n⏭️  Skipping Stage 2 (Projection)")

        # Stage 3: Fine-tuning
        if not args.skip_finetune:
            model = stage3_finetune(model, train_loader, val_loader, args, writer)
        else:
            print("\n⏭️  Skipping Stage 3 (Fine-tuning)")

        # Stage 4: Spatial Projector [NEW]
        if args.train_spatial_projector and not args.skip_spatial:
            spatial_projector, model = stage4_spatial_projector(
                model, train_loader, val_loader, args, writer
            )
            print(f"\n✓ Spatial projector trained and saved")
        elif not args.train_spatial_projector:
            pass  # Stage 4 not requested
        else:
            print("\n⏭️  Skipping Stage 4 (Spatial Projector)")

    finally:
        # Close TensorBoard writer
        writer.close()
        print(f"\n✓ TensorBoard logs saved to: {log_dir}")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"\nFinal model checkpoints:")
    print(f"  - warmup_best.pt (after Stage 1)")
    print(f"  - finetune_best.pt (after Stage 3)")
    if args.train_spatial_projector and not args.skip_spatial:
        print(f"  - spatial_projector_best.pt (after Stage 4)")
    print(f"\nTensorBoard logs:")
    print(f"  - {log_dir}")
    print(f"  - View with: tensorboard --logdir {Path(args.checkpoint_dir) / 'tensorboard'}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on retrieval tasks: python scripts/evaluate.py")
    print(f"  2. Visualize prototypes: python scripts/visualize_prototypes.py")
    print(f"  3. Test zero-shot capabilities with custom queries")
    print("=" * 70)


if __name__ == '__main__':
    main()
