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
    --subset_samples 10000

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

    # Create data loaders with augmentation
    from ProtoPNet.data.transforms import get_train_transforms, get_val_transforms
    train_loader, val_loader = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=get_train_transforms(augmentation_strength=args.augmentation_strength),
        val_transform=get_val_transforms(),
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
    print(f"  - warmup_best.pt")
    print(f"  - finetune_best.pt")
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
