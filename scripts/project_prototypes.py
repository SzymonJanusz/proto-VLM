"""
Run prototype projection on a trained model.

This script loads a trained ProtoCLIP checkpoint and projects prototypes
to nearest training patches for interpretability.

Usage:
    # Project prototypes from finetuned model
    python scripts/project_prototypes.py \
        --checkpoint ./checkpoints/overfitting_fix_3/finetune_best.pt \
        --imagenet_root imagenet_tiny \
        --output_checkpoint ./checkpoints/overfitting_fix_3/finetune_best_projected.pt

    # With visualization
    python scripts/project_prototypes.py \
        --checkpoint ./checkpoints/overfitting_fix_3/finetune_best.pt \
        --imagenet_root imagenet_tiny \
        --visualize \
        --num_viz 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from datetime import datetime

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.data.imagenet_dataset import create_imagenet_loaders
from ProtoPNet.data.transforms import get_val_transforms
from ProtoPNet.data.caption_generator import ImageNetCaptionGenerator, download_imagenet_class_mapping
from ProtoPNet.utils.projection import PrototypeProjector
from ProtoPNet.utils.visualization import visualize_prototypes, save_prototype_grid


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Project prototypes from a trained ProtoCLIP model'
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--imagenet_root', type=str, required=True,
                        help='Path to ImageNet root directory')

    # Model arguments (auto-detected from checkpoint if not provided)
    parser.add_argument('--num_prototypes', type=int, default=None,
                        help='Number of prototypes (auto-detected from checkpoint)')
    parser.add_argument('--pooling_mode', type=str, default=None,
                        choices=['max', 'attention'],
                        help='Pooling mode (auto-detected from checkpoint)')
    parser.add_argument('--projection_hidden_dim', type=int, default=None,
                        help='Projection head hidden dim (auto-detected from checkpoint)')

    # Data arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for projection')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use subset of data (faster for testing)')
    parser.add_argument('--subset_samples', type=int, default=10000,
                        help='Number of samples in subset')
    parser.add_argument('--class_mapping_file', type=str,
                        default='./data/imagenet_classes.json',
                        help='Path to ImageNet class mapping file')

    # Projection arguments
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Max batches to process for projection (None = all)')

    # Output arguments
    parser.add_argument('--output_checkpoint', type=str, default=None,
                        help='Path to save projected checkpoint (default: checkpoint_dir/finetune_best_projected.pt)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save projection info (default: checkpoint directory)')

    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate prototype visualizations')
    parser.add_argument('--num_viz', type=int, default=20,
                        help='Number of prototypes to visualize')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def load_model(checkpoint_path, args):
    """Load trained ProtoCLIP model from checkpoint"""
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    # Load checkpoint first to infer architecture
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Infer architecture from checkpoint
    if 'image_encoder.projection_head.0.weight' in state_dict:
        projection_hidden_dim = state_dict['image_encoder.projection_head.0.weight'].shape[0]
        print(f"Detected projection_hidden_dim: {projection_hidden_dim}")
        if args.projection_hidden_dim is None:
            args.projection_hidden_dim = projection_hidden_dim
    else:
        projection_hidden_dim = args.projection_hidden_dim or 512

    # Infer num_prototypes from checkpoint
    if 'image_encoder.prototype_layer.prototypes' in state_dict:
        num_prototypes = state_dict['image_encoder.prototype_layer.prototypes'].shape[0]
        print(f"Detected num_prototypes: {num_prototypes}")
        if args.num_prototypes is None:
            args.num_prototypes = num_prototypes
    else:
        num_prototypes = args.num_prototypes or 200

    # Infer pooling mode
    if 'image_encoder.prototype_layer.attention_pooling.attention_weights' in state_dict:
        pooling_mode = 'attention'
        print(f"Detected pooling_mode: attention")
        if args.pooling_mode is None:
            args.pooling_mode = pooling_mode
    else:
        pooling_mode = args.pooling_mode or 'max'

    # Create model with correct architecture
    model = ProtoCLIP(
        num_prototypes=num_prototypes,
        image_backbone='resnet50',
        text_model="openai/clip-vit-base-patch32",
        embedding_dim=512,
        freeze_text_encoder=True,
        temperature=0.07,
        pooling_mode=pooling_mode,
        projection_hidden_dim=projection_hidden_dim
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully")
    print(f"  Prototypes: {num_prototypes}")
    print(f"  Projection hidden dim: {projection_hidden_dim}")
    print(f"  Pooling mode: {pooling_mode}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {args.device}")

    return model, checkpoint


def main():
    args = parse_args()

    print("=" * 70)
    print("Prototype Projection")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ImageNet root: {args.imagenet_root}")
    print(f"Device: {args.device}")

    # Load model and original checkpoint
    model, original_checkpoint = load_model(args.checkpoint, args)

    # Set up output paths
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_checkpoint is None:
        # Default: add _projected suffix
        output_checkpoint = output_dir / f"{checkpoint_path.stem}_projected{checkpoint_path.suffix}"
    else:
        output_checkpoint = Path(args.output_checkpoint)

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

    # Create data loader with validation transforms (no augmentation)
    print("\n" + "=" * 70)
    print("Creating Data Loader")
    print("=" * 70)

    train_loader, _ = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=get_val_transforms(),  # Use val transforms for stable features
        val_transform=get_val_transforms(),
        use_subset=args.use_subset,
        subset_samples=args.subset_samples,
        caption_generator=caption_generator
    )

    print(f"Training samples: {len(train_loader.dataset)}")

    # Run projection
    print("\n" + "=" * 70)
    print("PROTOTYPE PROJECTION")
    print("=" * 70)

    projector = PrototypeProjector(
        model=model,
        device=args.device
    )

    new_prototypes, projection_info = projector.project_prototypes(
        train_loader,
        num_batches=args.max_batches
    )

    # Update model with projected prototypes
    model.set_prototypes(new_prototypes)
    print(f"\n✓ Prototypes replaced with training patches")

    # Save projection metadata
    projector.save_projection(
        save_dir=output_dir,
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

    # Save updated checkpoint
    print("\n" + "=" * 70)
    print("SAVING CHECKPOINT")
    print("=" * 70)

    # Create new checkpoint with projected prototypes
    new_checkpoint = {
        'epoch': original_checkpoint.get('epoch', -1),
        'model_state_dict': model.state_dict(),
        'projection_date': datetime.now().isoformat(),
        'projection_stats': stats,
    }

    # Preserve optimizer state if available
    if 'optimizer_state_dict' in original_checkpoint:
        new_checkpoint['optimizer_state_dict'] = original_checkpoint['optimizer_state_dict']

    # Preserve training history if available
    for key in ['train_losses', 'val_losses', 'best_val_loss']:
        if key in original_checkpoint:
            new_checkpoint[key] = original_checkpoint[key]

    torch.save(new_checkpoint, output_checkpoint)
    print(f"✓ Saved projected checkpoint to: {output_checkpoint}")

    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        try:
            visualize_prototypes(
                projection_info=projection_info,
                output_dir=output_dir,
                num_prototypes=args.num_viz,
                class_mapping_file=args.class_mapping_file
            )

            # Also create a grid image
            save_prototype_grid(
                projection_info=projection_info,
                output_dir=output_dir,
                grid_size=(5, 4)  # 5 rows x 4 cols = 20 prototypes
            )
            print(f"✓ Visualizations saved to: {output_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Visualization failed with error: {e}")
            print("   Continuing without visualizations...")

    print("\n" + "=" * 70)
    print("PROJECTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Checkpoint: {output_checkpoint}")
    print(f"  Projection info: {output_dir / 'projection_info.pkl'}")
    print(f"  Summary: {output_dir / 'projection_summary.txt'}")
    if args.visualize:
        print(f"  Visualizations: {output_dir / 'prototype_visualizations'}")

    print(f"\nYou can now use the projected checkpoint for evaluation:")
    print(f"  python scripts/evaluate.py \\")
    print(f"    --checkpoint {output_checkpoint} \\")
    print(f"    --imagenet_root {args.imagenet_root} \\")
    print(f"    --mode retrieval --visualize")


if __name__ == '__main__':
    main()
