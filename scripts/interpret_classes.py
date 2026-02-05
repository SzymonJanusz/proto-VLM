#!/usr/bin/env python
"""
Class-Level Interpretation using CNN Features + Text Decoders

Generates free-form text descriptions of class-level visual features by:
1. Extracting abstract CNN features (not similarity scores) from images
2. Aggregating features across class images
3. Projecting to CLIP space using trained projection head
4. Generating text with unsupervised decoders (ClipCap, etc.)

Usage:
    # Single image
    python scripts/interpret_classes.py \
        --mode single_image \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --projector_checkpoint checkpoints/cnn_projector/best.pt \
        --image_path imagenet_tiny/val/n04067472/n04067472_108.JPEG \
        --decoders simple_gpt2 \
        --output_dir results/single_image

    # Class-level
    python scripts/interpret_classes.py \
        --mode class \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --projector_checkpoint checkpoints/cnn_projector/best.pt \
        --data_root imagenet_tiny \
        --classes "corn,bell pepper,reel" \
        --aggregation mean \
        --decoders simple_gpt2 \
        --output_dir results/class_interpretation
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.models.cnn_feature_projector import CNNFeatureProjector
from ProtoPNet.utils.class_interpretation import (
    CNNFeatureExtractor,
    CLIPViTFeatureExtractor,
    FeatureAggregator,
    ClassFilteredDataset
)
from ProtoPNet.utils.text_decoders import get_text_decoder
from ProtoPNet.data.imagenet_dataset import ImageNetWithCaptions
from ProtoPNet.data.transforms import get_val_transforms


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Interpret ProtoPNet using CNN features and text decoders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single_image', 'class'],
                       help='Interpretation mode')

    # Feature extraction method
    parser.add_argument('--feature_extractor', type=str, default='cnn',
                       choices=['cnn', 'clip_vit'],
                       help='Feature extraction method: cnn (CNN+projector) or clip_vit (CLIP ViT directly)')

    # Model checkpoints
    parser.add_argument('--backbone_checkpoint', type=str,
                       help='Path to trained ProtoPNet checkpoint (required for cnn mode)')
    parser.add_argument('--projector_checkpoint', type=str,
                       help='Path to trained CNNFeatureProjector checkpoint (required for cnn mode)')

    # Data
    parser.add_argument('--data_root', type=str,
                       help='Path to ImageNet root directory (required for class mode)')
    parser.add_argument('--class_mapping', type=str, default='data/imagenet_classes.json',
                       help='Path to ImageNet class mapping JSON')

    # Single image mode
    parser.add_argument('--image_path', type=str,
                       help='Path to single image (required for single_image mode)')

    # Class mode
    parser.add_argument('--classes', type=str,
                       help='Comma-separated class names (required for class mode)')
    parser.add_argument('--class_ids', type=str,
                       help='Comma-separated class IDs (alternative to --classes)')
    parser.add_argument('--max_images_per_class', type=int, default=None,
                       help='Limit images per class (for faster testing)')

    # Feature aggregation
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'weighted_mean', 'median', 'max'],
                       help='Method for aggregating features across images')

    # Text decoders
    parser.add_argument('--decoders', type=str, default='simple_gpt2',
                       help='Comma-separated decoder names (simple_gpt2, clipcap)')
    parser.add_argument('--clipcap_model', type=str, default=None,
                       help='Path to ClipCap pretrained weights')
    parser.add_argument('--num_captions', type=int, default=5,
                       help='Number of captions to generate per interpretation')
    parser.add_argument('--max_caption_length', type=int, default=30,
                       help='Maximum caption length in tokens')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--save_embeddings', action='store_true',
                       help='Save CLIP embeddings as .npy files')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()


def load_class_mapping(class_mapping_path: str) -> dict:
    """Load ImageNet class ID to name mapping."""
    with open(class_mapping_path, 'r') as f:
        mapping = json.load(f)
    # Convert string keys to int
    return {int(k): v for k, v in mapping.items()}


def class_name_to_id(class_name: str, class_mapping: dict) -> int:
    """Find class ID from class name."""
    for class_id, name in class_mapping.items():
        if name.lower() == class_name.lower():
            return class_id
    raise ValueError(f"Class '{class_name}' not found in mapping")


def interpret_single_image(args, extractor, projector, decoders, class_mapping):
    """Interpret a single image."""
    print(f"\n{'=' * 80}")
    print("SINGLE IMAGE INTERPRETATION")
    print(f"{'=' * 80}")
    print(f"Image: {args.image_path}")
    print()

    # Load and preprocess image
    transforms = get_val_transforms()
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transforms(image)

    # Extract features
    if args.feature_extractor == 'cnn':
        print("Extracting CNN features...")
        features = extractor.extract_single_image_features(image_tensor)
        print(f"  Features shape: {features.shape}")

        # Project to CLIP space
        print("Projecting to CLIP space...")
        features = features.unsqueeze(0).to(args.device)  # (1, feature_dim)
        clip_embedding = projector(features).squeeze(0)  # (512,)
        print(f"  Embedding shape: {clip_embedding.shape}")
        print(f"  Embedding norm: {torch.norm(clip_embedding).item():.4f}")
    else:  # clip_vit
        print("Extracting CLIP ViT embeddings...")
        clip_embedding = extractor.extract_single_image_features(image_tensor)
        print(f"  Embedding shape: {clip_embedding.shape}")
        print(f"  Embedding norm: {torch.norm(clip_embedding).item():.4f}")

    # Generate text with each decoder
    results = {
        'image_path': args.image_path,
        'feature_extractor': args.feature_extractor,
        'clip_embedding_shape': list(clip_embedding.shape),
        'text_interpretations': {}
    }

    for decoder_name, decoder in decoders.items():
        print(f"\nGenerating captions with {decoder_name}...")
        captions = decoder.decode(
            clip_embedding,
            num_captions=args.num_captions,
            max_length=args.max_caption_length
        )

        results['text_interpretations'][decoder_name] = captions

        print(f"  {decoder_name.upper()}:")
        for i, caption in enumerate(captions, 1):
            print(f"    {i}. {caption}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'single_image_interpretation.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {json_path}")

    # Save embedding if requested
    if args.save_embeddings:
        embed_path = os.path.join(args.output_dir, 'clip_embedding.npy')
        np.save(embed_path, clip_embedding.cpu().numpy())
        print(f"Saved embedding to: {embed_path}")

    return results


def interpret_class(
    class_name: str,
    class_id: int,
    args,
    extractor,
    projector,
    decoders,
    dataset
):
    """Interpret a single class by aggregating features from all class images."""
    print(f"\n{'=' * 80}")
    print(f"CLASS: {class_name} (ID: {class_id})")
    print(f"{'=' * 80}")

    # Filter dataset to this class
    filtered_dataset = ClassFilteredDataset(dataset, [class_id])
    print(f"  Images in class: {len(filtered_dataset)}")

    if len(filtered_dataset) == 0:
        print("  Warning: No images found for this class!")
        return None

    # Create dataloader
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Extract features from all images
    if args.feature_extractor == 'cnn':
        print("  Extracting CNN features...")
        features, image_paths = extractor.extract_class_features(
            dataloader,
            max_images=args.max_images_per_class
        )
        print(f"  Extracted features: {features.shape}")

        # Aggregate features
        print(f"  Aggregating features (method: {args.aggregation})...")
        aggregator = FeatureAggregator()
        aggregated = aggregator.aggregate(features, method=args.aggregation)
        print(f"  Aggregated shape: {aggregated.shape}")

        # Project to CLIP space
        print("  Projecting to CLIP space...")
        aggregated = aggregated.unsqueeze(0).to(args.device)  # (1, feature_dim)
        clip_embedding = projector(aggregated).squeeze(0)  # (512,)
        print(f"  Embedding norm: {torch.norm(clip_embedding).item():.4f}")
    else:  # clip_vit
        print("  Extracting CLIP ViT embeddings...")
        features, image_paths = extractor.extract_class_features(
            dataloader,
            max_images=args.max_images_per_class
        )
        print(f"  Extracted features: {features.shape}")

        # Aggregate features
        print(f"  Aggregating features (method: {args.aggregation})...")
        aggregator = FeatureAggregator()
        clip_embedding = aggregator.aggregate(features, method=args.aggregation)
        print(f"  Aggregated shape: {clip_embedding.shape}")
        print(f"  Embedding norm: {torch.norm(clip_embedding).item():.4f}")

    # Generate text with each decoder
    results = {
        'class_name': class_name,
        'class_id': class_id,
        'num_images_processed': len(features),
        'feature_extractor': args.feature_extractor,
        'aggregation_method': args.aggregation,
        'features_shape': list(features.shape),
        'clip_embedding_shape': list(clip_embedding.shape),
        'text_interpretations': {}
    }

    for decoder_name, decoder in decoders.items():
        print(f"\n  Generating captions with {decoder_name}...")
        captions = decoder.decode(
            clip_embedding,
            num_captions=args.num_captions,
            max_length=args.max_caption_length
        )

        results['text_interpretations'][decoder_name] = captions

        print(f"  {decoder_name.upper()}:")
        for i, caption in enumerate(captions, 1):
            print(f"    {i}. {caption}")

    # Save per-class results
    class_dir = os.path.join(args.output_dir, class_name.replace(' ', '_'))
    os.makedirs(class_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(class_dir, 'interpretation.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {json_path}")

    # Save embedding if requested
    if args.save_embeddings:
        embed_path = os.path.join(class_dir, 'clip_embedding.npy')
        np.save(embed_path, clip_embedding.cpu().numpy())
        print(f"  Saved embedding to: {embed_path}")

    return results


def main():
    args = parse_args()

    # Validate arguments
    if args.mode == 'single_image' and not args.image_path:
        raise ValueError("--image_path is required for single_image mode")
    if args.mode == 'class' and not args.data_root:
        raise ValueError("--data_root is required for class mode")
    if args.mode == 'class' and not (args.classes or args.class_ids):
        raise ValueError("Either --classes or --class_ids is required for class mode")

    # Validate feature extractor requirements
    if args.feature_extractor == 'cnn':
        if not args.backbone_checkpoint:
            raise ValueError("--backbone_checkpoint is required for cnn feature extractor")
        if not args.projector_checkpoint:
            raise ValueError("--projector_checkpoint is required for cnn feature extractor")

    print("=" * 80)
    print("CLASS-LEVEL INTERPRETATION")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Feature extractor: {args.feature_extractor}")
    if args.feature_extractor == 'cnn':
        print(f"Backbone checkpoint: {args.backbone_checkpoint}")
        print(f"Projector checkpoint: {args.projector_checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Create feature extractor based on mode
    projector = None

    if args.feature_extractor == 'cnn':
        # Load ProtoCLIP model (for frozen backbone)
        print("Loading ProtoCLIP model...")

        # Load checkpoint
        checkpoint = torch.load(args.backbone_checkpoint, map_location=args.device, weights_only=False)

        # Infer projection_hidden_dim from checkpoint
        state_dict = checkpoint['model_state_dict']
        projection_first_layer_weight = state_dict['image_encoder.projection_head.0.weight']
        projection_hidden_dim = projection_first_layer_weight.shape[0]

        print(f"  Detected projection_hidden_dim: {projection_hidden_dim}")

        # Create model with correct config
        model = ProtoCLIP(
            num_prototypes=200,
            image_backbone='resnet50',
            embedding_dim=512,
            pooling_mode='max',
            freeze_text_encoder=True,
            projection_hidden_dim=projection_hidden_dim
        ).to(args.device)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        backbone = model.image_encoder.backbone
        print("  Loaded backbone")

        # Load CNN Feature Projector
        print("Loading CNN Feature Projector...")
        projector = CNNFeatureProjector.load(args.projector_checkpoint, device=args.device)
        projector.eval()
        print("  Loaded projector")

        # Create CNN Feature Extractor
        print("Creating CNN Feature Extractor...")
        extractor = CNNFeatureExtractor(backbone, device=args.device)
        print("  Created extractor")

    elif args.feature_extractor == 'clip_vit':
        # Create CLIP ViT Feature Extractor
        print("Creating CLIP ViT Feature Extractor...")
        extractor = CLIPViTFeatureExtractor(device=args.device, clip_model='ViT-B/32')
        print("  Created extractor")
        # No projector needed - CLIP embeddings are already in the right space

    # Load text decoders
    print("\nLoading text decoders...")
    decoder_names = [name.strip() for name in args.decoders.split(',')]
    decoders = {}

    for decoder_name in decoder_names:
        print(f"  Loading {decoder_name}...")
        if decoder_name == 'clipcap' and args.clipcap_model:
            decoder = get_text_decoder(
                decoder_name,
                device=args.device,
                model_path=args.clipcap_model
            )
        else:
            decoder = get_text_decoder(decoder_name, device=args.device)

        decoders[decoder_name] = decoder

    print(f"  Loaded {len(decoders)} decoders")

    # Mode-specific execution
    if args.mode == 'single_image':
        # Single image interpretation
        class_mapping = load_class_mapping(args.class_mapping) if args.class_mapping else {}
        results = interpret_single_image(args, extractor, projector, decoders, class_mapping)

    else:  # class mode
        # Load class mapping
        class_mapping = load_class_mapping(args.class_mapping)

        # Parse target classes
        if args.classes:
            class_names = [name.strip() for name in args.classes.split(',')]
            target_class_ids = [class_name_to_id(name, class_mapping) for name in class_names]
        else:
            target_class_ids = [int(cid.strip()) for cid in args.class_ids.split(',')]
            class_names = [class_mapping[cid] for cid in target_class_ids]

        print(f"\nTarget classes: {len(class_names)}")
        for name, cid in zip(class_names, target_class_ids):
            print(f"  - {name} (ID: {cid})")

        # Load dataset
        print("\nLoading dataset...")
        transforms = get_val_transforms()
        dataset = ImageNetWithCaptions(
            root=args.data_root,
            split='val',
            transform=transforms
        )
        print(f"  Total images: {len(dataset)}")

        # Interpret each class
        all_results = []
        for class_name, class_id in zip(class_names, target_class_ids):
            result = interpret_class(
                class_name, class_id, args,
                extractor, projector, decoders, dataset
            )
            if result:
                all_results.append(result)

        # Save summary
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'num_classes': len(all_results),
                'classes': all_results,
                'aggregation_method': args.aggregation,
                'decoders': decoder_names
            }, f, indent=2)
        print(f"\n{'=' * 80}")
        print(f"Saved summary to: {summary_path}")
        print(f"{'=' * 80}")

    print("\nDone!")


if __name__ == '__main__':
    main()
