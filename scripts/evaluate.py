"""
Evaluation script for ProtoCLIP model.

This script provides two main evaluation modes:
    1. Zero-shot classification on ImageNet validation set
    2. Image-to-text retrieval for learned prototypes

Usage:
    # Zero-shot classification
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/overfitting_fix_3/finetune_best.pt \
        --imagenet_root /path/to/imagenet \
        --mode zero-shot \
        --batch_size 128

    # Prototype image-to-text retrieval
    python scripts/evaluate.py
        --checkpoint ./checkpoints/overfitting_fix_3/finetune_best.pt \
        --imagenet_root /path/to/imagenet \
        --mode retrieval \
        --num_retrieval_samples 1000

    # Both evaluations
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/overfitting_fix_3/finetune_best.pt \
        --imagenet_root /path/to/imagenet \
        --mode both
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import pickle

from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.data.imagenet_dataset import create_imagenet_loaders
from ProtoPNet.data.caption_generator import ImageNetCaptionGenerator, download_imagenet_class_mapping
from ProtoPNet.data.transforms import get_val_transforms
from ProtoPNet.utils.projection import get_patch_bounding_box


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate ProtoCLIP model'
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--imagenet_root', type=str, required=True,
                        help='Path to ImageNet root directory')

    # Evaluation mode
    parser.add_argument('--mode', type=str, default='both',
                        choices=['zero-shot', 'retrieval', 'both'],
                        help='Evaluation mode')

    # Model arguments
    parser.add_argument('--num_prototypes', type=int, default=200,
                        help='Number of prototypes (must match checkpoint)')
    parser.add_argument('--pooling_mode', type=str, default='max',
                        choices=['max', 'attention'],
                        help='Pooling mode (must match checkpoint)')

    # Data arguments
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--class_mapping_file', type=str,
                        default='./data/imagenet_classes.json',
                        help='Path to ImageNet class mapping file')

    # Zero-shot arguments
    parser.add_argument('--use_ensemble', action='store_true',
                        help='Use ensemble of prompts for zero-shot (more accurate but slower)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of samples to evaluate (for quick testing)')

    # Retrieval arguments
    parser.add_argument('--num_retrieval_samples', type=int, default=1000,
                        help='Number of samples for retrieval evaluation')
    parser.add_argument('--top_k_retrieval', type=int, default=5,
                        help='Show top-k text matches for each prototype')

    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate grid visualizations of prototype matches')
    parser.add_argument('--num_prototypes_to_visualize', type=int, default=20,
                        help='Number of prototypes to visualize (default: 20)')
    parser.add_argument('--viz_grid_cols', type=int, default=5,
                        help='Number of columns in visualization grid (default: 5)')
    parser.add_argument('--viz_output_dir', type=str, default=None,
                        help='Directory to save visualizations (default: checkpoint_dir/visualizations)')

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
    # projection_head.0.weight shape is (projection_hidden_dim, num_prototypes)
    if 'image_encoder.projection_head.0.weight' in state_dict:
        projection_hidden_dim = state_dict['image_encoder.projection_head.0.weight'].shape[0]
        print(f"Detected projection_hidden_dim: {projection_hidden_dim}")
    else:
        projection_hidden_dim = 512
        print(f"Using default projection_hidden_dim: {projection_hidden_dim}")

    # Infer num_prototypes from checkpoint
    if 'image_encoder.prototype_layer.prototypes' in state_dict:
        num_prototypes = state_dict['image_encoder.prototype_layer.prototypes'].shape[0]
        print(f"Detected num_prototypes: {num_prototypes}")
        args.num_prototypes = num_prototypes  # Update args
    else:
        num_prototypes = args.num_prototypes

    # Infer pooling mode (check if attention pooling weights exist)
    if 'image_encoder.prototype_layer.attention_pooling.attention_weights' in state_dict:
        pooling_mode = 'attention'
        print(f"Detected pooling_mode: attention")
        args.pooling_mode = pooling_mode
    else:
        pooling_mode = args.pooling_mode

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

    return model


def zero_shot_classification(model, val_loader, caption_generator, args):
    """
    Evaluate zero-shot classification accuracy on ImageNet.

    For each image, compute similarity with all class text embeddings,
    and predict the class with highest similarity.
    """
    print("\n" + "=" * 70)
    print("ZERO-SHOT CLASSIFICATION EVALUATION")
    print("=" * 70)

    # Generate text embeddings for all available classes
    print("\nGenerating text embeddings for all classes...")
    all_class_captions = caption_generator.get_all_class_captions()

    # Get sorted class indices to maintain consistent ordering
    class_indices = sorted(all_class_captions.keys())
    num_classes = len(class_indices)

    # Create mapping from class index to position in text_embeddings
    class_to_idx = {class_idx: i for i, class_idx in enumerate(class_indices)}

    print(f"Number of classes: {num_classes}")
    print(f"Class range: {min(class_indices)} to {max(class_indices)}")

    if args.use_ensemble:
        # Use multiple templates and average embeddings
        print("Using ensemble of prompts for better accuracy...")
        templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a {} in the scene",
            "a cropped photo of a {}",
        ]

        text_embeddings_list = []
        for template in templates:
            captions = [template.format(caption_generator.class_names[idx])
                       for idx in class_indices]
            with torch.no_grad():
                embeddings = model.encode_text(captions)
            text_embeddings_list.append(embeddings)

        # Average embeddings across templates
        text_embeddings = torch.stack(text_embeddings_list).mean(dim=0)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
    else:
        # Use single template (faster)
        captions = [all_class_captions[idx] for idx in class_indices]
        with torch.no_grad():
            text_embeddings = model.encode_text(captions)

    print(f"Text embeddings shape: {text_embeddings.shape}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    skipped = 0

    # Convert class_indices list to tensor for mapping predictions
    class_indices_tensor = torch.tensor(class_indices, device=args.device)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Zero-shot evaluation")
        for batch_idx, batch in enumerate(pbar):
            if args.num_samples and total >= args.num_samples:
                break

            # Handle both (images, captions) and (images, targets, captions) formats
            if len(batch) == 2:
                images, captions = batch
                # Extract class indices from dataset
                # This is a workaround - we need targets for evaluation
                # Get the batch indices and look up targets from dataset
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(val_loader.dataset))
                targets = torch.tensor([val_loader.dataset.samples[i][1]
                                       for i in range(start_idx, end_idx)],
                                      device=args.device)
            else:
                images, targets, captions = batch
                targets = targets.to(args.device)

            images = images.to(args.device)

            # Filter out samples whose class is not in our class set
            valid_mask = torch.tensor([t.item() in class_to_idx for t in targets], device=args.device)
            if not valid_mask.any():
                skipped += targets.size(0)
                continue

            images = images[valid_mask]
            targets = targets[valid_mask]

            # Get image embeddings
            image_embeddings = model.encode_image(images)

            # Compute similarity with all class text embeddings
            # Shape: (batch_size, num_classes)
            logits = image_embeddings @ text_embeddings.t()

            # Top-1 accuracy
            # pred_top1_idx are indices into class_indices (0 to num_classes-1)
            pred_top1_idx = logits.argmax(dim=1)
            # Map back to actual class indices
            pred_top1 = class_indices_tensor[pred_top1_idx]
            correct_top1 += (pred_top1 == targets).sum().item()

            # Top-5 accuracy
            # pred_top5_idx shape: (batch_size, 5)
            pred_top5_idx = logits.topk(5, dim=1)[1]
            # Map back to actual class indices
            pred_top5 = class_indices_tensor[pred_top5_idx]  # (batch_size, 5)
            # Check if target is in top-5 predictions
            correct_top5 += sum([(targets[i] == pred_top5[i]).any().item() for i in range(len(targets))])

            total += targets.size(0)

            # Update progress bar
            top1_acc = 100.0 * correct_top1 / total if total > 0 else 0.0
            top5_acc = 100.0 * correct_top5 / total if total > 0 else 0.0
            pbar.set_postfix({
                'Top-1': f'{top1_acc:.2f}%',
                'Top-5': f'{top5_acc:.2f}%',
                'Samples': total
            })

    if skipped > 0:
        print(f"\nNote: Skipped {skipped} samples with classes not in the caption set")

    # Final results
    top1_accuracy = 100.0 * correct_top1 / total
    top5_accuracy = 100.0 * correct_top5 / total

    print("\n" + "=" * 70)
    print("ZERO-SHOT CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': total
    }


def visualize_prototype_matches(prototype_idx, matches, val_loader, output_dir,
                                projection_info=None, caption_generator=None, grid_cols=5):
    """
    Create a grid visualization of top-k images for a prototype.
    First image shows the ORIGINAL prototype patch, followed by top matches.

    Args:
        prototype_idx: Index of the prototype
        matches: List of match dictionaries with 'image_idx', 'caption', 'similarity'
        val_loader: DataLoader to access images
        output_dir: Directory to save visualization
        projection_info: Projection metadata (optional, shows original prototype if provided)
        caption_generator: Caption generator for prototype original image
        grid_cols: Number of columns in the grid
    """
    # Add 1 extra slot for the original prototype image
    has_original = projection_info is not None
    total_images = len(matches) + (1 if has_original else 0)
    grid_rows = (total_images + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3.5))
    fig.suptitle(f'Prototype {prototype_idx} - Original + Top {len(matches)} Matches',
                 fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if grid_rows > 1 or grid_cols > 1 else axes

    current_idx = 0

    # First, show the ORIGINAL prototype image if available
    if has_original and prototype_idx < len(projection_info['image_path']):
        ax = axes[current_idx]
        img_path = projection_info['image_path'][prototype_idx]
        patch_h = projection_info['patch_h'][prototype_idx]
        patch_w = projection_info['patch_w'][prototype_idx]
        class_id = projection_info['class_id'][prototype_idx]

        try:
            # Load original image
            img = Image.open(img_path).convert('RGB')
            img_width, img_height = img.size

            # Get patch bounding box for 224x224 input (what model saw)
            model_input_size = 224
            feature_map_size = 14
            top_224, left_224, bottom_224, right_224 = get_patch_bounding_box(
                patch_h, patch_w,
                img_size=model_input_size,
                feature_size=feature_map_size
            )

            # Scale coordinates to actual image size
            scale_h = img_height / model_input_size
            scale_w = img_width / model_input_size

            top = int(top_224 * scale_h)
            left = int(left_224 * scale_w)
            bottom = int(bottom_224 * scale_h)
            right = int(right_224 * scale_w)

            # Draw the full image with patch highlighted
            ax.imshow(img)

            # Highlight the prototype patch with a red rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((left, top), right - left, bottom - top,
                           linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.axis('off')

            # Add title with patch info
            if caption_generator and class_id >= 0:
                caption = caption_generator.generate_caption(class_id)
                title = f"**ORIGINAL PROTOTYPE**\n{caption}\nPatch: ({patch_h},{patch_w}) | Box: {right-left}x{bottom-top}px"
            else:
                title = f"**ORIGINAL PROTOTYPE**\nClass: {class_id}\nPatch: ({patch_h},{patch_w}) | Box: {right-left}x{bottom-top}px"

            ax.set_title(title, fontsize=8, pad=5, fontweight='bold', color='red')

            # Red border for original
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\nprototype image\n{e}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        current_idx += 1

    # Now show the top matches
    for match in matches:
        ax = axes[current_idx]

        # Get the actual image from dataset
        image_idx = match['image_idx']
        img_path = val_loader.dataset.samples[image_idx][0]

        try:
            # Load and display image
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.axis('off')

            # Add caption and similarity as title
            title = f"Match #{match['rank']}\n{match['caption']}\n(sim: {match['similarity']:.3f})"
            ax.set_title(title, fontsize=9, pad=5)

            # Add colored border based on similarity (green=high, red=low)
            similarity_color = plt.cm.RdYlGn(match['similarity'])
            for spine in ax.spines.values():
                spine.set_edgecolor(similarity_color)
                spine.set_linewidth(3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\nimage {image_idx}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        current_idx += 1

    # Hide unused subplots
    for idx in range(current_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = Path(output_dir) / f'prototype_{prototype_idx:03d}_matches.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def prototype_image_to_text_retrieval(model, val_loader, caption_generator, args):
    """
    For each learned prototype, find the most similar images and their captions.

    This shows what visual concepts each prototype has learned.
    """
    print("\n" + "=" * 70)
    print("PROTOTYPE IMAGE-TO-TEXT RETRIEVAL")
    print("=" * 70)

    # Try to load projection info if available
    checkpoint_dir = Path(args.checkpoint).parent
    projection_info_path = checkpoint_dir / 'projection_info.pkl'
    projection_info = None

    if projection_info_path.exists():
        try:
            with open(projection_info_path, 'rb') as f:
                projection_info = pickle.load(f)
            print(f"✓ Loaded projection info from {projection_info_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load projection info: {e}")
            print("   Visualizations will not show original prototype images")
    else:
        print(f"⚠️  Projection info not found at {projection_info_path}")
        print("   Visualizations will not show original prototype images")
        print("   Run Stage 2 (projection) first if you want to see prototype origins")

    # Get prototype vectors
    prototypes = model.get_prototypes()  # Shape: (num_prototypes, prototype_dim)
    num_prototypes = prototypes.shape[0]
    print(f"\nNumber of prototypes: {num_prototypes}")
    print(f"Prototype dimension: {prototypes.shape[1]}")

    # Collect image features and labels from validation set
    print(f"\nCollecting features from {args.num_retrieval_samples} samples...")
    all_features = []
    all_labels = []
    all_similarities = []  # Prototype similarity maps

    total_collected = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Extracting features")
        for batch_idx, batch in enumerate(pbar):
            if total_collected >= args.num_retrieval_samples:
                break

            # Handle both (images, captions) and (images, targets, captions) formats
            if len(batch) == 2:
                images, captions = batch
                # Extract class indices from dataset
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(val_loader.dataset))
                labels = [val_loader.dataset.samples[i][1] for i in range(start_idx, end_idx)]
            else:
                images, labels, captions = batch
                labels = labels.tolist() if torch.is_tensor(labels) else labels

            images = images.to(args.device)

            # Get features and prototype similarities
            # We need the feature maps before pooling
            features = model.image_encoder.backbone(images)  # (B, C, H, W)
            proto_similarities = model.image_encoder.prototype_layer(features)  # (B, num_proto, H, W)

            # Store max similarity for each prototype across spatial locations
            max_similarities = proto_similarities.amax(dim=(2, 3))  # (B, num_proto)

            all_features.append(features.cpu())
            all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
            all_similarities.append(max_similarities.cpu())

            total_collected += images.size(0)
            pbar.set_postfix({'Collected': total_collected})

    all_features = torch.cat(all_features, dim=0)[:args.num_retrieval_samples]
    all_similarities = torch.cat(all_similarities, dim=0)[:args.num_retrieval_samples]
    all_labels = all_labels[:args.num_retrieval_samples]

    print(f"\nCollected {len(all_labels)} samples")

    # For each prototype, find top-k most similar images
    print(f"\nFinding top-{args.top_k_retrieval} matches for each prototype...")
    prototype_matches = {}

    for proto_idx in range(num_prototypes):
        # Get similarity scores for this prototype across all images
        similarities = all_similarities[:, proto_idx]  # (num_samples,)

        # Get top-k most similar images
        top_k_values, top_k_indices = similarities.topk(args.top_k_retrieval)

        # Get corresponding labels and captions
        matches = []
        for rank, (idx, sim) in enumerate(zip(top_k_indices, top_k_values)):
            label = all_labels[idx.item()]
            caption = caption_generator.generate_caption(label)
            matches.append({
                'rank': rank + 1,
                'image_idx': idx.item(),
                'label': label,
                'caption': caption,
                'similarity': sim.item()
            })

        prototype_matches[proto_idx] = matches

    # Print results
    print("\n" + "=" * 70)
    print("PROTOTYPE IMAGE-TO-TEXT RETRIEVAL RESULTS")
    print("=" * 70)

    # Show detailed results for first 10 prototypes
    num_to_show = min(10, num_prototypes)
    print(f"\nShowing top-{args.top_k_retrieval} matches for first {num_to_show} prototypes:\n")

    for proto_idx in range(num_to_show):
        print(f"Prototype {proto_idx}:")
        for match in prototype_matches[proto_idx]:
            print(f"  {match['rank']}. {match['caption']:<40} (similarity: {match['similarity']:.4f})")
        print()

    # Compute diversity statistics
    print("\n" + "=" * 70)
    print("DIVERSITY STATISTICS")
    print("=" * 70)

    # Count unique classes per prototype
    unique_classes_per_proto = []
    for proto_idx in range(num_prototypes):
        unique_classes = set([m['label'] for m in prototype_matches[proto_idx]])
        unique_classes_per_proto.append(len(unique_classes))

    avg_unique_classes = sum(unique_classes_per_proto) / num_prototypes
    print(f"Average unique classes per prototype (top-{args.top_k_retrieval}): {avg_unique_classes:.2f}")

    # Count how many prototypes are specialized (top-k all same class)
    specialized_count = sum([1 for count in unique_classes_per_proto if count == 1])
    specialized_percent = 100.0 * specialized_count / num_prototypes
    print(f"Specialized prototypes (single class): {specialized_count}/{num_prototypes} ({specialized_percent:.1f}%)")

    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        # Set up output directory
        if args.viz_output_dir is None:
            checkpoint_dir = Path(args.checkpoint).parent
            viz_dir = checkpoint_dir / "visualizations" / "prototype_matches"
        else:
            viz_dir = Path(args.viz_output_dir)

        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving visualizations to: {viz_dir}")

        # Generate visualizations for top N prototypes
        num_to_viz = min(args.num_prototypes_to_visualize, num_prototypes)
        print(f"Generating grids for {num_to_viz} prototypes...")

        saved_files = []
        for proto_idx in tqdm(range(num_to_viz), desc="Creating visualizations"):
            output_path = visualize_prototype_matches(
                prototype_idx=proto_idx,
                matches=prototype_matches[proto_idx],
                val_loader=val_loader,
                output_dir=viz_dir,
                projection_info=projection_info,
                caption_generator=caption_generator,
                grid_cols=args.viz_grid_cols
            )
            saved_files.append(output_path)

        print(f"\n✓ Saved {len(saved_files)} visualization grids to {viz_dir}")
        print(f"  Example: {saved_files[0].name}")
        if projection_info:
            print(f"  Each grid shows the ORIGINAL prototype image (with red border) + top matches")

    return {
        'prototype_matches': prototype_matches,
        'avg_unique_classes': avg_unique_classes,
        'specialized_count': specialized_count,
        'specialized_percent': specialized_percent
    }


def save_results(results, checkpoint_path):
    """Save evaluation results to JSON file"""
    results_dir = Path(checkpoint_path).parent / "evaluation_results"
    results_dir.mkdir(exist_ok=True)

    timestamp = Path(checkpoint_path).stem
    results_file = results_dir / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return results_file


def main():
    args = parse_args()

    print("=" * 70)
    print("ProtoCLIP Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ImageNet root: {args.imagenet_root}")
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")

    # Load model
    model = load_model(args.checkpoint, args)

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

    # Create validation data loader
    print("\n" + "=" * 70)
    print("Creating Data Loader")
    print("=" * 70)

    _, val_loader = create_imagenet_loaders(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=None,
        val_transform=get_val_transforms(),
        use_subset=False,  # Use full validation set
        caption_generator=caption_generator
    )

    print(f"Validation samples: {len(val_loader.dataset)}")

    # Run evaluations
    results = {}

    if args.mode in ['zero-shot', 'both']:
        zero_shot_results = zero_shot_classification(model, val_loader, caption_generator, args)
        results['zero_shot'] = zero_shot_results

    if args.mode in ['retrieval', 'both']:
        retrieval_results = prototype_image_to_text_retrieval(model, val_loader, caption_generator, args)
        results['retrieval'] = retrieval_results

    # Save results
    results_file = save_results(results, args.checkpoint)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
