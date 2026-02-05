#!/usr/bin/env python
"""
CLIP-based Prototype Interpretation CLI

Interprets ProtoPNet prototypes using CLIP's pre-trained image encoder.
No additional training required.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ProtoPNet.utils.prototype_interpretation import (
    PrototypeImageGenerator,
    CLIPPrototypeInterpreter
)
from ProtoPNet.utils.clip_vocabulary import get_vocabulary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Interpret ProtoPNet prototypes using CLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prototype with visualization
  python scripts/interpret_prototypes.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --prototype_id 0 --patch_sizes 64 --visualize --save_images --output_dir results

  # Multiple patch sizes
  python scripts/interpret_prototypes.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --prototype_id 0 --patch_sizes 32,64,128 --visualize --save_images --output_dir results

  # Batch all prototypes
  python scripts/interpret_prototypes.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --all_prototypes --patch_sizes 64 --output_json results/all_interpretations.json
        """
    )

    # Required arguments
    parser.add_argument('--projection_info', type=str, required=True,
                        help='Path to projection_info.pkl file')

    # Prototype selection (mutually exclusive)
    proto_group = parser.add_mutually_exclusive_group(required=True)
    proto_group.add_argument('--prototype_id', type=int,
                             help='ID of prototype to interpret')
    proto_group.add_argument('--all_prototypes', action='store_true',
                             help='Interpret all prototypes')

    # Image generation parameters
    parser.add_argument('--patch_sizes', type=str, default='64',
                        help='Comma-separated patch sizes (default: 64)')
    parser.add_argument('--background', type=str, default='noise',
                        choices=['noise', 'gray', 'random', 'average'],
                        help='Background type for synthetic images (default: noise)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of synthetic images (default: 224)')

    # CLIP model parameters
    parser.add_argument('--clip_model', type=str,
                        default='openai/clip-vit-base-patch32',
                        help='CLIP model name (default: openai/clip-vit-base-patch32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Vocabulary parameters
    parser.add_argument('--vocabulary', type=str, default='imagenet',
                        choices=['imagenet', 'custom', 'hybrid'],
                        help='Vocabulary type (default: imagenet)')
    parser.add_argument('--class_mapping', type=str,
                        help='Path to ImageNet class mapping JSON (required for imagenet/hybrid)')
    parser.add_argument('--custom_domain', type=str, default='general',
                        choices=['general', 'animals', 'objects'],
                        help='Custom vocabulary domain (default: general)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top interpretations to return (default: 5)')

    # Aggregation parameters
    parser.add_argument('--aggregate', type=str, default='mean',
                        choices=['mean', 'max', 'vote'],
                        help='Aggregation method for multiple patch sizes (default: mean)')

    # VLM parameters
    parser.add_argument('--use_vlm', action='store_true',
                        help='Enable VLM enrichment of CLIP results')
    parser.add_argument('--vlm_backend', type=str, default='blip',
                        choices=['blip', 'git', 'llava'],
                        help='VLM backend to use (default: blip)')
    parser.add_argument('--vlm_device', type=str, default=None,
                        help='Device for VLM (default: same as CLIP device)')

    # Output parameters
    parser.add_argument('--output_json', type=str,
                        help='Path to save results as JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images to disk')
    parser.add_argument('--output_dir', type=str, default='interpretation_results',
                        help='Output directory for images (default: interpretation_results)')

    args = parser.parse_args()

    # Validate arguments
    if args.vocabulary in ['imagenet', 'hybrid'] and not args.class_mapping:
        parser.error('--class_mapping is required when using imagenet or hybrid vocabulary')

    return args


def save_interpretations_to_file(results: Dict, filepath: str):
    """
    Save all interpretation results to a readable text file.

    Args:
        results: Dictionary containing interpretation results
        filepath: Path to save the text file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"PROTOTYPE {results['prototype_idx']} INTERPRETATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        f.write(f"Ground Truth Class: {results.get('ground_truth', 'Unknown')}\n")
        if 'image_path' in results:
            f.write(f"Source Image: {results['image_path']}\n")
        if 'patch_location' in results:
            f.write(f"Patch Location: {results['patch_location']}\n")
        f.write("\n")

        # Individual patch size results
        interpretations = results.get('interpretations', {})
        for key in sorted(interpretations.keys()):
            if key.startswith('patch_size_'):
                patch_size = key.replace('patch_size_', '')
                f.write("-" * 80 + "\n")
                f.write(f"PATCH SIZE {patch_size}\n")
                f.write("-" * 80 + "\n\n")

                for rank, (text, similarity) in enumerate(interpretations[key], 1):
                    f.write(f"Rank {rank}:\n")
                    f.write(f"  Interpretation: {text}\n")
                    f.write(f"  Similarity: {similarity:.6f}\n\n")

        # Aggregated results
        if 'aggregated' in interpretations:
            f.write("-" * 80 + "\n")
            f.write("AGGREGATED RESULTS\n")
            f.write("-" * 80 + "\n\n")

            for rank, (text, similarity) in enumerate(interpretations['aggregated'], 1):
                f.write(f"Rank {rank}:\n")
                f.write(f"  Interpretation: {text}\n")
                f.write(f"  Similarity: {similarity:.6f}\n\n")


def save_vlm_description_to_file(results: Dict, filepath: str):
    """
    Save VLM enrichment results to a readable text file.

    Args:
        results: Dictionary containing results with 'vlm_enrichment' key
        filepath: Path to save the VLM description file
    """
    if 'vlm_enrichment' not in results:
        return

    vlm_data = results['vlm_enrichment']
    proto_idx = results['prototype_idx']

    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"PROTOTYPE {proto_idx} - VLM INTERPRETATION\n")
        f.write("=" * 80 + "\n")

        # Metadata
        if 'ground_truth' in results:
            f.write(f"Ground Truth Class: {results['ground_truth']}\n")
        if 'image_path' in results:
            f.write(f"Source Image: {results['image_path']}\n")
        if 'patch_location' in results:
            f.write(f"Patch Location: {results['patch_location']}\n")
        f.write("\n")

        # Detailed Visual Description
        if 'detailed_description' in vlm_data:
            f.write("-" * 80 + "\n")
            f.write("DETAILED VISUAL DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write(vlm_data['detailed_description'] + "\n\n")

        # Object/Concept Identification
        if 'concept_identification' in vlm_data:
            f.write("-" * 80 + "\n")
            f.write("OBJECT/CONCEPT IDENTIFICATION\n")
            f.write("-" * 80 + "\n")
            f.write(vlm_data['concept_identification'] + "\n\n")

        # Discriminative Reasoning
        if 'discriminative_reasoning' in vlm_data:
            f.write("-" * 80 + "\n")
            f.write("DISCRIMINATIVE REASONING\n")
            f.write("-" * 80 + "\n")
            f.write(vlm_data['discriminative_reasoning'] + "\n\n")

        f.write("=" * 80 + "\n")


def visualize_prototype(
    results: Dict,
    generator: PrototypeImageGenerator,
    patch_sizes: List[int],
    background: str,
    save_path: Optional[str] = None
):
    """
    Visualize prototype images only (ground truth, patches, synthetic images).
    Interpretation results are saved to a separate text file.

    Args:
        results: Dictionary containing interpretation results
        generator: PrototypeImageGenerator instance
        patch_sizes: List of patch sizes used
        background: Background type used for synthetic images
        save_path: Optional path to save the figure
    """
    proto_idx = results['prototype_idx']

    # Create figure
    n_patches = len(patch_sizes)
    fig_width = 16
    fig_height = 5 + (n_patches * 3)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

    # Create grid: 3 columns (Ground Truth | Patch | Synthetic)
    # First row is taller for ground truth image
    gs = GridSpec(n_patches + 1, 3, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1.5] + [3]*n_patches)

    # Row 0: Ground truth image (spans all columns) and info
    ax_gt = fig.add_subplot(gs[0, :])
    gt_img = generator.get_original_image(proto_idx)
    h, w = generator.get_patch_location(proto_idx)

    # Draw red box around prototype location
    ax_gt.imshow(gt_img)
    receptive_field = 14  # Assuming 14x14 receptive field
    rect = mpatches.Rectangle((w*receptive_field, h*receptive_field),
                              receptive_field, receptive_field,
                              linewidth=2, edgecolor='red', facecolor='none')
    ax_gt.add_patch(rect)
    ax_gt.axis('off')

    # Title with metadata
    gt_class = results.get('ground_truth', 'Unknown')
    title = f"Prototype {proto_idx} | Ground Truth: {gt_class}"
    if 'patch_location' in results:
        title += f" | Location: {results['patch_location']}"
    ax_gt.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Rows 1+: One row per patch size
    for i, patch_size in enumerate(patch_sizes, 1):
        # Extract patch
        patch = generator.extract_patch(proto_idx, patch_size=patch_size)
        synthetic = generator.create_synthetic_image(patch, background=background)

        # Column 0: Patch
        ax_patch = fig.add_subplot(gs[i, 0])
        ax_patch.imshow(patch)
        ax_patch.axis('off')
        ax_patch.set_title(f'Patch {patch_size}×{patch_size}', fontsize=12)

        # Column 1: Synthetic image
        ax_synth = fig.add_subplot(gs[i, 1])
        ax_synth.imshow(synthetic)
        ax_synth.axis('off')
        ax_synth.set_title(f'Synthetic Image ({background})', fontsize=12)

        # Column 2: Top-3 preview
        ax_preview = fig.add_subplot(gs[i, 2])
        ax_preview.axis('off')

        # Get top-3 interpretations for this patch size
        key = f'patch_size_{patch_size}'
        if key in results.get('interpretations', {}):
            top_3 = results['interpretations'][key][:3]
            preview_text = f"Top-3 for {patch_size}×{patch_size}:\n\n"
            for rank, (text, similarity) in enumerate(top_3, 1):
                preview_text += f"{rank}. {text[:40]}...\n"
                preview_text += f"   ({similarity:.4f})\n\n"

            ax_preview.text(0.05, 0.95, preview_text,
                          transform=ax_preview.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Prototype {proto_idx} Visualization',
                 fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")

        # Also save interpretation results to text file
        text_path = save_path.replace('.png', '_interpretations.txt')
        save_interpretations_to_file(results, text_path)
        print(f"Saved interpretations to: {text_path}")

    return fig


def interpret_single_prototype(
    proto_idx: int,
    generator: PrototypeImageGenerator,
    interpreter: CLIPPrototypeInterpreter,
    vocabulary: List[str],
    patch_sizes: List[int],
    background: str,
    aggregate_method: str,
    top_k: int,
    ground_truth_map: Optional[Dict[int, str]] = None,
    vlm_enricher: Optional[any] = None
) -> Dict:
    """
    Interpret a single prototype at multiple patch sizes.

    Args:
        proto_idx: Prototype index
        generator: PrototypeImageGenerator instance
        interpreter: CLIPPrototypeInterpreter instance
        vocabulary: List of text candidates
        patch_sizes: List of patch sizes
        background: Background type
        aggregate_method: Aggregation method
        top_k: Number of top results
        ground_truth_map: Optional mapping from prototype index to class name

    Returns:
        Dictionary with interpretation results
    """
    results = {
        'prototype_idx': proto_idx,
        'interpretations': {}
    }

    # Add ground truth if available
    if ground_truth_map and proto_idx in ground_truth_map:
        results['ground_truth'] = ground_truth_map[proto_idx]

    # Add metadata
    h, w = generator.get_patch_location(proto_idx)
    results['patch_location'] = (h, w)
    results['image_path'] = str(generator.get_image_path(proto_idx))

    # Interpret at each patch size
    size_results = []
    representative_synthetic = None  # Save for VLM enrichment

    for patch_size in patch_sizes:
        patch = generator.extract_patch(proto_idx, patch_size=patch_size)
        synthetic = generator.create_synthetic_image(patch, background=background)

        # Save first synthetic image for VLM
        if representative_synthetic is None:
            representative_synthetic = synthetic

        interpretations = interpreter.interpret_prototype(
            synthetic, vocabulary, top_k=top_k
        )

        results['interpretations'][f'patch_size_{patch_size}'] = interpretations
        size_results.append(interpretations)

    # Aggregate results if multiple patch sizes
    if len(patch_sizes) > 1:
        aggregated = interpreter.aggregate_interpretations(
            size_results, method=aggregate_method, top_k=top_k
        )
        results['interpretations']['aggregated'] = aggregated

    # VLM enrichment if enabled
    if vlm_enricher and representative_synthetic:
        # Use aggregated results if available, otherwise first patch size results
        if 'aggregated' in results['interpretations']:
            clip_results = results['interpretations']['aggregated']
        else:
            first_key = f'patch_size_{patch_sizes[0]}'
            clip_results = results['interpretations'][first_key]

        # Get ground truth for VLM context
        ground_truth = results.get('ground_truth', None)

        # Generate VLM enrichment
        vlm_enrichment = vlm_enricher.enrich_clip_results(
            image=representative_synthetic,
            clip_results=clip_results,
            ground_truth=ground_truth
        )
        results['vlm_enrichment'] = vlm_enrichment

    return results


def load_ground_truth_mapping(
    projection_info_path: str,
    class_mapping_path: Optional[str]
) -> Optional[Dict[int, str]]:
    """
    Load ground truth class mapping for prototypes.

    Args:
        projection_info_path: Path to projection_info.pkl
        class_mapping_path: Path to ImageNet class mapping JSON

    Returns:
        Dictionary mapping prototype index to class name, or None
    """
    if not class_mapping_path:
        return None

    try:
        import pickle

        # Load projection info
        with open(projection_info_path, 'rb') as f:
            projection_info = pickle.load(f)

        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)

        # Get prototype labels (class indices)
        prototype_labels = projection_info.get('class_id',
                                              projection_info.get('prototype_label', []))

        # Map prototype index to class name
        gt_map = {}
        for proto_idx, label_idx in enumerate(prototype_labels):
            if isinstance(label_idx, (list, tuple)):
                label_idx = label_idx[0] if len(label_idx) > 0 else 0

            class_name = class_mapping.get(str(label_idx), f"class_{label_idx}")
            gt_map[proto_idx] = class_name

        return gt_map

    except Exception as e:
        print(f"Warning: Could not load ground truth mapping: {e}")
        return None


def main():
    """Main function."""
    args = parse_args()

    # Parse patch sizes
    patch_sizes = [int(s.strip()) for s in args.patch_sizes.split(',')]

    print("=" * 80)
    print("CLIP-based Prototype Interpretation")
    print("=" * 80)
    print(f"Projection info: {args.projection_info}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Device: {args.device}")
    print(f"Vocabulary: {args.vocabulary}")
    print("=" * 80)
    print()

    # Load components
    print("Loading components...")

    generator = PrototypeImageGenerator.from_file(args.projection_info)
    print(f"  Loaded {generator.num_prototypes} prototypes")

    interpreter = CLIPPrototypeInterpreter(
        clip_model_name=args.clip_model,
        device=args.device
    )
    print(f"  Loaded CLIP model: {args.clip_model}")

    vocabulary = get_vocabulary(
        args.vocabulary,
        imagenet_json_path=args.class_mapping,
        custom_domain=args.custom_domain
    )
    print(f"  Loaded vocabulary with {len(vocabulary)} candidates")

    # Load ground truth mapping
    ground_truth_map = load_ground_truth_mapping(
        args.projection_info,
        args.class_mapping
    )
    if ground_truth_map:
        print(f"  Loaded ground truth mapping for {len(ground_truth_map)} prototypes")

    # Load VLM enricher if enabled
    vlm_enricher = None
    if args.use_vlm:
        from ProtoPNet.utils.vlm_interpretation import get_vlm_interpreter
        from ProtoPNet.utils.prototype_interpretation import VLMPrototypeEnricher

        vlm_device = args.vlm_device if args.vlm_device else args.device
        print(f"  Loading VLM backend: {args.vlm_backend} on {vlm_device}...")

        vlm_interpreter = get_vlm_interpreter(
            backend=args.vlm_backend,
            device=vlm_device
        )
        vlm_enricher = VLMPrototypeEnricher(vlm_interpreter)
        print(f"  VLM enrichment enabled")

    print()

    # Determine which prototypes to interpret
    if args.all_prototypes:
        prototype_ids = list(range(generator.num_prototypes))
        print(f"Interpreting all {len(prototype_ids)} prototypes...")
    else:
        prototype_ids = [args.prototype_id]
        print(f"Interpreting prototype {args.prototype_id}...")

    print()

    # Interpret prototypes
    all_results = []

    for i, proto_idx in enumerate(prototype_ids, 1):
        print(f"[{i}/{len(prototype_ids)}] Prototype {proto_idx}...", end=' ', flush=True)

        results = interpret_single_prototype(
            proto_idx=proto_idx,
            generator=generator,
            interpreter=interpreter,
            vocabulary=vocabulary,
            patch_sizes=patch_sizes,
            background=args.background,
            aggregate_method=args.aggregate,
            top_k=args.top_k,
            ground_truth_map=ground_truth_map,
            vlm_enricher=vlm_enricher
        )

        all_results.append(results)
        print("Done")

        # Visualize if requested
        if args.visualize or args.save_images:
            save_path = None
            if args.save_images:
                viz_dir = os.path.join(args.output_dir, 'visualizations')
                save_path = os.path.join(viz_dir, f'prototype_{proto_idx}.png')

            fig = visualize_prototype(
                results=results,
                generator=generator,
                patch_sizes=patch_sizes,
                background=args.background,
                save_path=save_path
            )

            # Save VLM description if enabled and saving images
            if args.save_images and vlm_enricher and 'vlm_enrichment' in results:
                vlm_path = os.path.join(viz_dir, f'prototype_{proto_idx}_vlm_description.txt')
                save_vlm_description_to_file(results, vlm_path)
                print(f"  Saved VLM description to: {vlm_path}")

            if args.visualize:
                plt.show()
            else:
                plt.close(fig)

    print()

    # Save JSON if requested
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to: {args.output_json}")

    # Compute evaluation metrics if we have ground truth
    if ground_truth_map and args.all_prototypes:
        print()
        print("=" * 80)
        print("Evaluation Metrics")
        print("=" * 80)

        # Get aggregation key
        if len(patch_sizes) > 1:
            agg_key = 'aggregated'
        else:
            agg_key = f'patch_size_{patch_sizes[0]}'

        top1_correct = 0
        top5_correct = 0

        for result in all_results:
            proto_idx = result['prototype_idx']
            gt = result.get('ground_truth', '')

            if agg_key in result['interpretations']:
                interpretations = result['interpretations'][agg_key]

                # Check top-1
                if interpretations and gt.lower() in interpretations[0][0].lower():
                    top1_correct += 1

                # Check top-5
                for text, _ in interpretations[:5]:
                    if gt.lower() in text.lower():
                        top5_correct += 1
                        break

        top1_acc = 100.0 * top1_correct / len(all_results)
        top5_acc = 100.0 * top5_correct / len(all_results)

        print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{len(all_results)})")
        print(f"Top-5 Accuracy: {top5_acc:.2f}% ({top5_correct}/{len(all_results)})")
        print()

    print("Done!")


if __name__ == '__main__':
    main()
