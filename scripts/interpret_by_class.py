#!/usr/bin/env python
"""
CLIP-based Prototype Interpretation by Class

Interprets prototypes grouped by their ground truth classes.
Useful for analyzing what visual concepts each class's prototypes represent.
"""

import argparse
import json
import os
import sys
import pickle
from collections import defaultdict
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
        description='Interpret ProtoPNet prototypes grouped by class',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interpret all prototypes grouped by class
  python scripts/interpret_by_class.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --patch_sizes 64 --output_dir results/by_class

  # Interpret top 3 prototypes per class only
  python scripts/interpret_by_class.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --patch_sizes 64 --max_per_class 3 --output_dir results/by_class

  # Multiple patch sizes with aggregation
  python scripts/interpret_by_class.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --patch_sizes 32,64,128 --aggregate mean --output_dir results/by_class
        """
    )

    # Required arguments
    parser.add_argument('--projection_info', type=str, required=True,
                        help='Path to projection_info.pkl file')
    parser.add_argument('--class_mapping', type=str, required=True,
                        help='Path to ImageNet class mapping JSON')

    # Prototype selection
    parser.add_argument('--max_per_class', type=int, default=None,
                        help='Maximum prototypes per class (default: all)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated list of class names to interpret (default: all)')

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
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--save_summary', action='store_true',
                        help='Save summary JSON per class')

    return parser.parse_args()


def load_class_prototype_mapping(
    projection_info_path: str,
    class_mapping_path: str
) -> Dict[str, List[int]]:
    """
    Load mapping from class names to prototype indices.

    Args:
        projection_info_path: Path to projection_info.pkl
        class_mapping_path: Path to ImageNet class mapping JSON

    Returns:
        Dictionary mapping class name to list of prototype indices
    """
    # Load projection info
    with open(projection_info_path, 'rb') as f:
        projection_info = pickle.load(f)

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # Get prototype labels (class indices)
    prototype_labels = projection_info.get('class_id',
                                          projection_info.get('prototype_label', []))

    # Build mapping: class_name -> [proto_idx1, proto_idx2, ...]
    class_to_protos = defaultdict(list)

    for proto_idx, label_idx in enumerate(prototype_labels):
        if isinstance(label_idx, (list, tuple)):
            label_idx = label_idx[0] if len(label_idx) > 0 else 0

        class_name = class_mapping.get(str(label_idx), f"class_{label_idx}")
        class_to_protos[class_name].append(proto_idx)

    return dict(class_to_protos)


def interpret_prototype(
    proto_idx: int,
    generator: PrototypeImageGenerator,
    interpreter: CLIPPrototypeInterpreter,
    vocabulary: List[str],
    patch_sizes: List[int],
    background: str,
    aggregate_method: str,
    top_k: int,
    ground_truth: Optional[str] = None,
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
        ground_truth: Optional ground truth class name
        vlm_enricher: Optional VLM enricher for enhanced descriptions

    Returns:
        Dictionary with interpretation results
    """
    results = {
        'prototype_idx': proto_idx,
        'interpretations': {}
    }

    # Add ground truth if available
    if ground_truth:
        results['ground_truth'] = ground_truth

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

        # Generate VLM enrichment
        vlm_enrichment = vlm_enricher.enrich_clip_results(
            image=representative_synthetic,
            clip_results=clip_results,
            ground_truth=ground_truth
        )
        results['vlm_enrichment'] = vlm_enrichment

    return results


def save_class_summary(
    class_name: str,
    proto_results: List[Dict],
    output_path: str
):
    """
    Save a summary of interpretations for all prototypes in a class.

    Args:
        class_name: Name of the class
        proto_results: List of interpretation results for prototypes in this class
        output_path: Path to save the summary text file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CLASS: {class_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total prototypes: {len(proto_results)}\n\n")

        for result in proto_results:
            proto_idx = result['prototype_idx']
            f.write("-" * 80 + "\n")
            f.write(f"PROTOTYPE {proto_idx}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Image: {result['image_path']}\n")
            f.write(f"Location: {result['patch_location']}\n\n")

            # Get the key to use (aggregated if available, else first patch size)
            interpretations = result.get('interpretations', {})
            if 'aggregated' in interpretations:
                key = 'aggregated'
                f.write("AGGREGATED RESULTS:\n")
            else:
                key = list(interpretations.keys())[0]
                f.write(f"{key.upper()} RESULTS:\n")

            for rank, (text, similarity) in enumerate(interpretations[key], 1):
                f.write(f"  {rank}. {text} ({similarity:.4f})\n")

            f.write("\n")


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


def visualize_class_prototypes(
    class_name: str,
    proto_results: List[Dict],
    generator: PrototypeImageGenerator,
    patch_sizes: List[int],
    background: str,
    save_dir: str
):
    """
    Create a visualization showing all prototypes for a class.

    Args:
        class_name: Name of the class
        proto_results: List of interpretation results
        generator: PrototypeImageGenerator instance
        patch_sizes: List of patch sizes used
        background: Background type
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create one figure per prototype
    for result in proto_results:
        proto_idx = result['prototype_idx']

        # Create figure
        n_patches = len(patch_sizes)
        fig_width = 16
        fig_height = 5 + (n_patches * 3)

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

        # Create grid: 3 columns (Ground Truth | Patch | Synthetic)
        gs = GridSpec(n_patches + 1, 3, figure=fig, hspace=0.4, wspace=0.3,
                      height_ratios=[1.5] + [3]*n_patches)

        # Row 0: Ground truth image
        ax_gt = fig.add_subplot(gs[0, :])
        gt_img = generator.get_original_image(proto_idx)
        h, w = generator.get_patch_location(proto_idx)

        # Draw red box around prototype location
        ax_gt.imshow(gt_img)
        receptive_field = 14
        rect = mpatches.Rectangle((w*receptive_field, h*receptive_field),
                                  receptive_field, receptive_field,
                                  linewidth=2, edgecolor='red', facecolor='none')
        ax_gt.add_patch(rect)
        ax_gt.axis('off')

        # Title
        title = f"Prototype {proto_idx} | Class: {class_name} | Location: {result['patch_location']}"
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

            # Get top-3 interpretations
            key = f'patch_size_{patch_size}'
            if key in result.get('interpretations', {}):
                top_3 = result['interpretations'][key][:3]
                preview_text = f"Top-3 for {patch_size}×{patch_size}:\n\n"
                for rank, (text, similarity) in enumerate(top_3, 1):
                    # Truncate long text
                    display_text = text if len(text) <= 40 else text[:37] + "..."
                    preview_text += f"{rank}. {display_text}\n"
                    preview_text += f"   ({similarity:.4f})\n\n"

                ax_preview.text(0.05, 0.95, preview_text,
                              transform=ax_preview.transAxes,
                              fontsize=10, verticalalignment='top',
                              fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'Class: {class_name} - Prototype {proto_idx}',
                     fontsize=18, fontweight='bold', y=0.995)

        # Save figure
        save_path = os.path.join(save_dir, f'prototype_{proto_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # Save detailed interpretations to text file
        text_path = os.path.join(save_dir, f'prototype_{proto_idx}_interpretations.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"PROTOTYPE {proto_idx} - CLASS: {class_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source Image: {result['image_path']}\n")
            f.write(f"Patch Location: {result['patch_location']}\n\n")

            # Write all interpretations
            interpretations = result.get('interpretations', {})
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

        # Save VLM description if available
        if 'vlm_enrichment' in result:
            vlm_path = os.path.join(save_dir, f'prototype_{proto_idx}_vlm_description.txt')
            save_vlm_description_to_file(result, vlm_path)


def main():
    """Main function."""
    args = parse_args()

    # Parse patch sizes
    patch_sizes = [int(s.strip()) for s in args.patch_sizes.split(',')]

    # Parse class filter if provided
    class_filter = None
    if args.classes:
        class_filter = set(c.strip() for c in args.classes.split(','))

    print("=" * 80)
    print("CLIP-based Prototype Interpretation by Class")
    print("=" * 80)
    print(f"Projection info: {args.projection_info}")
    print(f"Class mapping: {args.class_mapping}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    if args.max_per_class:
        print(f"Max prototypes per class: {args.max_per_class}")
    if class_filter:
        print(f"Class filter: {class_filter}")
    print("=" * 80)
    print()

    # Load class-to-prototype mapping
    print("Loading class-to-prototype mapping...")
    class_to_protos = load_class_prototype_mapping(
        args.projection_info,
        args.class_mapping
    )
    print(f"  Found {len(class_to_protos)} classes")

    # Apply class filter if specified
    if class_filter:
        class_to_protos = {
            cls: protos for cls, protos in class_to_protos.items()
            if cls in class_filter
        }
        print(f"  Filtered to {len(class_to_protos)} classes")

    # Print class distribution
    print("\nClass distribution:")
    for cls, protos in sorted(class_to_protos.items()):
        count = len(protos)
        if args.max_per_class:
            count = min(count, args.max_per_class)
        print(f"  {cls}: {count} prototypes")
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each class
    total_classes = len(class_to_protos)
    total_prototypes = 0

    for class_idx, (class_name, proto_indices) in enumerate(sorted(class_to_protos.items()), 1):
        print(f"[{class_idx}/{total_classes}] Processing class: {class_name}")
        print(f"  Prototypes: {len(proto_indices)}")

        # Limit prototypes per class if specified
        if args.max_per_class:
            proto_indices = proto_indices[:args.max_per_class]

        # Interpret each prototype in this class
        class_results = []

        for proto_idx in proto_indices:
            total_prototypes += 1
            print(f"    Prototype {proto_idx}...", end=' ', flush=True)

            result = interpret_prototype(
                proto_idx=proto_idx,
                generator=generator,
                interpreter=interpreter,
                vocabulary=vocabulary,
                patch_sizes=patch_sizes,
                background=args.background,
                aggregate_method=args.aggregate,
                top_k=args.top_k,
                ground_truth=class_name,
                vlm_enricher=vlm_enricher
            )

            class_results.append(result)
            print("Done")

        # Save results for this class
        class_dir = os.path.join(args.output_dir, class_name.replace(' ', '_'))
        os.makedirs(class_dir, exist_ok=True)

        # Save summary
        if args.save_summary or not args.save_images:
            summary_path = os.path.join(class_dir, 'summary.txt')
            save_class_summary(class_name, class_results, summary_path)
            print(f"  Saved summary to: {summary_path}")

        # Save JSON
        json_path = os.path.join(class_dir, 'interpretations.json')
        with open(json_path, 'w') as f:
            json.dump({
                'class_name': class_name,
                'num_prototypes': len(proto_indices),
                'prototypes': class_results
            }, f, indent=2)
        print(f"  Saved JSON to: {json_path}")

        # Save visualizations
        if args.save_images:
            visualize_class_prototypes(
                class_name=class_name,
                proto_results=class_results,
                generator=generator,
                patch_sizes=patch_sizes,
                background=args.background,
                save_dir=class_dir
            )
            print(f"  Saved visualizations to: {class_dir}")

        print()

    # Save overall summary
    summary_data = {
        'total_classes': len(class_to_protos),
        'total_prototypes': total_prototypes,
        'patch_sizes': patch_sizes,
        'clip_model': args.clip_model,
        'vocabulary': args.vocabulary,
        'classes': {
            cls: len(protos) if not args.max_per_class else min(len(protos), args.max_per_class)
            for cls, protos in class_to_protos.items()
        }
    }

    summary_path = os.path.join(args.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"Total classes processed: {len(class_to_protos)}")
    print(f"Total prototypes interpreted: {total_prototypes}")
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
