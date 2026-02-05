#!/usr/bin/env python
"""
Compare Different Prototype Projector Configurations

Systematically tests different:
- Projector architectures (Spatial CNN vs Hierarchical Pooling)
- Top-k sparsity levels (k=1, 5, 10, 20)
- Training durations (25 epochs each for quick comparison)

Generates a comparison report with:
- Performance metrics for each configuration
- Training curves
- Best configuration recommendations

Usage:
    python scripts/compare_prototype_projectors.py \
        --backbone_checkpoint checkpoints/overfitting_fix_4/finetune_best.pt \
        --data_root imagenet_tiny \
        --output_dir experiments/projector_comparison

Output:
    experiments/projector_comparison/
    ├── spatial_cnn_k1/          # Spatial CNN, k=1
    ├── spatial_cnn_k5/          # Spatial CNN, k=5
    ├── spatial_cnn_k10/         # Spatial CNN, k=10
    ├── hierarchical_k1/         # Hierarchical Pooling, k=1
    ├── hierarchical_k5/         # Hierarchical Pooling, k=5
    ├── hierarchical_k10/        # Hierarchical Pooling, k=10
    └── comparison_report.json   # Summary of all results
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare different prototype projector configurations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model & checkpoints
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                       help='Path to trained ProtoPNet checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Base directory for comparison experiments')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to ImageNet root directory')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs per experiment (default: 25 for quick comparison)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')

    # Comparison configurations
    parser.add_argument('--projector_types', nargs='+',
                       default=['spatial_cnn', 'hierarchical'],
                       choices=['spatial_cnn', 'hierarchical'],
                       help='Projector types to compare')
    parser.add_argument('--k_values', nargs='+', type=int,
                       default=[1, 5, 10],
                       help='Top-k sparsity values to test')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    return parser.parse_args()


def run_training(config, args):
    """
    Run a single training configuration.

    Args:
        config: dict with 'projector_type' and 'top_k'
        args: parsed command-line arguments

    Returns:
        results: dict with training metrics
    """
    projector_type = config['projector_type']
    top_k = config['top_k']

    # Create output directory for this configuration
    exp_name = f"{projector_type}_k{top_k}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING: {exp_name}")
    print(f"  Projector: {projector_type}")
    print(f"  Top-k: {top_k}")
    print(f"  Output: {exp_dir}")
    print("=" * 80)

    # Build command
    cmd = [
        sys.executable,  # python
        "scripts/train_prototype_projector.py",
        "--backbone_checkpoint", args.backbone_checkpoint,
        "--data_root", args.data_root,
        "--output_dir", exp_dir,
        "--projector_type", projector_type,
        "--top_k", str(top_k),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device
    ]

    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    training_time = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ Training failed for {exp_name}")
        return {
            'projector_type': projector_type,
            'top_k': top_k,
            'status': 'failed',
            'training_time': training_time
        }

    # Load results
    try:
        history_path = os.path.join(exp_dir, 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        # Get final epoch results
        final_epoch = history[-1]

        results = {
            'projector_type': projector_type,
            'top_k': top_k,
            'status': 'success',
            'training_time': training_time,
            'final_epoch': final_epoch['epoch'],
            'train_loss': final_epoch['train_loss'],
            'prototype_val_loss': final_epoch['prototype_val_loss'],
            'prototype_i2t_acc': final_epoch['prototype_i2t_acc'],
            'prototype_t2i_acc': final_epoch['prototype_t2i_acc'],
            'clip_vit_val_loss': final_epoch['clip_vit_val_loss'],
            'clip_vit_i2t_acc': final_epoch['clip_vit_i2t_acc'],
            'clip_vit_t2i_acc': final_epoch['clip_vit_t2i_acc'],
            'embedding_similarity_mean': final_epoch['embedding_cosine_mean'],
            'embedding_similarity_std': final_epoch['embedding_cosine_std'],
        }

        print(f"\n✅ Training completed for {exp_name}")
        print(f"  Val Loss: {results['prototype_val_loss']:.4f}")
        print(f"  I2T/T2I Acc: {results['prototype_i2t_acc']:.2f}% / {results['prototype_t2i_acc']:.2f}%")
        print(f"  Embedding Similarity: {results['embedding_similarity_mean']:.4f}")
        print(f"  Training Time: {training_time/60:.1f} minutes")

        return results

    except Exception as e:
        print(f"❌ Failed to load results for {exp_name}: {e}")
        return {
            'projector_type': projector_type,
            'top_k': top_k,
            'status': 'error',
            'error': str(e),
            'training_time': training_time
        }


def generate_comparison_report(all_results, args):
    """
    Generate a comprehensive comparison report.

    Args:
        all_results: list of result dicts from each training run
        args: command-line arguments
    """
    report_path = os.path.join(args.output_dir, 'comparison_report.json')

    # Filter successful runs
    successful_results = [r for r in all_results if r['status'] == 'success']

    if not successful_results:
        print("\n❌ No successful training runs to compare!")
        return

    # Find best configuration by I2T accuracy
    best_by_accuracy = max(successful_results, key=lambda r: r['prototype_i2t_acc'])

    # Find best configuration by embedding similarity
    best_by_similarity = max(successful_results, key=lambda r: r['embedding_similarity_mean'])

    # Find best configuration by validation loss
    best_by_loss = min(successful_results, key=lambda r: r['prototype_val_loss'])

    # Create report
    report = {
        'experiment_config': {
            'backbone_checkpoint': args.backbone_checkpoint,
            'epochs_per_run': args.epochs,
            'batch_size': args.batch_size,
            'total_runs': len(all_results),
            'successful_runs': len(successful_results)
        },
        'best_by_accuracy': {
            'config': f"{best_by_accuracy['projector_type']}_k{best_by_accuracy['top_k']}",
            'i2t_accuracy': best_by_accuracy['prototype_i2t_acc'],
            'details': best_by_accuracy
        },
        'best_by_similarity': {
            'config': f"{best_by_similarity['projector_type']}_k{best_by_similarity['top_k']}",
            'embedding_similarity': best_by_similarity['embedding_similarity_mean'],
            'details': best_by_similarity
        },
        'best_by_loss': {
            'config': f"{best_by_loss['projector_type']}_k{best_by_loss['top_k']}",
            'val_loss': best_by_loss['prototype_val_loss'],
            'details': best_by_loss
        },
        'all_results': all_results
    }

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    print(f"\n📊 Best by I2T Accuracy:")
    print(f"  Config: {report['best_by_accuracy']['config']}")
    print(f"  Accuracy: {report['best_by_accuracy']['i2t_accuracy']:.2f}%")

    print(f"\n📊 Best by Embedding Similarity:")
    print(f"  Config: {report['best_by_similarity']['config']}")
    print(f"  Similarity: {report['best_by_similarity']['embedding_similarity']:.4f}")

    print(f"\n📊 Best by Validation Loss:")
    print(f"  Config: {report['best_by_loss']['config']}")
    print(f"  Loss: {report['best_by_loss']['val_loss']:.4f}")

    print(f"\n📁 Report saved to: {report_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'I2T Acc':<10} {'Embed Sim':<12} {'Val Loss':<10} {'Status'}")
    print("-" * 80)

    for result in all_results:
        config_name = f"{result['projector_type']}_k{result['top_k']}"
        if result['status'] == 'success':
            print(f"{config_name:<20} {result['prototype_i2t_acc']:<10.2f} "
                  f"{result['embedding_similarity_mean']:<12.4f} "
                  f"{result['prototype_val_loss']:<10.4f} {result['status']}")
        else:
            print(f"{config_name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<10} {result['status']}")

    print("=" * 80)


def main():
    """Main comparison pipeline."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PROTOTYPE PROJECTOR COMPARISON")
    print("=" * 80)
    print(f"Projector types: {args.projector_types}")
    print(f"Top-k values: {args.k_values}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Total configurations: {len(args.projector_types) * len(args.k_values)}")
    print("=" * 80)

    # Generate all configurations
    configs = [
        {'projector_type': ptype, 'top_k': k}
        for ptype in args.projector_types
        for k in args.k_values
    ]

    print(f"\nRunning {len(configs)} experiments...")

    # Run all training experiments
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting experiment...")
        result = run_training(config, args)
        all_results.append(result)

    # Generate comparison report
    generate_comparison_report(all_results, args)

    print("\n✅ All experiments completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
