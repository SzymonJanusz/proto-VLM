"""
Inspect Proto-CLIP checkpoint structure and dimensions.

This script helps understand the checkpoint format before implementing the converter.

Usage:
    python scripts/inspect_checkpoints.py
    python scripts/inspect_checkpoints.py --checkpoint_dir ./custom_path/
"""

import sys
from pathlib import Path
import torch
import argparse


def inspect_checkpoint(checkpoint_path, verbose=True):
    """
    Inspect checkpoint structure and print details.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        verbose: If True, print detailed information

    Returns:
        dict: Checkpoint content
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"Inspecting: {checkpoint_path.name}")
    print(f"{'=' * 70}")

    try:
        # Load checkpoint to CPU
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Print file size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

        # Print checkpoint type
        print(f"Type: {type(ckpt)}")

        # Analyze structure
        if isinstance(ckpt, dict):
            print(f"\nCheckpoint keys: {list(ckpt.keys())}")
            print(f"\nDetailed structure:")
            print("-" * 70)

            for key, value in ckpt.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}:")
                    print(f"    - Type: Tensor")
                    print(f"    - Shape: {value.shape}")
                    print(f"    - Dtype: {value.dtype}")
                    print(f"    - Device: {value.device}")
                    if value.numel() > 0:
                        print(f"    - Min: {value.min().item():.4f}")
                        print(f"    - Max: {value.max().item():.4f}")
                        print(f"    - Mean: {value.float().mean().item():.4f}")
                        print(f"    - Std: {value.float().std().item():.4f}")

                elif isinstance(value, dict):
                    print(f"  {key}:")
                    print(f"    - Type: Dictionary")
                    print(f"    - Keys: {list(value.keys())}")
                    if verbose:
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                print(f"      {subkey}: Tensor {subvalue.shape}")
                            else:
                                print(f"      {subkey}: {type(subvalue)}")

                elif isinstance(value, list):
                    print(f"  {key}:")
                    print(f"    - Type: List")
                    print(f"    - Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    - First element type: {type(value[0])}")

                else:
                    print(f"  {key}: {type(value)} = {value}")

        elif isinstance(ckpt, torch.Tensor):
            print(f"\nCheckpoint is a single tensor:")
            print(f"  Shape: {ckpt.shape}")
            print(f"  Dtype: {ckpt.dtype}")
            print(f"  Min: {ckpt.min().item():.4f}")
            print(f"  Max: {ckpt.max().item():.4f}")
            print(f"  Mean: {ckpt.float().mean().item():.4f}")

        else:
            print(f"\nUnexpected checkpoint type: {type(ckpt)}")

        return ckpt

    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_all_checkpoints(checkpoint_dir="./pretrained_checkpoints/proto_clip_imagenet", verbose=True):
    """
    Inspect all Proto-CLIP checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        verbose: Print detailed information

    Returns:
        dict: All loaded checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        print("Run: python scripts/download_pretrained.py")
        return {}

    checkpoint_files = ["memory_bank_v.pt", "memory_bank_t.pt", "query_adapter.pt"]
    checkpoints = {}

    print("\n" + "=" * 70)
    print("Proto-CLIP Checkpoint Inspector")
    print("=" * 70)
    print(f"Directory: {checkpoint_dir}")

    for filename in checkpoint_files:
        filepath = checkpoint_dir / filename
        ckpt = inspect_checkpoint(filepath, verbose=verbose)
        if ckpt is not None:
            checkpoints[filename] = ckpt

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoints inspected: {len(checkpoints)}/{len(checkpoint_files)}")

    if "memory_bank_v.pt" in checkpoints:
        print("\n✓ Visual Memory Bank loaded successfully")
        print("  This will be used to initialize PrototypeLayer")

    if "memory_bank_t.pt" in checkpoints:
        print("\n✓ Text Memory Bank loaded successfully")
        print("  (Optional - can be used for reference)")

    if "query_adapter.pt" in checkpoints:
        print("\n✓ Query Adapter loaded successfully")
        print("  This may be used to initialize projection head (if compatible)")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Implement checkpoint_converter.py based on the structure above")
    print("  2. Map checkpoints to ProtoPNet-CLIP model components")
    print("  3. Test loading checkpoints into model")
    print("=" * 70)

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Proto-CLIP checkpoint structure',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--checkpoint_dir', type=str,
                        default='./pretrained_checkpoints/proto_clip_imagenet',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Inspect specific checkpoint file instead of all')
    parser.add_argument('--brief', action='store_true',
                        help='Brief output (less verbose)')

    args = parser.parse_args()

    if args.checkpoint:
        # Inspect single checkpoint
        inspect_checkpoint(args.checkpoint, verbose=not args.brief)
    else:
        # Inspect all checkpoints in directory
        inspect_all_checkpoints(args.checkpoint_dir, verbose=not args.brief)


if __name__ == '__main__':
    main()
