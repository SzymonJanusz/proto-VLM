"""
Quick start script for ProtoCLIP training.

This script automates the entire setup process:
1. Downloads Tiny-ImageNet (smallest, fastest option)
2. Downloads pretrained Proto-CLIP checkpoints
3. Downloads ImageNet class mapping
4. Runs a quick training test

Perfect for getting started quickly!

Usage:
    python scripts/quick_start.py
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False

    print(f"\n✓ Success: {description}")
    return True


def main():
    print("=" * 70)
    print("ProtoCLIP Quick Start")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Download Tiny-ImageNet (~250MB, 200 classes)")
    print("  2. Download pretrained Proto-CLIP checkpoints (~60MB)")
    print("  3. Download ImageNet class mappings")
    print("  4. Run a quick training test (5 epochs, 1000 samples)")
    print("\nTotal download size: ~310MB")
    print("Estimated time: 5-10 minutes")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Paths
    root_dir = Path(__file__).parent.parent
    imagenet_dir = root_dir / "imagenet_tiny"
    pretrained_dir = root_dir / "pretrained_checkpoints" / "proto_clip_imagenet"

    # Step 1: Download Tiny-ImageNet
    if imagenet_dir.exists():
        print(f"\n✓ Tiny-ImageNet already exists at: {imagenet_dir}")
    else:
        success = run_command(
            [sys.executable, "scripts/download_tiny_imagenet.py", "--output", str(imagenet_dir)],
            "Step 1: Download Tiny-ImageNet"
        )
        if not success:
            return

    # Step 2: Download pretrained Proto-CLIP
    if pretrained_dir.exists():
        print(f"\n✓ Proto-CLIP checkpoints already exist at: {pretrained_dir}")
    else:
        success = run_command(
            [sys.executable, "scripts/download_pretrained.py"],
            "Step 2: Download Proto-CLIP checkpoints"
        )
        if not success:
            return

    # Step 3: Verify dataset
    success = run_command(
        [sys.executable, "scripts/verify_imagenet.py", "--imagenet_root", str(imagenet_dir), "--quick"],
        "Step 3: Verify dataset"
    )
    if not success:
        print("\n⚠️  Dataset verification failed, but continuing...")

    # Step 4: Run quick training test
    print("\n" + "=" * 70)
    print("Step 4: Run Quick Training Test")
    print("=" * 70)
    print("\nThis will train for 5 epochs on 1000 samples")
    print("Estimated time: 5-10 minutes on GPU, 20-30 minutes on CPU")

    response = input("\nRun training test? (y/n): ")
    if response.lower() != 'y':
        print("\nSkipping training test.")
        print("\nYou can run training manually with:")
        print(f"  python scripts/train.py \\")
        print(f"      --imagenet_root {imagenet_dir} \\")
        print(f"      --pretrained_protoclip {pretrained_dir} \\")
        print(f"      --use_subset \\")
        print(f"      --subset_samples 1000")
        return

    success = run_command(
        [
            sys.executable, "scripts/train.py",
            "--imagenet_root", str(imagenet_dir),
            "--pretrained_protoclip", str(pretrained_dir),
            "--use_subset",
            "--subset_samples", "1000",
            "--warmup_epochs", "20",
            "--finetune_epochs", "30",
            "--batch_size", "32",
            "--skip_projection"  # Skip for quick test
        ],
        "Training test"
    )

    # Final summary
    print("\n" + "=" * 70)
    print("QUICK START COMPLETE!")
    print("=" * 70)

    if success:
        print("\n✓ Training test completed successfully!")
        print("\nCheckpoints saved to: ./checkpoints")
        print("\nNext steps:")
        print("  1. Check training results in ./checkpoints")
        print("  2. Run longer training:")
        print(f"       python scripts/train.py --imagenet_root {imagenet_dir} \\")
        print(f"           --pretrained_protoclip {pretrained_dir}")
        print("  3. Evaluate model:")
        print("       python scripts/evaluate.py --checkpoint ./checkpoints/finetune_best.pt")
        print("  4. Visualize prototypes:")
        print("       python scripts/visualize_prototypes.py --checkpoint ./checkpoints/finetune_best.pt")
    else:
        print("\n⚠️  Training test failed!")
        print("\nDebugging tips:")
        print("  1. Check error messages above")
        print("  2. Verify dataset: python scripts/verify_imagenet.py --imagenet_root", imagenet_dir)
        print("  3. Try CPU mode: add --device cpu to train.py")
        print("  4. Reduce batch size: add --batch_size 16 to train.py")

    print("=" * 70)


if __name__ == '__main__':
    main()
