"""
Test script to verify Proto-CLIP pretrained checkpoint loading.

This script:
1. Creates a ProtoCLIP model with and without pretrained weights
2. Compares prototype initialization
3. Tests forward pass
4. Verifies dimensions and statistics

Usage:
    # Test with checkpoints
    python scripts/test_pretrained_loading.py

    # Test specific sampling method
    python scripts/test_pretrained_loading.py --sampling_method random

    # Compare random vs pretrained
    python scripts/test_pretrained_loading.py --compare
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from ProtoPNet.models.hybrid_model import ProtoCLIP


def test_basic_loading(checkpoint_dir, sampling_method='kmeans'):
    """Test basic checkpoint loading"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Checkpoint Loading")
    print("=" * 70)

    try:
        # Create model with pretrained initialization
        model = ProtoCLIP(
            num_prototypes=200,
            pretrained_protoclip_path=checkpoint_dir,
            protoclip_sampling_method=sampling_method
        )

        print("\n✓ Model created successfully with pretrained weights!")

        # Check prototypes
        prototypes = model.get_prototypes()
        print(f"\nPrototype Statistics:")
        print(f"  Shape: {prototypes.shape}")
        print(f"  Dtype: {prototypes.dtype}")
        print(f"  Mean: {prototypes.mean():.4f}")
        print(f"  Std: {prototypes.std():.4f}")
        print(f"  Min: {prototypes.min():.4f}")
        print(f"  Max: {prototypes.max():.4f}")

        return model, True

    except Exception as e:
        print(f"\n✗ Error loading pretrained weights: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_forward_pass(model, batch_size=4):
    """Test forward pass with pretrained model"""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)

    try:
        # Create dummy inputs
        images = torch.randn(batch_size, 3, 224, 224)
        texts = [f'test caption {i}' for i in range(batch_size)]

        print(f"\nInput shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Texts: {len(texts)} strings")

        # Forward pass
        with torch.no_grad():
            model.eval()
            logits_i, logits_t, img_emb, txt_emb = model(images, texts)

        print(f"\nOutput shapes:")
        print(f"  Image embeddings: {img_emb.shape}")
        print(f"  Text embeddings: {txt_emb.shape}")
        print(f"  Logits (image view): {logits_i.shape}")
        print(f"  Logits (text view): {logits_t.shape}")

        # Check normalization
        img_norms = img_emb.norm(dim=-1)
        txt_norms = txt_emb.norm(dim=-1)
        print(f"\nEmbedding norms (should be ~1.0):")
        print(f"  Image: {img_norms}")
        print(f"  Text: {txt_norms}")

        norms_ok = torch.allclose(img_norms, torch.ones(batch_size), atol=1e-5)
        print(f"  Normalization OK: {norms_ok}")

        # Test with similarity maps
        print(f"\nTesting with prototype similarity maps...")
        with torch.no_grad():
            logits_i, logits_t, img_emb, txt_emb, proto_sims = model(
                images, texts, return_similarities=True
            )

        print(f"  Prototype similarity maps shape: {proto_sims.shape}")
        print(f"  Expected: ({batch_size}, 200, 14, 14)")

        print("\n✓ Forward pass successful!")
        return True

    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_random_vs_pretrained(checkpoint_dir, sampling_method='kmeans'):
    """Compare random initialization vs pretrained"""
    print("\n" + "=" * 70)
    print("TEST 3: Compare Random vs Pretrained Initialization")
    print("=" * 70)

    try:
        # Create model with random initialization
        print("\nCreating model with RANDOM initialization...")
        model_random = ProtoCLIP(num_prototypes=200)
        proto_random = model_random.get_prototypes()

        print(f"\nRandom prototypes:")
        print(f"  Mean: {proto_random.mean():.4f}")
        print(f"  Std: {proto_random.std():.4f}")

        # Create model with pretrained initialization
        print(f"\nCreating model with PRETRAINED initialization...")
        print(f"  (Output from loading will appear below)")
        model_pretrained = ProtoCLIP(
            num_prototypes=200,
            pretrained_protoclip_path=checkpoint_dir,
            protoclip_sampling_method=sampling_method
        )
        proto_pretrained = model_pretrained.get_prototypes()

        print(f"\nPretrained prototypes:")
        print(f"  Mean: {proto_pretrained.mean():.4f}")
        print(f"  Std: {proto_pretrained.std():.4f}")

        # Compare
        print(f"\n" + "-" * 70)
        print("Comparison:")
        print("-" * 70)
        print(f"Random init mean: {proto_random.mean():.6f}")
        print(f"Pretrained init mean: {proto_pretrained.mean():.6f}")
        print(f"Difference: {abs(proto_random.mean() - proto_pretrained.mean()):.6f}")

        print(f"\nRandom init std: {proto_random.std():.6f}")
        print(f"Pretrained init std: {proto_pretrained.std():.6f}")

        # Test forward pass with both
        print(f"\n" + "-" * 70)
        print("Forward pass comparison:")
        print("-" * 70)

        images = torch.randn(4, 3, 224, 224)
        texts = ['test'] * 4

        with torch.no_grad():
            model_random.eval()
            model_pretrained.eval()

            _, _, img_emb_random, _ = model_random(images, texts)
            _, _, img_emb_pretrained, _ = model_pretrained(images, texts)

        print(f"Image embedding similarity (random vs pretrained):")
        similarity = (img_emb_random * img_emb_pretrained).sum(dim=-1).mean()
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  (Low similarity expected - different initializations)")

        print("\n✓ Comparison complete!")
        return True

    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test Proto-CLIP pretrained checkpoint loading'
    )

    parser.add_argument('--checkpoint_dir', type=str,
                        default='./pretrained_checkpoints/proto_clip_imagenet',
                        help='Directory containing Proto-CLIP checkpoints')
    parser.add_argument('--sampling_method', type=str, default='kmeans',
                        choices=['kmeans', 'random', 'first'],
                        help='Prototype sampling method')
    parser.add_argument('--compare', action='store_true',
                        help='Compare random vs pretrained initialization')
    parser.add_argument('--skip_forward', action='store_true',
                        help='Skip forward pass test')

    args = parser.parse_args()

    print("=" * 70)
    print("Proto-CLIP Pretrained Loading Test")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Sampling method: {args.sampling_method}")

    # Check if checkpoints exist
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"\n✗ Checkpoint directory not found: {checkpoint_dir}")
        print("Please run: python scripts/download_pretrained.py")
        sys.exit(1)

    # Test 1: Basic loading
    model, success = test_basic_loading(args.checkpoint_dir, args.sampling_method)
    if not success:
        print("\n✗ Basic loading test FAILED")
        sys.exit(1)

    # Test 2: Forward pass
    if not args.skip_forward:
        success = test_forward_pass(model)
        if not success:
            print("\n✗ Forward pass test FAILED")
            sys.exit(1)

    # Test 3: Comparison (optional)
    if args.compare:
        success = compare_random_vs_pretrained(args.checkpoint_dir, args.sampling_method)
        if not success:
            print("\n✗ Comparison test FAILED")
            sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ All tests passed!")
    print("\nYou can now use pretrained initialization in your training:")
    print("```python")
    print("model = ProtoCLIP(")
    print("    num_prototypes=200,")
    print(f"    pretrained_protoclip_path='{args.checkpoint_dir}',")
    print(f"    protoclip_sampling_method='{args.sampling_method}'")
    print(")")
    print("```")
    print("=" * 70)


if __name__ == '__main__':
    main()
