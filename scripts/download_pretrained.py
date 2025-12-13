"""
CLI script to download Proto-CLIP pretrained checkpoints.

Usage:
    python scripts/download_pretrained.py --dataset imagenet-F
    python scripts/download_pretrained.py --verify_only
"""

import sys
from pathlib import Path

# Add parent directory to path to import ProtoPNet modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProtoPNet.utils.download_pretrained import download_protoclip_imagenet, verify_checkpoints
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Download Proto-CLIP pretrained checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download ImageNet-F checkpoints
  python scripts/download_pretrained.py --dataset imagenet-F

  # Verify existing checkpoints
  python scripts/download_pretrained.py --verify_only

  # Download to custom directory
  python scripts/download_pretrained.py --dataset imagenet-F --save_dir ./my_checkpoints/
        """
    )

    parser.add_argument('--dataset', type=str, default='imagenet-F',
                        choices=['imagenet-F'],
                        help='Dataset name (currently only imagenet-F is supported)')
    parser.add_argument('--save_dir', type=str,
                        default='./pretrained_checkpoints/proto_clip_imagenet',
                        help='Directory to save checkpoints')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify existing checkpoints without downloading')

    args = parser.parse_args()

    print("=" * 70)
    print("Proto-CLIP Pretrained Model Downloader")
    print("=" * 70)

    if args.verify_only:
        print(f"\nVerifying checkpoints for dataset: {args.dataset}")
        success = verify_checkpoints(args.save_dir)
        sys.exit(0 if success else 1)
    else:
        print(f"\nDataset: {args.dataset}")
        print(f"Save directory: {args.save_dir}\n")

        try:
            if args.dataset == 'imagenet-F':
                downloaded_paths = download_protoclip_imagenet(args.save_dir)
                print("\n✓ Download complete!")
                print("\nDownloaded files:")
                for key, path in downloaded_paths.items():
                    print(f"  - {key}: {path}")
            else:
                print(f"✗ Unsupported dataset: {args.dataset}")
                sys.exit(1)

        except Exception as e:
            print(f"\n✗ Error during download: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
