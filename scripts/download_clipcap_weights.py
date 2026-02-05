#!/usr/bin/env python
"""
Download Pretrained ClipCap Weights

Downloads pretrained ClipCap model from the official repository:
https://github.com/rmokady/CLIP_prefix_caption

Usage:
    python scripts/download_clipcap_weights.py --output_dir pretrained_checkpoints/clipcap
"""

import argparse
import os
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='', flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\nDownload complete!")


def main():
    parser = argparse.ArgumentParser(description='Download pretrained ClipCap weights')
    parser.add_argument('--output_dir', type=str, default='pretrained_checkpoints/clipcap',
                       help='Directory to save weights')
    parser.add_argument('--model', type=str, default='coco',
                       choices=['coco', 'conceptual'],
                       help='Model variant (coco or conceptual)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ClipCap pretrained weights URLs (from official repo)
    WEIGHTS_URLS = {
        'coco': 'https://drive.google.com/uc?export=download&id=14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT',
        'conceptual': 'https://drive.google.com/uc?export=download&id=1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX'
    }

    url = WEIGHTS_URLS[args.model]
    output_path = os.path.join(args.output_dir, f'clipcap_{args.model}.pt')

    print("=" * 80)
    print(f"Downloading ClipCap pretrained weights ({args.model} dataset)")
    print("=" * 80)
    print()

    try:
        download_file(url, output_path)
        print()
        print("=" * 80)
        print("Download successful!")
        print("=" * 80)
        print(f"Weights saved to: {output_path}")
        print()
        print("To use with interpret_classes.py:")
        print(f"  --decoders clipcap \\")
        print(f"  --clipcap_model {output_path}")

    except Exception as e:
        print(f"\nError downloading weights: {e}")
        print()
        print("Manual download instructions:")
        print("1. Visit: https://github.com/rmokady/CLIP_prefix_caption")
        print("2. Download pretrained weights from the 'Pretrained Models' section")
        print(f"3. Save to: {output_path}")


if __name__ == '__main__':
    main()
