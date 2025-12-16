"""
Utility to download pretrained checkpoints from Proto-CLIP repository.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_protoclip_imagenet(save_dir="./pretrained_checkpoints/proto_clip_imagenet"):
    """
    Download Proto-CLIP pretrained ImageNet checkpoints.

    Downloads 3 files:
    - memory_bank_v.pt: Visual/image memory bank (prototypes)
    - memory_bank_t.pt: Text memory bank (prototypes)
    - query_adapter.pt: Adapter network

    Args:
        save_dir: Directory to save checkpoints (default: ./pretrained_checkpoints/proto_clip_imagenet)

    Returns:
        dict: Paths to downloaded files
    """
    base_url = "https://github.com/IRVLUTD/Proto-CLIP/raw/main/pretrained_ckpt/imagenet-F/"
    files = {
        "visual_memory": "memory_bank_v.pt",
        "text_memory": "memory_bank_t.pt",
        "adapter": "query_adapter.pt"
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}

    print(f"Downloading Proto-CLIP ImageNet checkpoints to {save_dir}")
    print("=" * 60)

    for key, filename in files.items():
        url = base_url + filename
        output_path = save_dir / filename

        if output_path.exists():
            print(f"✓ {filename} already exists, skipping download")
            downloaded_paths[key] = str(output_path)
        else:
            print(f"Downloading {filename}...")
            try:
                download_file(url, output_path)
                downloaded_paths[key] = str(output_path)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
                raise

    print("=" * 60)
    print("✓ All checkpoints downloaded successfully!")
    print(f"\nCheckpoints location: {save_dir}")
    print("\nNext steps:")
    print("  1. Run checkpoint inspection: python scripts/inspect_checkpoints.py")
    print("  2. Load checkpoints into model using checkpoint_converter.py")

    return downloaded_paths


def verify_checkpoints(checkpoint_dir="./pretrained_checkpoints/proto_clip_imagenet"):
    """
    Verify that all required checkpoint files exist.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        bool: True if all files exist
    """
    checkpoint_dir = Path(checkpoint_dir)
    required_files = ["memory_bank_v.pt", "memory_bank_t.pt", "query_adapter.pt"]

    print(f"Verifying checkpoints in {checkpoint_dir}")
    all_exist = True

    for filename in required_files:
        filepath = checkpoint_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_exist = False

    if all_exist:
        print("\n✓ All checkpoint files present!")
    else:
        print("\n✗ Some checkpoint files are missing!")

    return all_exist


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download Proto-CLIP pretrained checkpoints')
    parser.add_argument('--save_dir', type=str,
                        default='./pretrained_checkpoints/proto_clip_imagenet',
                        help='Directory to save checkpoints')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify existing checkpoints without downloading')

    args = parser.parse_args()

    if args.verify_only:
        verify_checkpoints(args.save_dir)
    else:
        download_protoclip_imagenet(args.save_dir)
        print("\n" + "=" * 60)
        verify_checkpoints(args.save_dir)
