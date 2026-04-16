"""
Download pretrained weights for MCPNet backbones and save them in the format
expected by each model's load_pretrained() function.

Usage:
    python MCPNet/scripts/download_pretrained.py \
        --output-dir /net/tscratch/people/plgabedychaj/pretrained

Output files:
    resnet50.pth        - ResNet50 state_dict without fc layer
    inception_v3.pth    - InceptionV3 state_dict (full, load_pretrained handles fc)
    convnext_small.pth  - ConvNeXtWrapper (features_only) state_dict
"""

import argparse
from pathlib import Path

import torch
import torchvision.models as tvm


def download_resnet50(output_dir: Path):
    out = output_dir / 'resnet50.pth'
    if out.exists():
        print(f'  Already exists: {out}')
        return
    print('  Downloading ResNet50 pretrained weights...')
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), out)
    print(f'  Saved: {out}')


def download_inception_v3(output_dir: Path):
    out = output_dir / 'inception_v3.pth'
    if out.exists():
        print(f'  Already exists: {out}')
        return
    print('  Downloading InceptionV3 pretrained weights...')
    model = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), out)
    print(f'  Saved: {out}')


def download_convnext_small(output_dir: Path):
    out = output_dir / 'convnext_small.pth'
    if out.exists():
        print(f'  Already exists: {out}')
        return
    print('  Downloading ConvNeXt-Small pretrained weights via timm...')
    import timm
    from convnext import ConvNeXtWrapper
    backbone = timm.create_model(
        'convnext_small',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    wrapper = ConvNeXtWrapper(backbone)
    torch.save(wrapper.state_dict(), out)
    print(f'  Saved: {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save pretrained weight files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_resnet50(output_dir)
    download_inception_v3(output_dir)
    download_convnext_small(output_dir)

    print('\nDone. Files saved to:', output_dir)


if __name__ == '__main__':
    main()
