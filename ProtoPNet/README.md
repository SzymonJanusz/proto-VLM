# ProtoPNet-CLIP Hybrid Model

An interpretable vision-language model combining ProtoPNet's part-based prototypes with CLIP's text encoder.

## Overview

This project implements a hybrid architecture that:
- **Replaces CLIP's image encoder** with ProtoPNet's prototype-based encoder
- **Keeps CLIP's text encoder** for language understanding
- **Learns interpretable prototypes** grounded in image regions
- **Enables zero-shot capabilities** through vision-language alignment

**Key Features**:
- 🔍 **Interpretable**: Each prototype corresponds to a visual part
- 🚀 **Fast**: Uses pretrained Proto-CLIP initialization
- 💪 **Powerful**: Maintains CLIP's zero-shot capabilities
- 🎯 **Flexible**: Works with ImageNet, Tiny-ImageNet, or custom datasets

## Quick Start

The fastest way to get started:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated setup (downloads Tiny-ImageNet + pretrained weights)
python scripts/quick_start.py

# That's it! This will:
#   - Download Tiny-ImageNet (~250MB)
#   - Download pretrained Proto-CLIP checkpoints
#   - Run a quick training test
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd proto-VLM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Options

You have several options for datasets, from easiest to most comprehensive:

### Option 1: Tiny-ImageNet (Recommended for Testing) ⚡

**Best for**: Quick testing and experimentation

```bash
# Download and organize Tiny-ImageNet (200 classes, ~250MB)
python scripts/download_tiny_imagenet.py --output ./imagenet_tiny

# Verify
python scripts/verify_imagenet.py --imagenet_root ./imagenet_tiny
```

**Specs**: 200 classes, 100K training images, 10K validation images, 64×64 pixels

### Option 2: Full ImageNet (Best Results) 🎯

**Best for**: Production training and best performance

See [IMAGENET_SETUP.md](../scripts/IMAGENET_SETUP.md) for detailed instructions.

**Quick version**:
```bash
# Option A: Download from image-net.org (requires account)
# See IMAGENET_SETUP.md for details

# Option B: Download from Kaggle (easier)
kaggle competitions download -c imagenet-object-localization-challenge
python scripts/organize_kaggle_imagenet.py --input imagenet_kaggle --output imagenet

# Verify
python scripts/verify_imagenet.py --imagenet_root ./imagenet
```

**Specs**: 1000 classes, 1.28M training images, 50K validation images, ~145GB

### Option 3: Your Own Dataset 🔧

Use any image-text dataset by creating a custom data loader following the pattern in `ProtoPNet/data/imagenet_dataset.py`.

## Training

### 3-Stage Training Process

The model is trained in 3 stages:

1. **Stage 1 (Warmup)**: Train backbone + prototypes (text encoder frozen)
2. **Stage 2 (Projection)**: Project prototypes to nearest training patches
3. **Stage 3 (Fine-tuning)**: Fine-tune projection head (backbone + prototypes frozen)

### Basic Training

**With pretrained initialization** (recommended):
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --use_subset \
    --subset_samples 10000
```

**From scratch** (random initialization):
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --use_subset \
    --subset_samples 10000
```

### Advanced Training Options

**Full ImageNet training**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --batch_size 128 \
    --num_workers 8 \
    --warmup_epochs 15 \
    --finetune_epochs 10
```

**Custom hyperparameters**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --num_prototypes 200 \
    --pooling_mode max \
    --warmup_epochs 10 \
    --finetune_epochs 10 \
    --batch_size 64 \
    --lambda_contrastive 1.0 \
    --lambda_clustering 0.01 \
    --lambda_activation 0.01
```

**GPU optimization**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet \
    --batch_size 256 \
    --num_workers 16 \
    --gradient_accumulation 2
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--imagenet_root` | Required | Path to ImageNet directory |
| `--pretrained_protoclip` | None | Path to pretrained Proto-CLIP checkpoints |
| `--num_prototypes` | 200 | Number of prototypes to learn |
| `--pooling_mode` | max | Pooling mode: 'max' or 'attention' |
| `--warmup_epochs` | 10 | Epochs for Stage 1 (warmup) |
| `--finetune_epochs` | 10 | Epochs for Stage 3 (fine-tuning) |
| `--batch_size` | 64 | Training batch size |
| `--use_subset` | False | Use subset for faster experimentation |
| `--subset_samples` | 10000 | Number of samples in subset |

See `python scripts/train.py --help` for all options.

## Pretrained Models

### Downloading Pretrained Proto-CLIP

```bash
# Download pretrained Proto-CLIP checkpoints from GitHub
python scripts/download_pretrained.py

# Verify checkpoints
python scripts/inspect_checkpoints.py \
    --checkpoint_dir ./pretrained_checkpoints/proto_clip_imagenet

# Test loading
python scripts/test_pretrained_loading.py
```

This downloads 3 files:
- `memory_bank_v.pt` - Visual prototypes (16,000 → subsampled to 200)
- `memory_bank_t.pt` - Text prototypes
- `query_adapter.pt` - Adapter network

See [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md) for details.

## Model Architecture

```
IMAGE BRANCH:
Image (224×224)
  → ResNet-50 layer3 (1024×14×14 features)
  → Prototype Layer (compute similarities to 200 prototypes)
  → Weighted Pooling (200 similarity scores)
  → Projection Head (200 → 512-D)
  → L2 Normalize
  → Image Embedding (512-D)

TEXT BRANCH (from CLIP):
Text string
  → CLIP Tokenizer
  → CLIP Text Encoder (frozen initially)
  → Text Embedding (512-D, normalized)

CONTRASTIVE LEARNING:
Image Embeddings × Text Embeddings → Similarity Matrix → InfoNCE Loss
```

## Project Structure

```
proto-VLM/
├── ProtoPNet/
│   ├── models/
│   │   ├── resnet_features.py          # ResNet backbone
│   │   ├── prototype_layer.py          # Prototype similarity computation
│   │   ├── protopnet_encoder.py        # Complete image encoder
│   │   ├── clip_text_encoder.py        # CLIP text encoder wrapper
│   │   └── hybrid_model.py             # ProtoCLIP combined model
│   ├── training/
│   │   ├── losses.py                   # Contrastive + auxiliary losses
│   │   ├── trainer.py                  # Training loop orchestration
│   │   └── optimizer_config.py         # Learning rate schedules
│   ├── data/
│   │   ├── imagenet_dataset.py         # ImageNet with captions
│   │   ├── caption_generator.py        # Caption generation
│   │   └── transforms.py               # Data augmentation
│   └── utils/
│       ├── download_pretrained.py      # Download Proto-CLIP checkpoints
│       ├── checkpoint_converter.py     # Convert checkpoint format
│       ├── projection.py               # Prototype projection (TODO)
│       └── visualization.py            # Visualization (TODO)
├── scripts/
│   ├── train.py                        # Main training script
│   ├── quick_start.py                  # Automated setup
│   ├── download_pretrained.py          # Download checkpoints
│   ├── download_tiny_imagenet.py       # Download Tiny-ImageNet
│   ├── verify_imagenet.py              # Verify dataset
│   ├── organize_imagenet_val.py        # Organize validation set
│   └── test_pretrained_loading.py      # Test checkpoint loading
├── IMAGENET_SETUP.md                   # Detailed ImageNet setup guide
├── PRETRAINED_MODELS.md                # Pretrained model documentation
└── requirements.txt                    # Python dependencies
```

## Evaluation (TODO)

```bash
# Evaluate on image-text retrieval
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny

# Visualize learned prototypes
python scripts/visualize_prototypes.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --output_dir ./visualizations
```

## Tips & Tricks

### For Quick Experimentation

```bash
# Use Tiny-ImageNet with small subset
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --use_subset \
    --subset_samples 1000 \
    --warmup_epochs 3 \
    --finetune_epochs 2 \
    --batch_size 32
```

### For Best Results

```bash
# Full ImageNet with pretrained initialization
python scripts/train.py \
    --imagenet_root ./imagenet \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --batch_size 256 \
    --num_workers 16 \
    --warmup_epochs 15 \
    --finetune_epochs 10
```

### For GPU Memory Issues

```bash
# Reduce batch size and use gradient accumulation
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --num_workers 2
```

### For CPU-Only Systems

```bash
# Use smaller batch size and subset
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --device cpu \
    --batch_size 8 \
    --use_subset \
    --subset_samples 500
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use gradient accumulation
```bash
python scripts/train.py --batch_size 16 --gradient_accumulation 4
```

### Issue: "Split directory not found"
**Solution**: Verify ImageNet structure
```bash
python scripts/verify_imagenet.py --imagenet_root ./imagenet
```

### Issue: Training is too slow
**Solution**: Use Tiny-ImageNet or subset
```bash
python scripts/download_tiny_imagenet.py --output ./imagenet_tiny
```

### Issue: "No module named 'scipy'"
**Solution**: Install scipy
```bash
pip install scipy>=1.10.0
```

## Citation

If you use this code, please cite:

**ProtoPNet**:
```bibtex
@inproceedings{chen2019protopnet,
  title={This Looks Like That: Deep Learning for Interpretable Image Recognition},
  author={Chen, Chaofan and Li, Oscar and Tao, Daniel and Barnett, Alina and Rudin, Cynthia and Su, Jonathan K},
  booktitle={NeurIPS},
  year={2019}
}
```

**Proto-CLIP** (pretrained initialization):
```bibtex
@article{protoclip2024,
  title={Vision-Language Prototypical Network for Interpretable Text-to-Image Generation},
  author={[Authors]},
  journal={IROS},
  year={2024}
}
```

**CLIP**:
```bibtex
@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={ICML},
  year={2021}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) for the interpretable prototype architecture
- [Proto-CLIP](https://github.com/IRVLUTD/Proto-CLIP) for pretrained checkpoints
- [OpenAI CLIP](https://github.com/openai/CLIP) for the vision-language framework
- [Hugging Face Transformers](https://huggingface.co/transformers) for CLIP implementation

## Contact

For questions or issues, please open an issue on GitHub.
