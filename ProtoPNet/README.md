# ProtoPNet-CLIP Hybrid Model

An interpretable vision-language model combining ProtoPNet's part-based prototypes with CLIP's text encoder.

## Overview

This project implements a hybrid architecture that:
- **Replaces CLIP's image encoder** with ProtoPNet's prototype-based encoder
- **Keeps CLIP's text encoder** for language understanding
- **Learns interpretable prototypes** grounded in real image patches
- **Enables zero-shot capabilities** through vision-language alignment

**Key Features**:
- 🔍 **Interpretable**: Each prototype corresponds to an actual training image patch
- 🚀 **Fast**: Uses pretrained Proto-CLIP initialization
- 💪 **Powerful**: Maintains CLIP's zero-shot capabilities
- 🎯 **Flexible**: Works with ImageNet, Tiny-ImageNet, or custom datasets
- 📊 **Visualizable**: HTML visualizations show which image patches prototypes represent

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ProtoCLIP Model                         │
├──────────────────────────────┬──────────────────────────────┤
│      IMAGE ENCODER           │      TEXT ENCODER            │
│    (ProtoPNet-based)         │      (CLIP-based)            │
└──────────────────────────────┴──────────────────────────────┘
              ↓                              ↓
        Image Embedding                 Text Embedding
           (512-D, L2-normalized)        (512-D, L2-normalized)
                    ↘                  ↙
                   Cosine Similarity Matrix
                           ↓
                   Contrastive Loss (InfoNCE)
```

### Detailed Architecture

#### Image Encoder Pipeline

```
Input Image (B, 3, 224, 224)
    ↓
┌───────────────────────────────────────────────────────────┐
│ 1. CNN Backbone (ResNet-50)                               │
│    - Pretrained on ImageNet                               │
│    - Extract features from layer3                         │
│    - Output: (B, 1024, 14, 14)                           │
│    - 14×14 = 196 spatial patches per image               │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│ 2. Prototype Layer                                        │
│    - M learnable prototypes (default: 200)                │
│    - Each prototype: 1024-dimensional vector              │
│    - Compute L2 distance: ||patch - prototype||²         │
│    - Convert to similarity: -distance²                    │
│    - Output: (B, M, 14, 14) similarity maps              │
│                                                            │
│    Formula: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩         │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│ 3. Weighted Pooling                                       │
│    - Mode 1: Max Pooling (default)                       │
│      Takes max similarity over 14×14 spatial grid        │
│    - Mode 2: Attention Pooling                           │
│      Learnable attention-weighted average                 │
│    - Output: (B, M) - one score per prototype            │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│ 4. Projection Head                                        │
│    - 2-layer MLP with BatchNorm                          │
│    - Layer 1: (M → hidden_dim) + ReLU + Dropout         │
│    - Layer 2: (hidden_dim → 512) + BatchNorm            │
│    - Configurable hidden_dim (default: 512)              │
│    - Configurable dropout rate (default: 0.5)            │
│    - Reduced capacity to prevent overfitting             │
│    - Output: (B, 512) embeddings                         │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│ 5. L2 Normalization                                       │
│    - Normalize to unit sphere                            │
│    - Enables cosine similarity computation                │
│    - Final Output: (B, 512) normalized embeddings        │
└───────────────────────────────────────────────────────────┘
```

#### Text Encoder (CLIP)

```
Input Text (e.g., "a photo of a golden retriever")
    ↓
CLIP Tokenizer
    ↓
CLIP Text Transformer (openai/clip-vit-base-patch32)
    ↓
L2 Normalization
    ↓
Text Embedding (512-D, normalized)
```

### Key Architectural Innovations

1. **Prototype-Based Reasoning**: Instead of black-box features, the model learns M prototypes that represent distinctive visual patterns
2. **Spatial Similarity Maps**: For each prototype, we get a 14×14 map showing where similar patterns appear in the image
3. **Interpretability Through Projection**: After Stage 2, each prototype corresponds to an actual training image patch
4. **CLIP Alignment**: Contrastive learning aligns visual prototypes with text descriptions
5. **Learnable Temperature**: Uses log-space initialization (`log(1/0.07)`) for better gradient flow
6. **Reduced Projection Head**: Configurable hidden dimension (default: 512) to prevent overfitting
7. **Standard ImageNet Indexing**: Uses synset-to-index mapping for consistent class ordering across datasets

## Training Process

### 3-Stage Training Pipeline

The model is trained in 3 distinct stages, each optimizing different components:

```
Stage 1: WARMUP          Stage 2: PROJECTION     Stage 3: FINE-TUNING
(10 epochs)              (single pass)           (10 epochs)
    ↓                         ↓                        ↓
Train backbone +         Project prototypes      Fine-tune projection
prototypes               to training patches     head only
```

#### Stage 1: Warmup Training

**Goal**: Learn good prototype representations and feature extractors

**What's trained**:
- ✅ ResNet backbone (low learning rate)
- ✅ Prototype vectors (higher learning rate)
- ✅ Projection head
- ❌ Text encoder (frozen)

**Losses**:
- **Contrastive Loss** (λ=1.0): Align image-text pairs
- **Clustering Loss** (λ=0.1): Encourage prototypes to be diverse
- **Activation Loss** (λ=0.1): Encourage sparse prototype activation

**Note**: The clustering and activation loss weights were increased from 0.01 to 0.1 in recent updates for better prototype diversity.

**Learning Rates**:
- Backbone: 5e-5 (pretrained init) or 1e-4 (random init)
- Prototypes: 5e-4 (pretrained init) or 1e-3 (random init)
- Scheduler: Cosine annealing

**Output**: `warmup_best.pt` checkpoint with learned prototypes

#### Stage 2: Prototype Projection

**Goal**: Ground prototypes in actual training image patches for interpretability

**Algorithm**:
```python
For each prototype p_m (m = 1 to M):
    best_distance[m] = ∞
    best_patch[m] = None

For each batch in training set:
    Extract features: (B, 1024, 14, 14)
    Reshape to patches: (B×196, 1024)

    For each prototype p_m:
        Compute L2 distances to all patches
        Find minimum distance patch
        If distance < best_distance[m]:
            Update best_distance[m]
            Update best_patch[m]
            Store metadata (image path, coordinates)

Replace prototypes with best_patches
```

**Key Features**:
- Memory-efficient streaming algorithm
- Processes batches incrementally
- Uses validation transforms (no augmentation) for stable features
- Optional batch limiting for faster iteration

**Outputs**:
- Updated prototype vectors (grounded in training patches)
- `projection_info.pkl`: Metadata for each prototype
- `projection_summary.txt`: Human-readable statistics
- `prototype_visualization.html`: Interactive HTML report (optional)
- `prototype_grid.jpg`: Grid of prototype images (optional)

**Visualization Content**:
Each prototype visualization shows:
- Source training image with red bounding box
- Class name (e.g., "golden retriever")
- Patch coordinates (h, w) in 14×14 feature map
- L2 distance from learned prototype
- Class ID

#### Stage 3: Fine-Tuning

**Goal**: Adapt projection head to work with projected (frozen) prototypes

**What's trained**:
- ✅ Projection head only
- ❌ Backbone (frozen)
- ❌ Prototypes (frozen - grounded in training patches)
- ❌ Text encoder (frozen, unless `--unfreeze_text_encoder`)

**Regularization**:
- **Dropout**: Configurable rate (default: 0.5)
- **Weight Decay**: L2 regularization (default: 0.01)
- **Early Stopping**: Patience-based (default: 5 epochs)

**Learning Rate Schedule**:
- Warmup: Linear increase for first 2 epochs (1% → 100% of target LR)
- Main: Cosine annealing decay
- Target LR: 1e-5

**Losses**:
- **Contrastive Loss** (primary, with optional label smoothing λ=0.1)
- **Clustering Loss** (λ=0.005, optional): Maintain prototype diversity
- **Activation Loss** (λ=0.005, optional): Maintain sparse activation

**Label Smoothing**: Prevents overconfident predictions by using soft targets (0.9 for positive pairs, 0.1/(B-1) for negatives). This improves generalization and reduces overfitting.

**Early Stopping**:
- Monitors validation loss
- Stops if no improvement for N epochs (default: 5)
- Saves best checkpoint automatically

**Output**: `finetune_best.pt` - final trained model

### Training Data Flow

```
ImageNet Dataset
    ↓
Caption Generator (creates text descriptions)
    ↓
[Image, Caption] pairs
    ↓
Data Augmentation (Stage 1 & 3 only)
    ↓
Batch Loading (batch_size=64)
    ↓
GPU Transfer
    ↓
Model Forward Pass
    ↓
Loss Computation
    ↓
Backpropagation (Stage 1 & 3 only)
    ↓
Optimizer Step
```

**Note**: Stage 2 uses validation transforms (no augmentation) to ensure stable feature extraction for projection.

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

**What's New**: The codebase now includes label smoothing, stronger augmentation options, TensorBoard logging, and improved class mapping for better training stability and evaluation accuracy.

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

### Basic Training (All 3 Stages)

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

### Stage-Specific Training

**Run only Stage 2 (Projection) with visualization**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --use_subset --subset_samples 5000 \
    --skip_warmup \
    --skip_finetune \
    --resume_from ./checkpoints/warmup_best.pt \
    --projection_visualize \
    --projection_num_viz 50
```

**Fast projection (limit batches)**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --skip_warmup --skip_finetune \
    --resume_from ./checkpoints/warmup_best.pt \
    --projection_max_batches 100 \
    --projection_visualize
```

**Skip specific stages**:
```bash
# Skip warmup (resume from checkpoint)
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --skip_warmup \
    --resume_from ./checkpoints/warmup_best.pt

# Skip projection
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --skip_projection
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

**Fine-tuning with early stopping and regularization**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --finetune_epochs 20 \
    --finetune_patience 5 \
    --finetune_dropout 0.5 \
    --finetune_weight_decay 0.01 \
    --finetune_warmup_epochs 2
```

**GPU optimization**:
```bash
python scripts/train.py \
    --imagenet_root ./imagenet \
    --batch_size 256 \
    --num_workers 16 \
    --gradient_accumulation 2
```

**With CutMix augmentation** (improved regularization):
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --use_cutmix \
    --cutmix_alpha 1.0 \
    --mixup_prob 0.5
```

**With all regularization features** (best for preventing overfitting):
```bash
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --augmentation_strength strong_v2 \
    --use_cutmix \
    --finetune_label_smoothing 0.1 \
    --finetune_dropout 0.5 \
    --projection_hidden_dim 512 \
    --finetune_lambda_clustering 0.005 \
    --finetune_lambda_activation 0.005
```

### Training Arguments

#### Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--imagenet_root` | Required | Path to ImageNet directory |
| `--pretrained_protoclip` | None | Path to pretrained Proto-CLIP checkpoints |
| `--num_prototypes` | 200 | Number of prototypes to learn |
| `--pooling_mode` | max | Pooling mode: 'max' or 'attention' |
| `--batch_size` | 64 | Training batch size |
| `--device` | cuda | Device: 'cuda' or 'cpu' |
| `--checkpoint_dir` | ./checkpoints | Directory to save checkpoints |

#### Stage 1 (Warmup) Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--warmup_epochs` | 10 | Number of warmup epochs |
| `--warmup_lr_backbone` | Auto | LR for backbone (1e-4 random, 5e-5 pretrained) |
| `--warmup_lr_prototypes` | Auto | LR for prototypes (1e-3 random, 5e-4 pretrained) |
| `--warmup_weight_decay` | 0.01 | Weight decay for warmup stage |
| `--warmup_patience` | 3 | Early stopping patience (0 to disable) |
| `--warmup_min_delta` | 0.0 | Minimum validation loss improvement |
| `--lambda_contrastive` | 1.0 | Weight for contrastive loss |
| `--lambda_clustering` | 0.1 | Weight for clustering loss (increased from 0.01) |
| `--lambda_activation` | 0.1 | Weight for activation loss (increased from 0.01) |

#### Stage 2 (Projection) Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--projection_max_batches` | None | Max batches to process (None = all) |
| `--projection_visualize` | False | Generate HTML visualizations |
| `--projection_num_viz` | 20 | Number of prototypes to visualize |
| `--skip_projection` | False | Skip Stage 2 entirely |

#### Stage 3 (Fine-tuning) Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--finetune_epochs` | 10 | Number of fine-tuning epochs |
| `--finetune_lr` | 1e-5 | Learning rate |
| `--finetune_weight_decay` | 0.01 | L2 regularization strength |
| `--finetune_dropout` | 0.5 | Dropout rate in projection head |
| `--projection_hidden_dim` | 512 | Hidden dimension for projection head (reduces overfitting) |
| `--finetune_warmup_epochs` | 2 | LR warmup epochs (0 to disable) |
| `--finetune_patience` | 5 | Early stopping patience |
| `--finetune_min_delta` | 0.0 | Min improvement to reset patience |
| `--finetune_lambda_clustering` | 0.005 | Clustering loss weight (0 to disable) |
| `--finetune_lambda_activation` | 0.005 | Activation loss weight (0 to disable) |
| `--finetune_label_smoothing` | 0.1 | Label smoothing factor (0 to disable) |
| `--unfreeze_text_encoder` | False | Unfreeze CLIP text encoder (may hurt zero-shot) |

#### Data Augmentation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--augmentation_strength` | medium | Augmentation level: 'light', 'medium', 'strong', 'strong_v2' |
| `--use_cutmix` | False | Enable CutMix augmentation |
| `--cutmix_alpha` | 1.0 | CutMix alpha parameter (mixing strength) |
| `--mixup_prob` | 0.5 | Probability of applying CutMix |

**Augmentation Strengths**:
- `light`: Minimal augmentation (resize + random crop + flip)
- `medium`: Standard augmentation (+ color jitter with brightness/contrast/saturation=0.2)
- `strong`: Aggressive augmentation (color jitter with 0.3 and AutoAugment)
- `strong_v2`: Very aggressive (color jitter 0.3 + random grayscale 10%)

**CutMix** creates virtual training examples by cutting and pasting patches between images. Unlike Mixup which blends entire images, CutMix preserves local structure, making it more suitable for prototype-based models.

#### Other Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_subset` | False | Use subset for experimentation |
| `--subset_samples` | 10000 | Number of samples in subset |
| `--num_workers` | 4 | Data loading workers |
| `--gradient_accumulation` | 1 | Gradient accumulation steps |
| `--skip_warmup` | False | Skip Stage 1 |
| `--skip_finetune` | False | Skip Stage 3 |
| `--resume_from` | None | Resume from checkpoint |
| `--seed` | 42 | Random seed |

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

## Output Files

After training, you'll find these files in `./checkpoints/`:

### Stage 1 Output
- `warmup_best.pt` - Best warmup checkpoint
- `warmup_latest.pt` - Latest warmup checkpoint

### Stage 2 Output
- `projection_info.pkl` - Projection metadata (pickle format)
- `projection_summary.txt` - Human-readable statistics
- `prototype_visualization.html` - Interactive HTML report
- `prototype_grid.jpg` - Grid of prototype images
- `prototype_images/` - Individual prototype images with bounding boxes

### Stage 3 Output
- `finetune_best.pt` - Best fine-tuned model
- `finetune_latest.pt` - Latest fine-tuned checkpoint

## TensorBoard Logging

The training script automatically logs metrics to TensorBoard for visualization and monitoring.

### Viewing Logs

```bash
# Start TensorBoard server
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

### Logged Metrics

**Training Metrics** (per batch and per epoch):
- Total loss
- Contrastive loss
- Clustering loss (if enabled)
- Activation loss (if enabled)

**Validation Metrics** (per epoch):
- Validation loss
- Validation contrastive loss

**Per-Stage Tracking**:
- Metrics are separated by stage (warmup/, finetune/)
- Allows easy comparison between training stages

See [TENSORBOARD_GUIDE.md](../TENSORBOARD_GUIDE.md) for detailed visualization guide.

## Project Structure

```
ProtoPNet/
├── configs/
│   └── base_config.yaml            # Base configuration template
├── models/
│   ├── resnet_features.py          # ResNet backbone feature extractor
│   ├── prototype_layer.py          # Prototype similarity computation
│   ├── protopnet_encoder.py        # Complete ProtoPNet image encoder
│   ├── clip_text_encoder.py        # CLIP text encoder wrapper
│   └── hybrid_model.py             # ProtoCLIP combined model
├── training/
│   ├── losses.py                   # Contrastive + auxiliary losses
│   ├── trainer.py                  # Training loop orchestration
│   ├── label_smoothing.py          # Label smoothing for contrastive loss
│   └── mixup.py                    # CutMix augmentation
├── data/
│   ├── imagenet_dataset.py         # ImageNet dataset with captions
│   ├── caption_generator.py        # Caption generation for images
│   └── transforms.py               # Data augmentation pipelines
└── utils/
    ├── early_stopping.py           # Early stopping handler
    ├── projection.py               # Prototype projection algorithm
    ├── visualization.py            # HTML visualization generator
    ├── download_pretrained.py      # Download Proto-CLIP checkpoints
    └── checkpoint_converter.py     # Convert checkpoint formats
```

## Recent Improvements

The codebase has undergone several improvements to enhance training stability and prevent overfitting:

### Architecture Changes
- **Reduced Projection Head**: Configurable hidden dimension (default: 512, previously fixed at 1024) to reduce model capacity and prevent overfitting
- **Temperature Initialization**: Changed from linear (`1/0.07`) to log-space (`log(1/0.07)`) for better gradient flow during contrastive learning
- **Standard ImageNet Indexing**: Uses synset-to-index mapping for consistent class ordering across datasets

### Training Enhancements
- **Label Smoothing**: Optional label smoothing (default: 0.1) for contrastive loss to prevent overconfident predictions
- **Stronger Regularization**: Increased auxiliary loss weights (clustering and activation from 0.01 to 0.1 in warmup, 0.005 in fine-tuning)
- **Advanced Augmentation**: New `strong_v2` augmentation option with random grayscale for better regularization
- **CutMix Support**: Preserves local structure better than Mixup for prototype learning
- **TensorBoard Integration**: Comprehensive logging of training metrics for monitoring and debugging

### Evaluation Features
- **Zero-shot Classification**: Full implementation with top-1 and top-5 accuracy metrics
- **Prototype Retrieval**: Analyze which text descriptions best match learned prototypes
- **Visualization Tools**: Generate grid visualizations of prototype matches with bounding boxes

These improvements were introduced in commits `18c08b7`, `3e149e2`, and related updates.

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

### For Debugging Prototypes

```bash
# Run only projection with visualization on small data
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --use_subset --subset_samples 100 \
    --skip_warmup --skip_finetune \
    --resume_from ./checkpoints/warmup_best.pt \
    --projection_visualize \
    --projection_num_viz 20
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use gradient accumulation
```bash
python scripts/train.py --batch_size 16 --gradient_accumulation 4
```

### Issue: "RuntimeError: Input type and weight type should be the same"
**Solution**: This is fixed in the latest version. Model is automatically moved to device after checkpoint loading.

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

### Issue: Projection visualization missing class names
**Solution**: Ensure `--class_mapping_file` points to valid JSON
```bash
# Download class mapping if needed
python scripts/download_imagenet_classes.py
```

### Issue: Class indices don't match between datasets
**Solution**: The codebase now uses standard ImageNet synset-to-index mapping (via `data/imagenet_class_index.json`) for consistent class ordering. This ensures:
- Consistent class indices across train/val splits
- Compatibility with pretrained models
- Proper evaluation metrics

If using Tiny-ImageNet or custom datasets, the loader will automatically skip synsets not in ImageNet-1K and report statistics.

## Evaluation

The evaluation script provides comprehensive model assessment including zero-shot classification and prototype-text retrieval.

### Zero-Shot Classification

Evaluate the model's ability to classify images using text prompts:

```bash
# Basic zero-shot evaluation
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode zero-shot \
    --batch_size 128

# With ensemble prompts (more accurate)
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode zero-shot \
    --use_ensemble

# Quick test on subset
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode zero-shot \
    --num_samples 1000
```

**Metrics**:
- Top-1 Accuracy: Percentage of correct first predictions
- Top-5 Accuracy: Percentage where correct class is in top 5 predictions

### Prototype Image-to-Text Retrieval

Analyze what text descriptions best match learned prototypes:

```bash
# Evaluate retrieval performance
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode retrieval \
    --num_retrieval_samples 1000 \
    --top_k_retrieval 5

# With visualizations
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode retrieval \
    --visualize \
    --num_prototypes_to_visualize 20
```

This shows:
- Top-k text captions that best match each prototype
- Cosine similarity scores
- Optional grid visualizations showing prototype matches

### Combined Evaluation

Run both zero-shot and retrieval evaluations:

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --imagenet_root ./imagenet_tiny \
    --mode both \
    --visualize
```

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--imagenet_root` | Required | Path to ImageNet directory |
| `--mode` | both | 'zero-shot', 'retrieval', or 'both' |
| `--batch_size` | 128 | Batch size for evaluation |
| `--use_ensemble` | False | Use ensemble prompts (slower but more accurate) |
| `--num_samples` | None | Limit samples for quick testing |
| `--num_retrieval_samples` | 1000 | Samples for retrieval evaluation |
| `--top_k_retrieval` | 5 | Number of top text matches to show |
| `--visualize` | False | Generate visualization grids |
| `--num_prototypes_to_visualize` | 20 | Number of prototypes to visualize |
| `--viz_output_dir` | None | Custom visualization output directory |

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
