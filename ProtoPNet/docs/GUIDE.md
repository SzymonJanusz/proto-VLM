# ProtoPNet Comprehensive Usage Guide

This guide covers all extended capabilities of the ProtoPNet/ProtoCLIP system beyond the core training workflow.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prototype-to-Text Generation](#prototype-to-text-generation)
3. [CLIP-Based Prototype Interpretation](#clip-based-prototype-interpretation)
4. [ClipCap Integration](#clipcap-integration)
5. [TensorBoard Monitoring](#tensorboard-monitoring)
6. [Best Practices and Troubleshooting](#best-practices-and-troubleshooting)

---

## Introduction

The ProtoPNet system provides three main capabilities:

### 1. ProtoCLIP Training (Core)
Train a hybrid model combining ProtoPNet's interpretable prototype-based image encoder with CLIP's text encoder for vision-language alignment. See main [README.md](../README.md) for details.

### 2. Prototype-to-Text Generation
Generate natural language descriptions of images based on prototype activations using learned projectors and ClipCap decoder. **Requires training**.

### 3. CLIP-Based Interpretation
Interpret what visual concepts prototypes represent using pre-trained CLIP. **No additional training required**.

**When to use each:**
- **ProtoCLIP Training**: When you want to train an interpretable vision-language model from scratch
- **Prototype-to-Text**: When you want to generate captions for images based on which prototypes activate
- **CLIP Interpretation**: When you want to understand what visual concepts your trained prototypes represent

---

## Prototype-to-Text Generation

Generate natural language descriptions of images based on prototype activations.

### Overview

The prototype projector system:
1. Extracts spatial prototype similarities (B, 200, 14, 14) from trained ProtoPNet
2. Applies sparse top-k selection (e.g., k=5) to focus on most active prototypes
3. Projects to CLIP embedding space (B, 512) using learned projector
4. Generates captions via ClipCap decoder

**Architecture Flow:**
```
Image → ProtoPNet → Spatial Similarities (B, 200, 14, 14)
                   ↓ [Top-k Selection]
                   Sparse Similarities (B, 200, 14, 14)
                   ↓ [Projector: Spatial CNN or Hierarchical Pooling]
                   CLIP Embeddings (B, 512)
                   ↓ [ClipCap Decoder]
                   Natural Language Description
```

### Architecture Options

#### Hierarchical Pooling Projector (Recommended)

**Best for:** Sparse activations (k=1 to k=20)

**How it works:**
- Computes max, mean, std statistics over spatial dimensions
- Concatenates features (600-dim) and projects to CLIP space (512-dim)
- Works well even with very sparse data

**Parameters:**
- `--projector_type hierarchical`
- `--hidden_dim 1024` (default, size of MLP hidden layer)
- `--dropout 0.3` (default)

#### Spatial CNN Projector

**Best for:** Less sparse activations (k=10+)

**How it works:**
- Convolutional layers that respect spatial structure
- Learns spatial relationships between prototype activations
- Requires more non-zero channels to work effectively

**Parameters:**
- `--projector_type spatial_cnn`
- `--hidden_channels 512` (default, CNN hidden channels)
- `--dropout 0.3` (default)

### Prerequisites

**Install dependencies:**
```bash
pip install torch torchvision transformers matplotlib gdown
pip install git+https://github.com/openai/CLIP.git
```

**Download ClipCap weights:**
```bash
python scripts/download_clipcap_weights.py
```

**Required files:**
- ProtoCLIP checkpoint: `checkpoints/protoclip_best.pt`
- ClipCap weights: `pretrained_checkpoints/clipcap/clipcap_coco.pt`
- Dataset: ImageNet-Tiny at `imagenet_tiny/`

### Training a Prototype Projector

#### Basic Training

```bash
python scripts/train_prototype_projector.py \
    --backbone_checkpoint checkpoints/protoclip_best.pt \
    --data_root imagenet_tiny \
    --output_dir checkpoints/prototype_projector \
    --projector_type hierarchical \
    --top_k 5 \
    --epochs 100 \
    --batch_size 128 \
    --evaluate_captions \
    --clipcap_model pretrained_checkpoints/clipcap/clipcap_coco.pt
```

**Output files:**
- `best.pt` - Best validation checkpoint
- `checkpoint_epoch_25.pt`, `checkpoint_epoch_50.pt`, etc. - Periodic checkpoints
- `training_history.json` - Full metrics history
- `training_curves.png` - Loss, accuracy, embedding similarity curves
- `captions_epoch_25.json`, etc. - Generated captions every 25 epochs

**Training time:** ~13-15 hours for 100 epochs on RTX 3090/V100

#### Different Sparsity Levels

**k=1 (most sparse):** Only strongest prototype
```bash
python scripts/train_prototype_projector.py \
    --projector_type hierarchical \
    --top_k 1 \
    ...
```

**k=5 (recommended):** Top 5 prototypes - good balance
```bash
--top_k 5
```

**k=10 (moderate):** More context, better for Spatial CNN
```bash
--top_k 10
```

**k=20 (less sparse):** Approaching dense, maximum context
```bash
--top_k 20
```

### Comparing Configurations

Test multiple configurations systematically:

```bash
python scripts/compare_prototype_projectors.py \
    --backbone_checkpoint checkpoints/protoclip_best.pt \
    --data_root imagenet_tiny \
    --output_dir experiments/comparison \
    --projector_types spatial_cnn hierarchical \
    --k_values 1 5 10 20 \
    --epochs 25
```

This tests 8 configurations (2 architectures × 4 k values) with 25 epochs each.

**Output:**
- Individual directories for each config (e.g., `hierarchical_k5/`)
- `comparison_report.json` - Summary with best configurations by accuracy, similarity, loss

### Performance Expectations

**After 25 epochs:**
- I2T Accuracy: 10-15%
- Embedding Similarity: 0.3-0.5
- Coherent captions starting to form

**After 100 epochs:**
- I2T Accuracy: 15-25%
- Embedding Similarity: 0.5-0.7
- High-quality, diverse captions

### Troubleshooting

**Low accuracy (<5%) after 25 epochs:**
- Check that text encoder matches (Hugging Face CLIP for ProtoCLIP)
- Verify checkpoint is correct ProtoCLIP model
- Try different k value (k=5 or k=10)

**NaN losses:**
- Reduce learning rate: `--lr 5e-5`
- Add gradient clipping (modify training script)
- Use smaller hidden dimension for hierarchical: `--hidden_dim 512`

**Slow training:**
- Skip caption evaluation: remove `--evaluate_captions`
- Reduce batch size if OOM: `--batch_size 64`
- Use ImageNet-Tiny instead of full ImageNet

---

## CLIP-Based Prototype Interpretation

Interpret what visual concepts prototypes represent using pre-trained CLIP - **no additional training required**.

### Overview

**How it works:**
1. Extract prototype patch from original training image
2. Create synthetic image (patch on noise background)
3. Encode with CLIP image encoder
4. Compare to text embeddings from pre-generated vocabulary
5. Return top-k nearest text descriptions

**Key advantage:** Uses pre-trained CLIP only - no model training needed!

### Prerequisites

```bash
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib pillow
```

### Usage

#### Batch Interpretation (All Prototypes)

Interpret all 200 prototypes:

```bash
python scripts/interpret_prototypes.py \
    --checkpoint checkpoints/protoclip_best.pt \
    --data_root imagenet_tiny \
    --output_dir results/interpretation \
    --patch_sizes 32 64 128 \
    --top_k 5
```

**Parameters:**
- `--patch_sizes 32 64 128` - Test multiple context windows
- `--top_k 5` - Return top 5 text descriptions per prototype
- `--batch_size 32` - Batch size for CLIP encoding

**Output structure:**
```
results/interpretation/
├── prototype_images/           # Cropped prototype patches
│   ├── prototype_000_patch32.png
│   ├── prototype_000_patch64.png
│   └── ...
├── prototype_text_interpretations/  # Text files with descriptions
│   ├── prototype_000_patch32.txt
│   ├── prototype_000_patch64.txt
│   └── ...
└── interpretation_summary.json      # Complete results JSON
```

#### Class-Organized Interpretation

Interpret prototypes for a specific class:

```bash
python scripts/interpret_by_class.py \
    --checkpoint checkpoints/protoclip_best.pt \
    --data_root imagenet_tiny \
    --output_dir results/class_interpretation \
    --class_id 45 \
    --patch_sizes 64 \
    --top_k 10
```

**Output:** Same structure as batch interpretation, but filtered to one class

### Interpretation Results

**Text output format:**
```
Top 5 interpretations for Prototype 0 (patch size: 64):

1. brown fur texture (similarity: 0.87)
2. animal fur (similarity: 0.85)
3. brown hair (similarity: 0.82)
4. soft fur (similarity: 0.80)
5. mammal fur pattern (similarity: 0.78)
```

### Patch Size Selection

**Patch size 32:** Fine-grained details (eyes, specific textures)
**Patch size 64:** Balanced view (typical parts like heads, legs)
**Patch size 128:** Broader context (full objects, backgrounds)

**Recommendation:** Use all three sizes to get multi-scale understanding

### Advanced Options

**Custom vocabulary:**
Create your own vocabulary file with domain-specific terms:
```python
# In scripts/interpret_prototypes.py, modify vocabulary generation
from ProtoPNet.utils.clip_vocabulary import generate_vocabulary
custom_vocab = ["dog breed terrier", "cat whiskers", "bird beak", ...]
```

**VLM enrichment (optional):**
Use BLIP or GIT for richer descriptions:
```python
from ProtoPNet.utils.vlm_interpretation import interpret_with_blip
# Generates full sentences instead of short phrases
```

---

## ClipCap Integration

Use ClipCap (CLIP prefix + GPT-2) for generating natural language captions from CLIP embeddings.

### Setup

**Download ClipCap weights:**
```bash
python scripts/download_clipcap_weights.py
```

This downloads pretrained weights to `pretrained_checkpoints/clipcap/clipcap_coco.pt` (~600 MB).

**Or download manually:**
```bash
mkdir -p pretrained_checkpoints/clipcap
cd pretrained_checkpoints/clipcap

# Download from Google Drive
gdown 1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX  # clipcap_coco.pt
```

### Using ClipCap

ClipCap is integrated into the prototype projector training pipeline. It generates captions every 25 epochs during validation.

**Enable caption generation:**
```bash
python scripts/train_prototype_projector.py \
    --evaluate_captions \
    --clipcap_model pretrained_checkpoints/clipcap/clipcap_coco.pt \
    --num_caption_samples 10 \
    --captions_per_image 5 \
    ...
```

**Caption output format:**
```json
{
  "epoch": 25,
  "num_samples": 10,
  "samples": [
    {
      "image_path": "imagenet_tiny/val/n01440764/val_123.JPEG",
      "class_name": "tench",
      "prototype_captions": [
        "a fish swimming in clear water",
        "a tench in a pond",
        ...
      ],
      "clip_vit_captions": [
        "a freshwater fish in natural habitat",
        ...
      ]
    }
  ]
}
```

### Known Issues and Workarounds

**Issue 1: Repetitive captions**
- **Cause:** ClipCap trained on COCO captions, projector trained on classification
- **Symptom:** Captions like "a photo of a photo of a photo..."
- **Workaround:** Use CLIP ViT baseline captions as reference, or increase training epochs

**Issue 2: Generic descriptions**
- **Cause:** CLIP embeddings from classification vs caption objectives differ
- **Solution:** This is expected - focus on whether captions are coherent, not perfect

### ClipCap Decoder Usage (Standalone)

```python
from ProtoPNet.utils.text_decoders import ClipCapDecoder

# Load decoder
decoder = ClipCapDecoder(
    model_path='pretrained_checkpoints/clipcap/clipcap_coco.pt',
    device='cuda'
)

# Generate caption from CLIP embedding
clip_embedding = ...  # (512,) tensor, L2-normalized
captions = decoder.decode(clip_embedding, num_captions=5)

print(captions)
# ['a fish in the water', 'a fish swimming', ...]
```

---

## TensorBoard Monitoring

Monitor ProtoCLIP training in real-time using TensorBoard.

### Setup

TensorBoard is automatically enabled during ProtoCLIP training. Logs are saved to `checkpoints/{experiment_name}/tensorboard/`.

### Launching TensorBoard

```bash
# From project root
tensorboard --logdir checkpoints/protoclip_experiment/tensorboard

# Access at http://localhost:6006
```

**Multi-experiment comparison:**
```bash
tensorboard --logdir checkpoints/
```

This compares all experiments in the checkpoints directory.

### Key Metrics to Monitor

#### Training Metrics
- **train/loss** - Overall training loss (should decrease)
- **train/cross_entropy_loss** - Classification loss component
- **train/cluster_loss** - Prototype clustering loss
- **train/separation_loss** - Inter-class separation loss
- **train/accuracy** - Training accuracy

#### Validation Metrics
- **val/loss** - Validation loss (monitor for overfitting)
- **val/accuracy** - Validation accuracy
- **val/top5_accuracy** - Top-5 validation accuracy

#### Learning Dynamics
- **learning_rate** - Current learning rate (cosine annealing)
- **epoch** - Current epoch number

### What to Look For

**Healthy training:**
- Training loss steadily decreasing
- Validation accuracy improving
- Gap between train and val accuracy <10%

**Overfitting:**
- Training loss still decreasing
- Validation loss increasing or plateauing
- Large gap between train and val accuracy (>15%)
- **Solution:** Add more data augmentation, reduce model capacity, or early stop

**Underfitting:**
- Both train and val loss high
- Both train and val accuracy low
- **Solution:** Train longer, increase model capacity, or check data quality

### TensorBoard Tips

**Smoothing:** Adjust smoothing slider to see trends more clearly (default 0.6)

**Comparing runs:** Select multiple runs in the left panel to overlay curves

**Downloading data:** Click on the three-dot menu on any plot to download as CSV or JSON

---

## Best Practices and Troubleshooting

### Prototype Projector Best Practices

#### Choosing Top-k

| k value | Use case | Expected accuracy |
|---------|----------|-------------------|
| k=1 | When only strongest prototype matters | 5-10% |
| k=5 | **Recommended starting point** | 10-15% |
| k=10 | Balance of focus and context | 15-20% |
| k=20 | Maximum context, approaching dense | 20-25% |

**Rule of thumb:** Start with k=5, increase if you need more context

#### Choosing Architecture

**Use Hierarchical Pooling when:**
- k ≤ 10 (sparse activation)
- You want faster training
- You want more robust results

**Use Spatial CNN when:**
- k ≥ 10 (less sparse)
- You want to preserve spatial relationships
- You have more compute budget

#### Training Duration

**25 epochs:** Quick experiment, rough performance estimate
**50 epochs:** Good starting point for real results
**100 epochs:** Full training, best performance

### Common Issues

#### Issue: "CUDA out of memory"

**Solutions:**
```bash
# Reduce batch size
--batch_size 64  # or 32

# Skip caption evaluation
# (remove --evaluate_captions flag)

# Use CPU (much slower)
--device cpu
```

#### Issue: "Checkpoint not found"

**Check:**
```bash
ls checkpoints/protoclip_best.pt  # Verify file exists
```

**Solution:** Train ProtoCLIP first (see main README.md)

#### Issue: "ClipCap weights not found"

**Solution:**
```bash
python scripts/download_clipcap_weights.py
```

#### Issue: Low embedding similarity (<0.1)

**Causes:**
- Text encoder mismatch (Hugging Face vs OpenAI CLIP)
- Wrong ProtoCLIP checkpoint
- Prototype projector not converged

**Solutions:**
- Verify you're using Hugging Face CLIP text encoder
- Train for more epochs
- Check training curves in training_history.json

### Performance Optimization

**Speed up training:**
- Use smaller dataset (ImageNet-Tiny vs full ImageNet)
- Reduce number of workers: `--num_workers 2`
- Skip caption evaluation during training
- Use mixed precision (modify training script to add AMP)

**Improve results:**
- Increase k value for more context
- Train for 100 epochs instead of 50
- Use Hierarchical Pooling for better stability
- Ensure high-quality ProtoCLIP checkpoint

### Dataset Requirements

**ImageNet-Tiny:**
- 200 classes × 500 training images = 100,000 images
- 200 classes × 50 validation images = 10,000 images
- ~3 GB disk space
- Sufficient for prototype projector training

**Full ImageNet:**
- 1000 classes × ~1,300 training images = 1.3M images
- 1000 classes × 50 validation images = 50,000 images
- ~150 GB disk space
- Better results but longer training

---

## Summary

This guide covered:

1. **Prototype-to-Text Generation** - Train projectors to generate captions from prototype activations
2. **CLIP Interpretation** - Zero-training prototype interpretation using pre-trained CLIP
3. **ClipCap Integration** - Natural language generation from CLIP embeddings
4. **TensorBoard** - Real-time training monitoring
5. **Best Practices** - Architecture selection, troubleshooting, optimization

**Quick reference:**

```bash
# Train prototype projector (recommended config)
python scripts/train_prototype_projector.py \
    --projector_type hierarchical \
    --top_k 5 \
    --epochs 100 \
    --evaluate_captions

# Compare configurations
python scripts/compare_prototype_projectors.py \
    --k_values 1 5 10 20 \
    --epochs 25

# Interpret prototypes (no training)
python scripts/interpret_prototypes.py \
    --patch_sizes 32 64 128 \
    --top_k 5

# Monitor training
tensorboard --logdir checkpoints/
```

For core ProtoCLIP training, see the main [README.md](../README.md).
