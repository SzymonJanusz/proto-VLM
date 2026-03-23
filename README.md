# Proto-VLM: ProtoCLIP with Spatial Feature Alignment

Interpretable image classification using ProtoPNet with CLIP-based prototype interpretation and spatial feature alignment.

## Overview

This project combines:
- **ProtoPNet** - Interpretable deep learning using prototypical parts
- **CLIP** - Vision-language model for semantic interpretation
- **Spatial MSE Projector** - Maps prototypes to CLIP spatial features for dense prediction tasks

The system learns prototypical image patches and provides:
1. **Global embeddings** for text-image retrieval and zero-shot classification
2. **Spatial features** for dense prediction tasks (segmentation, localization)
3. **Natural language interpretations** of prototypes using CLIP

## Training

### Option 1: ProtoCLIP 3-Stage Training (Recommended)

Train ProtoCLIP for text-image retrieval and zero-shot classification:

```bash
# Standard 3-stage training
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --batch_size 64 \
    --warmup_epochs 10 \
    --finetune_epochs 10

# With pretrained Proto-CLIP initialization (better performance)
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --augmentation_strength strong_v2 \
    --use_cutmix \
    --batch_size 64
```

**Pipeline:**
- **Stage 1 (Warmup)**: Train backbone + prototypes with contrastive learning
- **Stage 2 (Projection)**: Ground prototypes in real image patches
- **Stage 3 (Fine-tuning)**: Fine-tune projection head for better alignment

**Outputs:** `warmup_best.pt`, `finetune_best.pt`

### Option 2: ProtoCLIP + Spatial MSE Projector (Full Pipeline)

Train both global embeddings AND spatial features:

```bash
# Full 4-stage training with sparse spatial projector (top-5)
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --train_spatial_projector \
    --spatial_top_k 5 \
    --spatial_freeze_backbone \
    --batch_size 64
```

**Additional Stage 4:** Trains spatial MSE projector that maps prototype similarities to CLIP layer3 spatial features (1024-dim, 14×14)

**Outputs:** `warmup_best.pt`, `finetune_best.pt`, `spatial_projector_best.pt`

### Option 3: Standalone Spatial MSE Projector

Train only the spatial projector independently:

```bash
# From pretrained Proto-CLIP repository checkpoints (recommended)
python scripts/train_spatial_mse_projector.py \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --sampling_method kmeans \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_protoclip \
    --epochs 50 \
    --batch_size 64 \
    --top_k 5

# From local checkpoint (train.py output)
python scripts/train_spatial_mse_projector.py \
    --backbone_checkpoint checkpoints/finetune_best.pt \
    --freeze_backbone \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_frozen \
    --epochs 50 \
    --batch_size 64 \
    --top_k 1

# From scratch (random initialization)
python scripts/train_spatial_mse_projector.py \
    --train_from_scratch \
    --data_root imagenet_tiny \
    --output_dir checkpoints/spatial_mse_scratch \
    --epochs 50 \
    --batch_size 64 \
    --top_k 1
```

**Sparsity Control:**
- `--top_k 1`: Most sparse (only 1 prototype per image)
- `--top_k 5`: Moderate sparsity
- `--top_k 200`: No sparsity (all prototypes)

## Prototype Interpretation

After training, interpret prototypes using CLIP:

**Single Prototype:**
```bash
python scripts/interpret_prototypes.py \
    --projection_info checkpoints/projection_info.pkl \
    --class_mapping data/imagenet_classes.json \
    --prototype_id 0 \
    --patch_sizes 64 \
    --visualize \
    --save_images \
    --output_dir results
```

**All Classes (Organized by Class):**
```bash
python scripts/interpret_by_class.py \
    --projection_info checkpoints/projection_info.pkl \
    --class_mapping data/imagenet_classes.json \
    --patch_sizes 64 \
    --save_images \
    --save_summary \
    --output_dir results/by_class
```

**Output:**
- `results/visualizations/prototype_0.png` - Visual elements
- `results/visualizations/prototype_0_interpretations.txt` - Interpretation results
- `results/by_class/<class_name>/` - Per-class organized results

## Checkpoints

After training with `train.py`, you'll have:

```
checkpoints/
├── warmup_best.pt              # After Stage 1 (backbone + prototypes)
├── finetune_best.pt            # After Stage 3 (full ProtoCLIP model)
└── spatial_projector_best.pt   # After Stage 4 (optional, spatial features)
```

**Loading for inference:**
```python
from ProtoPNet.models.hybrid_model import ProtoCLIP
from ProtoPNet.models.spatial_mse_projector import SpatialMSEProjector

# Load ProtoCLIP for retrieval/classification
model = ProtoCLIP(num_prototypes=200, ...)
checkpoint = torch.load('checkpoints/finetune_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Optionally load spatial projector
spatial_proj = SpatialMSEProjector.load('checkpoints/spatial_projector_best.pt')

# Use both
image_embeddings = model.encode_image(images)        # (B, 512) global
proto_sims = model.image_encoder.get_similarities(images)  # (B, 200, 14, 14)
spatial_features = spatial_proj(proto_sims)          # (B, 1024, 14, 14) spatial
```

## Documentation

- **[TECHNICAL_APPROACH.md](ProtoPNet/TECHNICAL_APPROACH.md)** - Technical research explanation of the interpretation methodology
- **[CLIP_INTERPRETATION_README.md](CLIP_INTERPRETATION_README.md)** - Complete usage guide for CLIP-based prototype interpretation
- **[TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)** - TensorBoard usage guide

## Monitoring Training

All training scripts support TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir checkpoints/tensorboard

# View at http://localhost:6006
```

**Metrics tracked:**
- Train/validation loss
- Cosine similarity (spatial projector)
- Learning rate schedules
- Sparsity statistics (spatial projector)

## Key Features

### Training
✅ **3-Stage ProtoCLIP Training** - Warmup → Projection → Fine-tuning
✅ **Optional 4th Stage** - Add spatial MSE projector for dense predictions
✅ **Pretrained Proto-CLIP Init** - Initialize from 16k Proto-CLIP prototypes
✅ **Multiple Training Scripts** - Flexible pipelines for different use cases
✅ **Sparse Prototype Selection** - Configurable top-k sparsity (1-200)
✅ **Data Augmentation** - CutMix, strong augmentation, label smoothing

### Model Capabilities
✅ **Global Embeddings** - Text-image retrieval, zero-shot classification
✅ **Spatial Features** - CLIP-aligned 14×14 feature maps for dense tasks
✅ **Interpretable Prototypes** - Grounded in real image patches
✅ **Natural Language Explanations** - CLIP provides semantic interpretations

### Tools & Utilities
✅ **TensorBoard Integration** - Real-time training monitoring
✅ **Early Stopping** - Automatic training termination
✅ **Multiple Checkpoints** - Best and periodic model saving
✅ **Batch Processing** - Interpret all prototypes at once
✅ **Class-based Organization** - Analyze prototypes grouped by their class

## Project Structure

```
proto-VLM/
├── ProtoPNet/                          # Core ProtoPNet implementation
│   ├── models/
│   │   ├── hybrid_model.py            # ProtoCLIP model
│   │   ├── spatial_mse_projector.py   # Spatial MSE projector
│   │   └── clip_spatial_extractor.py  # CLIP layer3 extractor
│   ├── training/
│   │   ├── trainer.py                 # ProtoCLIP trainer
│   │   └── losses.py                  # Combined, Spatial MSE losses
│   ├── utils/                         # CLIP interpretation, projection
│   └── data/                          # Data loading, transforms
├── scripts/
│   ├── train.py                       # 3/4-stage ProtoCLIP training
│   ├── train_spatial_mse_projector.py # Standalone spatial projector
│   ├── train_prototype_projector.py   # Alternative projector training
│   ├── train_cnn_projector.py         # Baseline CNN projector
│   ├── interpret_prototypes.py        # Single prototype interpretation
│   └── interpret_by_class.py          # Class-based interpretation
├── checkpoints/                       # Model checkpoints
├── data/                              # Dataset and mappings
└── README.md                          # This file
```

## Training Scripts Comparison

| Script | Purpose | Training Time | Output | Use Case |
|--------|---------|---------------|--------|----------|
| **train.py** | Full ProtoCLIP pipeline (3 or 4 stages) | Long (3-4 stages) | Global + optional spatial | Complete training from scratch |
| **train_spatial_mse_projector.py** | Spatial projector only | Medium (50 epochs) | Spatial features | Add spatial capability to existing model |
| **train_prototype_projector.py** | Prototype → CLIP global | Long (100 epochs) | Global embeddings | Alternative to train.py Stage 3 |
| **train_cnn_projector.py** | CNN → CLIP (baseline) | Short (10 epochs) | Global embeddings | Comparison baseline |

**Recommendation:**
- **New project**: Use `train.py` with `--pretrained_protoclip` and `--train_spatial_projector`
- **Add spatial features**: Use `train_spatial_mse_projector.py --backbone_checkpoint`
- **Quick experiment**: Use `train_spatial_mse_projector.py --pretrained_protoclip`

## Requirements

- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- Pillow, numpy, matplotlib
- tqdm, tensorboard

Install dependencies:
```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:

**ProtoPNet:**
```
@inproceedings{chen2019protopnet,
  title={This looks like that: deep learning for interpretable image recognition},
  author={Chen, Chaofan and Li, Oscar and Tao, Daniel and Barnett, Alina and Rudin, Cynthia and Su, Jonathan K},
  booktitle={NeurIPS},
  year={2019}
}
```

**CLIP:**
```
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

## License

This project builds upon ProtoPNet. Please refer to the original repositories for licensing information.
