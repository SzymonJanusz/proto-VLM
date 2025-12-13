# ProtoPNet-CLIP Hybrid Model

A prototype-based vision-language model combining ProtoPNet's interpretable image encoder with CLIP's text encoder.

## Overview

This project implements a hybrid architecture that replaces CLIP's image encoder with a ProtoPNet-based network that learns interpretable prototypes. The model maintains CLIP's zero-shot capabilities while providing visual explanations through prototype activations.

## Architecture

```
IMAGE BRANCH:
Image (224×224)
  → ResNet-50 layer3 (1024×14×14 features)
  → Prototype Layer (200 prototypes)
  → Weighted Pooling
  → Projection Head (200 → 512-D)
  → L2 Normalize
  → Image Embedding (512-D)

TEXT BRANCH (CLIP):
Text → CLIP Tokenizer → CLIP Text Encoder → Text Embedding (512-D)

TRAINING:
Contrastive loss (InfoNCE) between image and text embeddings
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision transformers opencv-python PyYAML matplotlib tqdm
```

## Project Structure

```
ProtoPNet/
├── models/
│   ├── prototype_layer.py       # Core prototype layer
│   ├── resnet_features.py       # ResNet backbone
│   ├── protopnet_encoder.py     # ProtoPNet image encoder
│   ├── clip_text_encoder.py     # CLIP text encoder wrapper
│   └── hybrid_model.py          # Combined ProtoCLIP model
├── training/
│   └── losses.py                # Loss functions
├── data/
│   └── (datasets and dataloaders - to be implemented)
├── utils/
│   └── (visualization and projection tools - to be implemented)
└── configs/
    └── base_config.yaml         # Hyperparameters
```

## Quick Start

### 1. Basic Usage

```python
from ProtoPNet.models.hybrid_model import ProtoCLIP

# Create model
model = ProtoCLIP(
    num_prototypes=200,
    image_backbone='resnet50',
    text_model='openai/clip-vit-base-patch32',
    embedding_dim=512,
    freeze_text_encoder=True,
    temperature=0.07,
    pooling_mode='max'
)

# Example forward pass
import torch

images = torch.randn(4, 3, 224, 224)  # Batch of 4 images
texts = ['a photo of a cat', 'a photo of a dog', 'an image of a bird', 'a picture of a car']

# Get logits and embeddings
logits_per_image, logits_per_text, image_emb, text_emb = model(images, texts)

print(f"Image embeddings shape: {image_emb.shape}")  # (4, 512)
print(f"Text embeddings shape: {text_emb.shape}")    # (4, 512)
print(f"Logits shape: {logits_per_image.shape}")     # (4, 4)
```

### 2. Training (3-Stage Process)

**Stage 1: Warmup** - Train CNN backbone + prototypes (text encoder frozen)
```python
from ProtoPNet.training.losses import CombinedLoss
import torch.optim as optim

# Setup
optimizer = optim.AdamW([
    {'params': model.image_encoder.backbone.parameters(), 'lr': 1e-4},
    {'params': model.image_encoder.prototype_layer.parameters(), 'lr': 1e-3},
    {'params': model.image_encoder.projection_head.parameters(), 'lr': 1e-3}
])

loss_fn = CombinedLoss(
    lambda_contrastive=1.0,
    lambda_clustering=0.01,
    lambda_activation=0.01
)

# Training loop (simplified)
for epoch in range(10):
    for images, texts in train_loader:
        optimizer.zero_grad()

        # Forward with similarity maps
        logits_i, logits_t, img_emb, txt_emb, proto_sims = model(
            images, texts, return_similarities=True
        )

        # Compute pooled similarities
        pooled_sims = model.image_encoder.pooling(proto_sims)

        # Loss
        loss, loss_dict = loss_fn(
            logits_i, logits_t,
            prototype_vectors=model.get_prototypes(),
            prototype_similarities=proto_sims,
            pooled_similarities=pooled_sims
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Stage 2: Prototype Projection** - Ground prototypes in training patches
```python
# TODO: Implement PrototypeProjector
# This will find nearest training patches for each prototype
```

**Stage 3: Fine-tuning** - Fine-tune projection head
```python
# Freeze backbone and prototypes
for param in model.image_encoder.backbone.parameters():
    param.requires_grad = False
for param in model.image_encoder.prototype_layer.parameters():
    param.requires_grad = False

# Fine-tune projection head only
optimizer = optim.AdamW([
    {'params': model.image_encoder.projection_head.parameters(), 'lr': 1e-5}
])

# Continue training...
```

## Configuration

Edit `ProtoPNet/configs/base_config.yaml` to customize:
- Number of prototypes
- Backbone architecture
- Learning rates
- Batch size
- Loss weights

## Key Features

✅ **Implemented**:
- Prototype-based image encoder
- CLIP text encoder integration
- Contrastive loss (InfoNCE)
- Prototype diversity loss
- Prototype activation loss
- Multi-stage training support
- Configurable hyperparameters

🚧 **To Be Implemented**:
- Training scripts (`scripts/train.py`)
- Dataset loaders (`ProtoPNet/data/datasets.py`)
- Prototype projection (`ProtoPNet/utils/projection.py`)
- Visualization tools (`ProtoPNet/utils/visualization.py`)
- Evaluation metrics (`ProtoPNet/utils/metrics.py`)

## Data Requirements

**Minimum**: 10K image-text pairs for training

**Recommended Datasets**:
- COCO Captions: 118K images with 5 captions each
- Conceptual Captions: 3.3M image-text pairs
- Custom domain data: 1K+ pairs for fine-tuning

**Data Format**: JSON file with structure:
```json
[
    {"image_path": "/path/to/image1.jpg", "caption": "a description"},
    {"image_path": "/path/to/image2.jpg", "caption": "another description"}
]
```

## Model Components

### PrototypeLayer
Computes L2 distance between image features and learned prototypes:
- Input: (B, C, H, W) feature maps
- Output: (B, M, H, W) similarity maps

### WeightedPooling
Aggregates spatial prototype activations:
- **Max pooling**: Finds strongest activation per prototype
- **Attention pooling**: Learnable weighted aggregation

### ProtoPNetEncoder
Complete image encoder:
1. ResNet-50 backbone (layer3)
2. Prototype layer (200 prototypes)
3. Weighted pooling
4. Projection head (200 → 512-D)
5. L2 normalization

### CLIPTextEncoder
Wrapper around pretrained CLIP text encoder:
- Frozen by default (preserves CLIP's knowledge)
- Can be unfrozen for task-specific adaptation

## References

- **ProtoPNet**: ["This Looks Like That: Deep Learning for Interpretable Image Recognition"](https://github.com/cfchen-duke/ProtoPNet) (NeurIPS 2019)
- **CLIP**: [Contrastive Language-Image Pretraining](https://github.com/openai/CLIP)
- **Proto-CLIP**: [Vision-Language Prototypical Network](https://arxiv.org/abs/2307.03073) (IROS 2024)

## Next Steps

1. Implement dataset loaders for COCO Captions
2. Create training script with 3-stage procedure
3. Implement prototype projection utility
4. Add visualization tools for prototype activation maps
5. Evaluate on retrieval benchmarks

## License

This project builds upon:
- ProtoPNet (Duke University)
- CLIP (OpenAI)
- Transformers (Hugging Face)

Please cite the original papers if you use this code in your research.
