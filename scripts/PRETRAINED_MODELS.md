# Using Pretrained Models from Proto-CLIP

This guide explains how to download and use Proto-CLIP's pretrained ImageNet checkpoints to initialize your ProtoPNet-CLIP model.

## Why Use Pretrained Models?

Starting from Proto-CLIP's pretrained weights provides several benefits:
- ✅ **Better initialization** than random weights
- ✅ **Faster convergence** during training
- ✅ **Better final performance**
- ✅ **Less prone to dead prototypes** (initialized from real data)

## Quick Start

### Step 1: Download Pretrained Checkpoints

```bash
# Download Proto-CLIP ImageNet-F checkpoints
python scripts/download_pretrained.py --dataset imagenet-F

# Verify downloads
python scripts/download_pretrained.py --verify_only
```

This will download 3 checkpoint files to `./pretrained_checkpoints/proto_clip_imagenet/`:
- `memory_bank_v.pt` - Visual prototypes (will initialize our PrototypeLayer)
- `memory_bank_t.pt` - Text prototypes (optional reference)
- `query_adapter.pt` - Adapter network (may initialize projection head)

### Step 2: Inspect Checkpoint Structure

```bash
# Inspect all checkpoints
python scripts/inspect_checkpoints.py

# Inspect specific checkpoint
python scripts/inspect_checkpoints.py --checkpoint ./pretrained_checkpoints/proto_clip_imagenet/memory_bank_v.pt

# Brief output
python scripts/inspect_checkpoints.py --brief
```

**Important**: Run inspection BEFORE implementing the converter to understand the checkpoint format!

### Step 3: Convert and Load Checkpoints

Once you've inspected the checkpoints and implemented the converter:

```python
from ProtoPNet.models.hybrid_model import ProtoCLIP

# Create model with pretrained initialization
model = ProtoCLIP(
    num_prototypes=200,
    pretrained_protoclip_path="../pretrained_checkpoints/proto_clip_imagenet/",
    freeze_text_encoder=True
)

# Prototypes are now initialized from Proto-CLIP!
print("Model initialized with pretrained weights")
```

## Checkpoint Mapping

Proto-CLIP checkpoints map to our model as follows:

```
Proto-CLIP Checkpoint          →  Our ProtoPNet-CLIP Model
=======================           ===========================
memory_bank_v.pt               →  PrototypeLayer.prototype_vectors
memory_bank_t.pt               →  [Optional reference]
query_adapter.pt               →  projection_head (if compatible)
```

## Training with Pretrained Models

### Modified Training Workflow

**Stage 0: Download and Initialize** (NEW)
```bash
python scripts/download_pretrained.py --dataset imagenet-F
python scripts/inspect_checkpoints.py
```

**Stage 1: Warmup** (Modified - use lower learning rates)
```python
model = ProtoCLIP(
    pretrained_protoclip_path="./pretrained_checkpoints/proto_clip_imagenet/"
)

# Lower learning rates since starting from pretrained
optimizer = AdamW([
    {'params': model.image_encoder.backbone.parameters(), 'lr': 5e-5},  # was 1e-4
    {'params': model.image_encoder.prototype_layer.parameters(), 'lr': 5e-4},  # was 1e-3
    {'params': model.image_encoder.projection_head.parameters(), 'lr': 5e-4}
])
```

**Stages 2-3**: Continue as normal (projection and fine-tuning)

## Handling Dimension Mismatches

The checkpoint converter handles dimension mismatches gracefully:

```python
from ProtoPNet.utils.checkpoint_converter import ProtoClipCheckpointConverter

# Flexible loading (adapts to dimension mismatches)
converter = ProtoClipCheckpointConverter(strict=False)
converter.initialize_model_from_protoclip(
    model,
    visual_memory_path="./pretrained_checkpoints/proto_clip_imagenet/memory_bank_v.pt"
)

# Strict loading (raises error on mismatch)
converter = ProtoClipCheckpointConverter(strict=True)
```

**Common Issues**:
1. **Prototype count mismatch**: Converter will subsample or interpolate
2. **Feature dimension mismatch**: May need to adjust backbone architecture
3. **Adapter incompatibility**: Converter will skip if dimensions don't match

## Verification

After loading pretrained weights, verify they're working correctly:

```python
import torch

# Test forward pass
images = torch.randn(4, 3, 224, 224)
texts = ['test caption'] * 4

logits_i, logits_t, img_emb, txt_emb = model(images, texts)

print(f"Image embeddings: {img_emb.shape}")  # Should be (4, 512)
print(f"Embeddings normalized: {torch.allclose(img_emb.norm(dim=-1), torch.ones(4))}")  # Should be True

# Check prototypes
prototypes = model.get_prototypes()
print(f"Prototypes shape: {prototypes.shape}")  # e.g., (200, 1024)
print(f"Prototypes stats - mean: {prototypes.mean():.4f}, std: {prototypes.std():.4f}")
```

## Command Reference

### Download Commands
```bash
# Download ImageNet-F checkpoints (default)
python scripts/download_pretrained.py --dataset imagenet-F

# Download to custom directory
python scripts/download_pretrained.py --dataset imagenet-F --save_dir ./my_checkpoints/

# Verify existing checkpoints
python scripts/download_pretrained.py --verify_only
```

### Inspection Commands
```bash
# Inspect all checkpoints
python scripts/inspect_checkpoints.py

# Inspect checkpoints in custom directory
python scripts/inspect_checkpoints.py --checkpoint_dir ./my_checkpoints/

# Inspect single checkpoint file
python scripts/inspect_checkpoints.py --checkpoint ./path/to/checkpoint.pt

# Brief output (less verbose)
python scripts/inspect_checkpoints.py --brief
```

## Troubleshooting

### Problem: Download fails
**Solution**: Check internet connection and GitHub accessibility
```bash
# Try manual download from browser:
# https://github.com/IRVLUTD/Proto-CLIP/tree/main/pretrained_ckpt/imagenet-F
```

### Problem: Checkpoint structure unexpected
**Solution**: Run inspection script to see actual structure
```bash
python scripts/inspect_checkpoints.py
# Adjust converter implementation based on output
```

### Problem: Dimension mismatch errors
**Solution**: Use flexible loading mode
```python
converter = ProtoClipCheckpointConverter(strict=False)
```

### Problem: Out of memory when loading
**Solution**: Load to CPU first
```python
checkpoint = torch.load(path, map_location='cpu')
model = model.to('cuda')  # Move model to GPU after loading
```

## Next Steps

1. ✅ Download checkpoints using `download_pretrained.py`
2. ✅ Inspect structure using `inspect_checkpoints.py`
3. ⏭️ Implement `checkpoint_converter.py` based on inspection results
4. ⏭️ Test loading checkpoints into model
5. ⏭️ Train model starting from pretrained weights

## References

- Proto-CLIP Repository: https://github.com/IRVLUTD/Proto-CLIP
- Proto-CLIP Paper: https://arxiv.org/abs/2307.03073
- ImageNet-F Checkpoints: https://github.com/IRVLUTD/Proto-CLIP/tree/main/pretrained_ckpt/imagenet-F
