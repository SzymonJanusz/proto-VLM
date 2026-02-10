# Proto-VLM: ProtoPNet with CLIP Interpretation

Interpretable image classification using ProtoPNet with CLIP-based prototype interpretation.

## Overview

This project combines:
- **ProtoPNet** - Interpretable deep learning using prototypical parts
- **CLIP** - Vision-language model for semantic interpretation

The system learns prototypical image patches and uses CLIP to provide natural language interpretations of what each prototype represents - **without any additional training**.

## Quick Start

### 1. Train ProtoPNet Model

```bash
venv/Scripts/python scripts/train.py --imagenet_root data/imagenet --num_prototypes 200
```

### 2. Interpret Prototypes with CLIP

**Single Prototype:**
```bash
venv/Scripts/python scripts/interpret_prototypes.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --prototype_id 0 --patch_sizes 64 --visualize --save_images --output_dir results
```

**All Classes (Organized by Class):**
```bash
venv/Scripts/python scripts/interpret_by_class.py --projection_info checkpoints/overfitting_fix_4/projection_info.pkl --class_mapping data/imagenet_classes.json --patch_sizes 64 --save_images --save_summary --output_dir results/by_class
```

**Output:**
- `results/visualizations/prototype_0.png` - Visual elements
- `results/visualizations/prototype_0_interpretations.txt` - Interpretation results
- `results/by_class/<class_name>/` - Per-class organized results

## Documentation

- **[TECHNICAL_APPROACH.md](ProtoPNet/TECHNICAL_APPROACH.md)** - Technical research explanation of the interpretation methodology
- **[CLIP_INTERPRETATION_README.md](CLIP_INTERPRETATION_README.md)** - Complete usage guide for CLIP-based prototype interpretation
- **[TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)** - TensorBoard usage guide

## Key Features

✅ **Interpretable by design** - Prototypical parts-based reasoning
✅ **Natural language explanations** - CLIP provides semantic interpretations
✅ **No additional training** - Uses pre-trained models directly
✅ **Multiple patch sizes** - Experiment with different context windows
✅ **Clean visualizations** - Separate image and text outputs
✅ **Batch processing** - Interpret all prototypes at once
✅ **Class-based organization** - Analyze prototypes grouped by their class

## Project Structure

```
proto-VLM/
├── ProtoPNet/              # Core ProtoPNet implementation
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   ├── utils/              # Utilities including CLIP interpretation
│   └── data/              # Data loading
├── scripts/                # Training and interpretation scripts
├── checkpoints/            # Model checkpoints
├── data/                   # Dataset and mappings
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch
- transformers (HuggingFace)
- Pillow, numpy, matplotlib

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
