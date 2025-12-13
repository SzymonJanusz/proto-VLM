# ImageNet Dataset Setup Guide

This guide explains how to download and prepare ImageNet for ProtoCLIP training.

## Table of Contents
1. [Official ImageNet Download](#official-imagenet-download)
2. [Dataset Structure](#dataset-structure)
3. [Alternative Options](#alternative-options)
4. [Verification](#verification)

## Official ImageNet Download

### Option 1: ImageNet ILSVRC2012 (Recommended)

**Requirements**:
- ImageNet account (free registration at https://image-net.org/)
- ~150GB disk space for full dataset
- ~20GB for validation set only

**Steps**:

1. **Register for ImageNet**:
   - Go to https://image-net.org/
   - Create an account and agree to terms of use
   - Request access to ILSVRC2012 dataset

2. **Download the dataset**:
   ```bash
   # Training set (~138GB)
   wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

   # Validation set (~6.3GB)
   wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

   # Development kit (contains class mappings)
   wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
   ```

3. **Extract and organize**:
   ```bash
   # Create directory structure
   mkdir -p imagenet/train imagenet/val

   # Extract training set
   cd imagenet/train
   tar -xf ../../ILSVRC2012_img_train.tar

   # Extract each class tar file
   for f in *.tar; do
       d=`basename $f .tar`
       mkdir -p $d
       tar -xf $f -C $d
       rm $f
   done

   cd ../..

   # Extract validation set
   cd imagenet/val
   tar -xf ../../ILSVRC2012_img_val.tar

   # Run our validation set organization script (see below)
   python ../../scripts/organize_imagenet_val.py
   ```

### Option 2: Kaggle ImageNet (Easier)

**Steps**:

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials**:
   - Go to https://www.kaggle.com/settings
   - Create API token (downloads kaggle.json)
   - Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

3. **Download ImageNet**:
   ```bash
   # Download from Kaggle
   kaggle competitions download -c imagenet-object-localization-challenge

   # Extract
   unzip imagenet-object-localization-challenge.zip -d imagenet_kaggle
   ```

4. **Organize structure**:
   ```bash
   python scripts/organize_kaggle_imagenet.py --input imagenet_kaggle --output imagenet
   ```

## Dataset Structure

After setup, your directory should look like this:

```
imagenet/
├── train/
│   ├── n01440764/          # Class folder (tench)
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/          # Class folder (goldfish)
│   │   ├── n01443537_10007.JPEG
│   │   └── ...
│   └── ...                 # 1000 class folders
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    ├── n01443537/
    │   └── ...
    └── ...                 # 1000 class folders
```

**Key Points**:
- **1000 classes** total (ILSVRC2012)
- **Train set**: ~1.28M images (~1,300 images per class)
- **Val set**: 50,000 images (50 images per class)
- **Class folders**: Named with WordNet IDs (e.g., n01440764)

## Alternative Options

### Option 3: Mini-ImageNet (for Quick Testing)

If you just want to test the code:

```bash
# Download Mini-ImageNet (100 classes, ~60K images)
git clone https://github.com/y2l/mini-imagenet-tools.git
cd mini-imagenet-tools
python process_mini_imagenet.py

# Or use our helper script
python scripts/download_mini_imagenet.py --output ./imagenet_mini
```

### Option 4: Tiny-ImageNet (Smallest)

Tiny-ImageNet is a small subset perfect for rapid experimentation:

```bash
# Download Tiny-ImageNet (200 classes, 100K images, 64x64)
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# Convert to our format
python scripts/organize_tiny_imagenet.py --input tiny-imagenet-200 --output imagenet_tiny
```

### Option 5: Custom Image-Text Dataset

If you don't have ImageNet access, you can use other image-text datasets:

**COCO Captions**:
```bash
# Download COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

Then modify the training script to use COCO instead:
```python
# Use ProtoPNet/data/coco_dataset.py instead of imagenet_dataset.py
```

## Verification

### Check Dataset Structure

Run our verification script:

```bash
python scripts/verify_imagenet.py --imagenet_root ./imagenet
```

This will check:
- ✓ Directory structure is correct
- ✓ Expected number of classes (1000)
- ✓ Images are readable
- ✓ Class folders are properly named

### Quick Test

Test data loading with a small subset:

```bash
python scripts/train.py \
    --imagenet_root ./imagenet \
    --use_subset \
    --subset_samples 100 \
    --warmup_epochs 1 \
    --skip_projection \
    --skip_finetune
```

## Class Mapping File

Our code uses a class mapping file (`imagenet_classes.json`) that maps class indices to human-readable names.

**Download automatically** (recommended):
```python
# This happens automatically when you run train.py
python -c "from ProtoPNet.data.caption_generator import download_imagenet_class_mapping; download_imagenet_class_mapping()"
```

**Or download manually**:
```bash
wget https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json -O data/imagenet_classes.json
```

## Disk Space Requirements

| Dataset | Train | Val | Total |
|---------|-------|-----|-------|
| Full ImageNet | 138GB | 6.3GB | **~145GB** |
| Mini-ImageNet | 2.5GB | 0.5GB | **~3GB** |
| Tiny-ImageNet | 250MB | 50MB | **~300MB** |

## Troubleshooting

### Issue 1: "Split directory not found"
**Error**: `ValueError: Split directory not found: imagenet/train`

**Solution**: Check that you've extracted the dataset and organized it correctly. The directory should be `imagenet/train`, not `imagenet/ILSVRC2012/train`.

### Issue 2: Validation set not organized by class
**Problem**: Validation images are flat in one directory

**Solution**: Run the validation organization script:
```bash
python scripts/organize_imagenet_val.py --val_dir imagenet/val --devkit_dir ILSVRC2012_devkit_t12
```

### Issue 3: Permission denied downloading from image-net.org
**Problem**: Need to be logged in to download

**Solution**:
1. Use Kaggle option (easier)
2. Or use `wget --user=<username> --password=<password>` with ImageNet credentials

### Issue 4: Out of disk space
**Problem**: Not enough space for full ImageNet

**Solutions**:
- Use Mini-ImageNet (3GB) or Tiny-ImageNet (300MB)
- Download only validation set for testing
- Use external hard drive or cloud storage

## Next Steps

Once ImageNet is set up:

1. **Verify setup**:
   ```bash
   python scripts/verify_imagenet.py --imagenet_root ./imagenet
   ```

2. **Download pretrained Proto-CLIP**:
   ```bash
   python scripts/download_pretrained.py
   ```

3. **Test with small subset**:
   ```bash
   python scripts/train.py \
       --imagenet_root ./imagenet \
       --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
       --use_subset \
       --subset_samples 1000
   ```

4. **Full training**:
   ```bash
   python scripts/train.py \
       --imagenet_root ./imagenet \
       --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
       --batch_size 128 \
       --num_workers 8
   ```

## Support

- **ImageNet official**: https://image-net.org/
- **Kaggle ImageNet**: https://www.kaggle.com/c/imagenet-object-localization-challenge
- **Tiny-ImageNet**: http://cs231n.stanford.edu/tiny-imagenet-200.zip
- **COCO**: https://cocodataset.org/

For issues with this code, check the main README or create an issue.
