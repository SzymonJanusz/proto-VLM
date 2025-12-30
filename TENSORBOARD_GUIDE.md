# TensorBoard Logging Guide

TensorBoard logging has been integrated into the training script to help you monitor training progress in real-time.

## What's Being Logged

All metrics are now organized by training stage (warmup or finetune) for clear separation in TensorBoard.

### Warmup Stage Metrics

#### Training Metrics (per batch and per epoch)
- **warmup/train/batch_loss**: Loss for each batch during warmup training
- **warmup/train/contrastive_loss**: Contrastive loss component
- **warmup/train/clustering_loss**: Clustering loss component (if enabled)
- **warmup/train/activation_loss**: Activation loss component (if enabled)
- **warmup/train/epoch_loss**: Average loss per epoch
- **warmup/train/epoch_contrastive_loss**: Average contrastive loss per epoch
- **warmup/train/epoch_clustering_loss**: Average clustering loss per epoch (if enabled)
- **warmup/train/epoch_activation_loss**: Average activation loss per epoch (if enabled)

#### Validation Metrics
- **warmup/val/loss**: Validation loss per epoch
- **warmup/val/contrastive_loss**: Validation contrastive loss

#### Learning Rate
- **warmup/learning_rate**: Learning rate schedule during warmup

### Fine-tuning Stage Metrics

#### Training Metrics (per batch and per epoch)
- **finetune/train/batch_loss**: Loss for each batch during fine-tuning
- **finetune/train/contrastive_loss**: Contrastive loss component
- **finetune/train/clustering_loss**: Clustering loss component (if enabled)
- **finetune/train/activation_loss**: Activation loss component (if enabled)
- **finetune/train/epoch_loss**: Average loss per epoch
- **finetune/train/epoch_contrastive_loss**: Average contrastive loss per epoch
- **finetune/train/epoch_clustering_loss**: Average clustering loss per epoch (if enabled)
- **finetune/train/epoch_activation_loss**: Average activation loss per epoch (if enabled)

#### Validation Metrics
- **finetune/val/loss**: Validation loss per epoch
- **finetune/val/contrastive_loss**: Validation contrastive loss

#### Learning Rate
- **finetune/learning_rate**: Learning rate schedule during fine-tuning

## Stage Separation Benefits

The metrics are now organized by training stage, which provides several advantages:

1. **Clear Visual Distinction**: Warmup and fine-tuning metrics appear in separate folders in TensorBoard's sidebar
2. **Easy Stage Comparison**: Can overlay warmup vs finetune curves to compare learning dynamics
3. **No Confusion**: Each stage has its own independent timeline, preventing metric overlap
4. **Better Analysis**: Easier to identify which stage needs hyperparameter tuning

## TensorBoard Structure

In the TensorBoard interface, you'll see metrics organized like this:

```
warmup/
  ├── train/
  │   ├── batch_loss
  │   ├── contrastive_loss
  │   ├── epoch_loss
  │   └── ...
  ├── val/
  │   ├── loss
  │   └── contrastive_loss
  └── learning_rate

finetune/
  ├── train/
  │   ├── batch_loss
  │   ├── contrastive_loss
  │   ├── epoch_loss
  │   └── ...
  ├── val/
  │   ├── loss
  │   └── contrastive_loss
  └── learning_rate
```

## How to Use TensorBoard

### 1. During Training

When you run the training script, TensorBoard logs will be automatically saved to:
```
./checkpoints/<your_checkpoint_dir>/tensorboard/run_<timestamp>/
```

The training script will print the exact path when it starts.

### 2. Launch TensorBoard

Open a new terminal and run:

```bash
tensorboard --logdir ./checkpoints/<your_checkpoint_dir>/tensorboard
```

Or to view all runs:
```bash
tensorboard --logdir ./checkpoints/overfitting_fix_2/tensorboard
```

### 3. View in Browser

After launching TensorBoard, open your browser and navigate to:
```
http://localhost:6006
```

## Key Features

### Scalars Tab
- View all logged metrics over time
- Compare multiple runs side-by-side
- Smooth curves with adjustable smoothing factor
- Download data as CSV

### Time Series Tab
- Monitor training progress in real-time
- See how loss evolves during training
- Identify overfitting (when train loss decreases but val loss increases)

## Example Usage

```bash
# Start training (logs will be created automatically)
python scripts/train.py \
    --imagenet_root ./imagenet_tiny \
    --pretrained_protoclip ./pretrained_checkpoints/proto_clip_imagenet \
    --batch_size 32 \
    --checkpoint_dir ./checkpoints/experiment_1

# In another terminal, launch TensorBoard
tensorboard --logdir ./checkpoints/experiment_1/tensorboard

# Open browser to http://localhost:6006
```

## Tips

1. **Compare Runs**: Keep all runs in the same tensorboard directory to compare different experiments
2. **Real-time Monitoring**: TensorBoard updates automatically - no need to refresh
3. **Smoothing**: Use the smoothing slider to remove noise from training curves
4. **Multiple Experiments**: Run multiple experiments with different hyperparameters and compare in TensorBoard
5. **Compare Stages**: To compare warmup vs finetune performance:
   - Select both `warmup/val/loss` and `finetune/val/loss` in TensorBoard
   - Use the overlay feature to see both curves on the same plot
   - Analyze whether fine-tuning improves over warmup results
6. **Stage-Specific Analysis**: Focus on one stage at a time by expanding only that folder in the sidebar

## Troubleshooting

**TensorBoard not updating?**
- Make sure training is still running
- Try refreshing the browser
- Check that the log directory is correct

**Port 6006 already in use?**
```bash
tensorboard --logdir <path> --port 6007
```

**Can't find logs?**
- Check the console output when training starts for the exact log directory path
- Make sure the checkpoint directory exists
