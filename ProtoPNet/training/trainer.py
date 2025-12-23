"""
Training loop orchestration for ProtoCLIP model.
Implements 3-stage training: Warmup → Projection → Fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class ProtoCLIPTrainer:
    """
    Trainer for ProtoCLIP model.

    Implements 3-stage training:
    1. Warmup: Train image encoder + prototypes (text encoder frozen)
    2. Projection: Project prototypes to training patches (no training)
    3. Fine-tuning: Fine-tune projection head (backbone + prototypes frozen)

    Args:
        model: ProtoCLIP model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to train on ('cuda' or 'cpu')
        log_interval: Log every N batches
        checkpoint_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device='cuda',
        log_interval=100,
        checkpoint_dir='./checkpoints',
        tensorboard_writer=None,
        stage_name=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = tensorboard_writer
        self.stage_name = stage_name if stage_name else ""

        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, return_similarities=False, augmenter=None):
        """
        Train for one epoch.

        Args:
            return_similarities: Whether to compute prototype similarities (needed for aux losses)
            augmenter: Optional augmentation function (e.g., CutMix) to apply to images

        Returns:
            tuple: (average_loss, loss_dict_list)
        """
        self.model.train()
        epoch_losses = []
        epoch_loss_dicts = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, texts) in enumerate(pbar):
            images = images.to(self.device)

            # Apply augmentation if provided (e.g., CutMix)
            if augmenter is not None:
                images, _ = augmenter(images)

            # Forward pass
            if return_similarities:
                logits_i, logits_t, img_emb, txt_emb, proto_sims = self.model(
                    images, texts, return_similarities=True
                )
                # Get pooled similarities for activation loss
                pooled_sims = self.model.image_encoder.pooling(proto_sims)
            else:
                logits_i, logits_t, img_emb, txt_emb = self.model(images, texts)
                proto_sims = None
                pooled_sims = None

            # Compute loss
            loss, loss_dict = self.loss_fn(
                logits_i,
                logits_t,
                prototype_vectors=self.model.get_prototypes() if return_similarities else None,
                prototype_similarities=proto_sims,
                pooled_similarities=pooled_sims
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Logging
            epoch_losses.append(loss.item())
            epoch_loss_dicts.append(loss_dict)

            # Log to TensorBoard every batch
            if self.writer is not None:
                prefix = f"{self.stage_name}/" if self.stage_name else ""
                self.writer.add_scalar(f'{prefix}train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar(f'{prefix}train/contrastive_loss', loss_dict.get('contrastive', 0), self.global_step)
                if 'clustering' in loss_dict:
                    self.writer.add_scalar(f'{prefix}train/clustering_loss', loss_dict['clustering'], self.global_step)
                if 'activation' in loss_dict:
                    self.writer.add_scalar(f'{prefix}train/activation_loss', loss_dict['activation'], self.global_step)

            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cont': f"{loss_dict.get('contrastive', 0):.4f}"
                })

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        # Log epoch-level metrics to TensorBoard
        if self.writer is not None:
            prefix = f"{self.stage_name}/" if self.stage_name else ""
            self.writer.add_scalar(f'{prefix}train/epoch_loss', avg_loss, self.current_epoch)
            # Compute average component losses for the epoch
            avg_contrastive = np.mean([d.get('contrastive', 0) for d in epoch_loss_dicts])
            self.writer.add_scalar(f'{prefix}train/epoch_contrastive_loss', avg_contrastive, self.current_epoch)
            if 'clustering' in epoch_loss_dicts[0]:
                avg_clustering = np.mean([d['clustering'] for d in epoch_loss_dicts])
                self.writer.add_scalar(f'{prefix}train/epoch_clustering_loss', avg_clustering, self.current_epoch)
            if 'activation' in epoch_loss_dicts[0]:
                avg_activation = np.mean([d['activation'] for d in epoch_loss_dicts])
                self.writer.add_scalar(f'{prefix}train/epoch_activation_loss', avg_activation, self.current_epoch)

        self.current_epoch += 1

        return avg_loss, epoch_loss_dicts

    @torch.no_grad()
    def validate(self):
        """
        Validate model.

        Returns:
            tuple: (average_loss, loss_dict_list)
        """
        self.model.eval()
        val_losses = []
        val_loss_dicts = []

        for images, texts in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)

            # Forward pass
            logits_i, logits_t, img_emb, txt_emb = self.model(images, texts)

            # Compute loss (without auxiliary losses for validation)
            loss, loss_dict = self.loss_fn(logits_i, logits_t)

            val_losses.append(loss.item())
            val_loss_dicts.append(loss_dict)

        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)

        # Log validation metrics to TensorBoard
        if self.writer is not None:
            prefix = f"{self.stage_name}/" if self.stage_name else ""
            self.writer.add_scalar(f'{prefix}val/loss', avg_val_loss, self.current_epoch)
            # Compute average component losses
            avg_contrastive = np.mean([d.get('contrastive', 0) for d in val_loss_dicts])
            self.writer.add_scalar(f'{prefix}val/contrastive_loss', avg_contrastive, self.current_epoch)

        return avg_val_loss, val_loss_dicts

    def save_checkpoint(self, filename, **kwargs):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
            **kwargs: Additional data to save
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            **kwargs
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")

        return checkpoint

    def freeze_backbone(self):
        """Freeze backbone (for Stage 3 fine-tuning)"""
        for param in self.model.image_encoder.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")

    def freeze_prototypes(self):
        """Freeze prototype layer (for Stage 3 fine-tuning)"""
        for param in self.model.image_encoder.prototype_layer.parameters():
            param.requires_grad = False
        print("✓ Prototypes frozen")

    def unfreeze_text_encoder(self):
        """Unfreeze text encoder (optional for Stage 3)"""
        self.model.text_encoder.unfreeze()
        print("✓ Text encoder unfrozen")

    def get_trainable_param_count(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_status(self):
        """Print training status"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.get_trainable_param_count()

        print(f"\nTraining Status:")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")

        if self.train_losses:
            print(f"  Latest train loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"  Latest val loss: {self.val_losses[-1]:.4f}")
