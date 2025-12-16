"""
Checkpoint converter to load Proto-CLIP pretrained weights into ProtoPNet-CLIP model.

Based on inspection results:
- memory_bank_v.pt: (16000, 1024) visual prototypes → subsample to (M, 1024)
- memory_bank_t.pt: (1000, 1024) text prototypes → optional reference
- query_adapter.pt: Spatial conv adapter → not compatible, skip
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


class ProtoClipCheckpointConverter:
    """
    Convert Proto-CLIP checkpoints to ProtoPNet-CLIP format.

    Args:
        strict: If True, raise error on dimension mismatch. If False, adapt gracefully.
        sampling_method: How to subsample prototypes ('random', 'kmeans', 'first')
        verbose: Print detailed loading information
    """

    def __init__(self, strict=False, sampling_method='kmeans', verbose=True):
        self.strict = strict
        self.sampling_method = sampling_method
        self.verbose = verbose

    def load_visual_memory(self, checkpoint_path):
        """
        Load visual memory bank (prototypes).

        Args:
            checkpoint_path: Path to memory_bank_v.pt

        Returns:
            torch.Tensor: Visual prototypes (M, D)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if self.verbose:
            print(f"Loading visual memory from: {checkpoint_path}")

        # Load checkpoint (it's a Parameter object)
        visual_memory = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Convert from Parameter to Tensor if needed
        if isinstance(visual_memory, nn.Parameter):
            visual_memory = visual_memory.data

        # Convert to float32 (from float16)
        visual_memory = visual_memory.float()

        if self.verbose:
            print(f"  Loaded shape: {visual_memory.shape}")
            print(f"  Dtype: {visual_memory.dtype}")
            print(f"  Stats: mean={visual_memory.mean():.4f}, std={visual_memory.std():.4f}")

        return visual_memory

    def load_text_memory(self, checkpoint_path):
        """
        Load text memory bank (optional).

        Args:
            checkpoint_path: Path to memory_bank_t.pt

        Returns:
            torch.Tensor: Text prototypes (N, D)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            if self.verbose:
                print(f"Text memory not found: {checkpoint_path}, skipping")
            return None

        if self.verbose:
            print(f"Loading text memory from: {checkpoint_path}")

        text_memory = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(text_memory, nn.Parameter):
            text_memory = text_memory.data

        text_memory = text_memory.float()

        if self.verbose:
            print(f"  Loaded shape: {text_memory.shape}")

        return text_memory

    def subsample_prototypes(self, prototypes, target_count):
        """
        Subsample prototypes from (M_loaded, D) to (target_count, D).

        Args:
            prototypes: Original prototypes (M_loaded, D)
            target_count: Desired number of prototypes (M)

        Returns:
            torch.Tensor: Subsampled prototypes (M, D)
        """
        M_loaded, D = prototypes.shape

        if M_loaded == target_count:
            if self.verbose:
                print(f"  Prototype count matches ({target_count}), no subsampling needed")
            return prototypes

        if self.verbose:
            print(f"  Subsampling {M_loaded} → {target_count} prototypes using '{self.sampling_method}'")

        if self.sampling_method == 'random':
            # Random sampling
            indices = torch.randperm(M_loaded)[:target_count]
            subsampled = prototypes[indices]

        elif self.sampling_method == 'first':
            # Take first N prototypes
            subsampled = prototypes[:target_count]

        elif self.sampling_method == 'kmeans':
            # K-means clustering to get representative prototypes
            if self.verbose:
                print(f"  Running K-means clustering (k={target_count})...")

            # Use MiniBatchKMeans for efficiency with large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=target_count,
                batch_size=1000,
                random_state=42,
                verbose=0
            )

            # Fit K-means
            prototypes_np = prototypes.numpy()
            kmeans.fit(prototypes_np)

            # Use cluster centers as prototypes
            subsampled = torch.from_numpy(kmeans.cluster_centers_).float()

            if self.verbose:
                print(f"  K-means complete, using cluster centers as prototypes")

        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        if self.verbose:
            print(f"  Subsampled shape: {subsampled.shape}")

        return subsampled

    def adapt_prototypes(self, prototypes, target_count, target_dim):
        """
        Adapt prototypes to match target dimensions.

        Args:
            prototypes: Loaded prototypes (M_loaded, D_loaded)
            target_count: Target number of prototypes (M)
            target_dim: Target feature dimension (D)

        Returns:
            torch.Tensor: Adapted prototypes (M, D)
        """
        M_loaded, D_loaded = prototypes.shape

        # Handle prototype count mismatch
        if M_loaded != target_count:
            prototypes = self.subsample_prototypes(prototypes, target_count)

        # Handle feature dimension mismatch
        if D_loaded != target_dim:
            if self.strict:
                raise ValueError(
                    f"Feature dimension mismatch: loaded {D_loaded}, expected {target_dim}"
                )
            else:
                if self.verbose:
                    print(f"  WARNING: Feature dimension mismatch ({D_loaded} vs {target_dim})")

                if D_loaded < target_dim:
                    # Pad with zeros
                    padding = torch.zeros(target_count, target_dim - D_loaded)
                    prototypes = torch.cat([prototypes, padding], dim=1)
                    if self.verbose:
                        print(f"  Padded to {target_dim} dimensions")
                else:
                    # Truncate
                    prototypes = prototypes[:, :target_dim]
                    if self.verbose:
                        print(f"  Truncated to {target_dim} dimensions")

        return prototypes

    def initialize_model_from_protoclip(
        self,
        model,
        visual_memory_path,
        text_memory_path=None,
        adapter_path=None
    ):
        """
        Initialize ProtoPNet-CLIP model with Proto-CLIP pretrained weights.

        Args:
            model: ProtoCLIP model instance
            visual_memory_path: Path to memory_bank_v.pt
            text_memory_path: Path to memory_bank_t.pt (optional)
            adapter_path: Path to query_adapter.pt (currently not used - incompatible)

        Returns:
            dict: Loading statistics
        """
        stats = {
            'visual_memory_loaded': False,
            'text_memory_loaded': False,
            'adapter_loaded': False,
            'original_prototype_count': None,
            'loaded_prototype_count': None
        }

        print("=" * 70)
        print("Loading Proto-CLIP Pretrained Weights")
        print("=" * 70)

        # Get model's expected dimensions
        target_M = model.image_encoder.num_prototypes
        target_D = model.image_encoder.feature_dim

        if self.verbose:
            print(f"\nTarget model configuration:")
            print(f"  Prototype count (M): {target_M}")
            print(f"  Feature dimension (D): {target_D}")

        # Load visual memory
        try:
            print(f"\n[1/3] Loading Visual Memory Bank")
            visual_memory = self.load_visual_memory(visual_memory_path)
            stats['original_prototype_count'] = visual_memory.shape[0]

            # Adapt to model dimensions
            adapted_prototypes = self.adapt_prototypes(
                visual_memory,
                target_count=target_M,
                target_dim=target_D
            )
            stats['loaded_prototype_count'] = adapted_prototypes.shape[0]

            # Load into model
            model.image_encoder.set_prototypes(adapted_prototypes)
            stats['visual_memory_loaded'] = True

            print(f"  ✓ Visual prototypes loaded successfully!")
            print(f"  Initialized {target_M} prototypes from {stats['original_prototype_count']} Proto-CLIP prototypes")

        except Exception as e:
            print(f"  ✗ Error loading visual memory: {e}")
            if self.strict:
                raise

        # Load text memory (optional - for reference only)
        if text_memory_path:
            try:
                print(f"\n[2/3] Loading Text Memory Bank (optional)")
                text_memory = self.load_text_memory(text_memory_path)
                if text_memory is not None:
                    stats['text_memory_loaded'] = True
                    print(f"  ✓ Text memory loaded (shape: {text_memory.shape})")
                    print(f"  Note: Text memory is for reference, not loaded into model")
                else:
                    print(f"  ⊘ Text memory skipped")
            except Exception as e:
                print(f"  ⊘ Text memory not loaded: {e}")

        # Adapter (skip - incompatible architecture)
        if adapter_path:
            print(f"\n[3/3] Query Adapter")
            print(f"  ⊘ Skipping adapter (incompatible architecture)")
            print(f"  Adapter is spatial conv network, our projection head expects vectors")
            print(f"  Using random initialization for projection head")
        else:
            print(f"\n[3/3] Query Adapter: Not provided")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Visual prototypes: {'✓ Loaded' if stats['visual_memory_loaded'] else '✗ Failed'}")
        print(f"Text prototypes: {'✓ Loaded' if stats['text_memory_loaded'] else '⊘ Skipped'}")
        print(f"Adapter: ⊘ Skipped (incompatible)")

        if stats['visual_memory_loaded']:
            print(f"\n✓ Model successfully initialized with Proto-CLIP pretrained weights!")
            print(f"  - {stats['loaded_prototype_count']} prototypes loaded")
            print(f"  - Feature dimension: {target_D}")
            print(f"  - Sampling method: {self.sampling_method}")
        else:
            print(f"\n⚠ WARNING: Visual memory not loaded, model using random initialization")

        print("=" * 70)

        return stats


def quick_load_protoclip(model, checkpoint_dir, sampling_method='kmeans', verbose=True):
    """
    Convenience function to quickly load Proto-CLIP weights.

    Args:
        model: ProtoCLIP model instance
        checkpoint_dir: Directory containing Proto-CLIP checkpoints
        sampling_method: How to subsample prototypes ('random', 'kmeans', 'first')
        verbose: Print loading information

    Returns:
        dict: Loading statistics
    """
    checkpoint_dir = Path(checkpoint_dir)

    converter = ProtoClipCheckpointConverter(
        strict=False,
        sampling_method=sampling_method,
        verbose=verbose
    )

    stats = converter.initialize_model_from_protoclip(
        model,
        visual_memory_path=checkpoint_dir / "memory_bank_v.pt",
        text_memory_path=checkpoint_dir / "memory_bank_t.pt",
        adapter_path=checkpoint_dir / "query_adapter.pt"
    )

    return stats


if __name__ == '__main__':
    # Test loading
    print("Checkpoint Converter Test")
    print("This requires a ProtoCLIP model instance to run")
    print("Usage: See example in test script or model initialization")
