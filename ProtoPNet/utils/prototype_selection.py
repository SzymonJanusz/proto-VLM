"""
Prototype Selection Utilities

Functions for selecting top-k most activated prototypes from spatial similarity maps.
Implements sparse activation by zeroing out less relevant prototypes.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def select_top_k_prototypes(
    spatial_similarities: torch.Tensor,
    k: int = 1,
    return_indices: bool = False,
    return_max_activations: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Select top-k most activated prototypes per image, zero out the rest.

    Creates sparse spatial similarities by keeping only the k prototypes with
    highest max activation, and zeroing out all other prototypes.

    Args:
        spatial_similarities: (B, M, H, W) - Full spatial prototype similarities
            B: Batch size
            M: Number of prototypes (typically 200)
            H, W: Spatial dimensions (typically 14, 14)
        k: Number of prototypes to keep per image (default: 1)
            k=1: Only keep most activated prototype (sparsest)
            k=5: Keep top-5 prototypes
            k=M: Keep all prototypes (no sparsity)
        return_indices: If True, also return indices of selected prototypes
        return_max_activations: If True, also return max activation values

    Returns:
        sparse_similarities: (B, M, H, W) - Sparse spatial similarities
            Same shape as input, but only top-k prototypes have non-zero values
        indices: (B, k) - Indices of selected prototypes (if return_indices=True)
        max_activations: (B, M) - Max activation per prototype (if return_max_activations=True)

    Algorithm:
        1. Compute max activation per prototype: max over spatial dimensions (H, W)
        2. Find top-k prototypes by max activation using torch.topk()
        3. Create binary mask: 1 for top-k prototypes, 0 for rest
        4. Apply mask to spatial similarities element-wise

    Example:
        >>> # Select only most activated prototype per image
        >>> spatial_sims = torch.randn(32, 200, 14, 14)  # 32 images, 200 prototypes
        >>> sparse_sims = select_top_k_prototypes(spatial_sims, k=1)
        >>> assert sparse_sims.shape == (32, 200, 14, 14)
        >>> # Only 1 out of 200 prototypes will have non-zero values

        >>> # Select top-5 prototypes with indices
        >>> sparse_sims, indices = select_top_k_prototypes(
        ...     spatial_sims, k=5, return_indices=True
        ... )
        >>> assert indices.shape == (32, 5)  # Indices of top-5 prototypes per image
    """
    B, M, H, W = spatial_similarities.shape

    # Input validation
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    if k > M:
        raise ValueError(
            f"k ({k}) cannot be greater than number of prototypes ({M}). "
            f"Use k={M} to keep all prototypes (no sparsity)."
        )

    # Step 1: Compute max activation per prototype
    # Flatten spatial dimensions and take max
    max_activations = spatial_similarities.view(B, M, -1).max(dim=2)[0]  # (B, M)

    # Step 2: Find top-k prototypes by max activation
    top_k_values, top_k_indices = torch.topk(max_activations, k=k, dim=1)  # (B, k), (B, k)

    # Step 3: Create binary mask for top-k prototypes
    # Initialize mask with zeros
    mask = torch.zeros_like(spatial_similarities)  # (B, M, H, W)

    # Set mask to 1 for top-k prototypes
    # Expand indices to match spatial dimensions
    batch_indices = torch.arange(B, device=spatial_similarities.device)[:, None]  # (B, 1)

    # Use advanced indexing to set mask for selected prototypes
    mask[batch_indices, top_k_indices] = 1.0  # (B, k, H, W) locations set to 1

    # Step 4: Apply mask (element-wise multiplication)
    sparse_similarities = spatial_similarities * mask  # (B, M, H, W)

    # Prepare return values based on flags
    returns = [sparse_similarities]

    if return_indices:
        returns.append(top_k_indices)  # (B, k)

    if return_max_activations:
        returns.append(max_activations)  # (B, M)

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)


def visualize_selected_prototypes(
    spatial_similarities: torch.Tensor,
    sparse_similarities: torch.Tensor,
    image_idx: int = 0,
    top_k_indices: Optional[torch.Tensor] = None
) -> dict:
    """
    Helper function to visualize which prototypes were selected for a single image.

    Args:
        spatial_similarities: (B, M, H, W) - Original full similarities
        sparse_similarities: (B, M, H, W) - Sparse similarities after selection
        image_idx: Index of image to visualize (default: 0)
        top_k_indices: (B, k) - Indices of selected prototypes (optional)

    Returns:
        info: dict with visualization information:
            - 'selected_prototypes': List of prototype indices that are non-zero
            - 'num_selected': Number of selected prototypes (should equal k)
            - 'sparsity_ratio': Fraction of prototypes that are zero
            - 'max_activations_full': Max activation per prototype (before sparsification)
            - 'max_activations_sparse': Max activation per prototype (after sparsification)

    Example:
        >>> spatial_sims = torch.randn(32, 200, 14, 14)
        >>> sparse_sims, indices = select_top_k_prototypes(
        ...     spatial_sims, k=3, return_indices=True
        ... )
        >>> info = visualize_selected_prototypes(
        ...     spatial_sims, sparse_sims, image_idx=0, top_k_indices=indices
        ... )
        >>> print(f"Selected prototypes: {info['selected_prototypes']}")
        >>> print(f"Sparsity: {info['sparsity_ratio']:.2%}")
    """
    B, M, H, W = spatial_similarities.shape

    # Get data for specific image
    full_sim = spatial_similarities[image_idx]  # (M, H, W)
    sparse_sim = sparse_similarities[image_idx]  # (M, H, W)

    # Find which prototypes are non-zero in sparse version
    # A prototype is "selected" if it has any non-zero values
    prototype_norms = sparse_sim.view(M, -1).abs().sum(dim=1)  # (M,)
    selected_mask = prototype_norms > 1e-8  # Tolerance for floating point
    selected_prototypes = torch.where(selected_mask)[0].tolist()

    # Compute statistics
    num_selected = len(selected_prototypes)
    sparsity_ratio = (M - num_selected) / M

    # Compute max activations
    max_act_full = full_sim.view(M, -1).max(dim=1)[0]  # (M,)
    max_act_sparse = sparse_sim.view(M, -1).max(dim=1)[0]  # (M,)

    info = {
        'selected_prototypes': selected_prototypes,
        'num_selected': num_selected,
        'sparsity_ratio': sparsity_ratio,
        'max_activations_full': max_act_full.cpu().numpy(),
        'max_activations_sparse': max_act_sparse.cpu().numpy(),
    }

    # Add top-k indices if provided
    if top_k_indices is not None:
        info['top_k_indices'] = top_k_indices[image_idx].cpu().tolist()

    return info


def compute_sparsity_statistics(
    spatial_similarities: torch.Tensor,
    sparse_similarities: torch.Tensor
) -> dict:
    """
    Compute statistics about sparsification across entire batch.

    Args:
        spatial_similarities: (B, M, H, W) - Original full similarities
        sparse_similarities: (B, M, H, W) - Sparse similarities after selection

    Returns:
        stats: dict with keys:
            - 'mean_sparsity': Average fraction of zero prototypes per image
            - 'std_sparsity': Standard deviation of sparsity
            - 'mean_num_selected': Average number of selected prototypes per image
            - 'total_parameters_full': Total number of values in full similarities
            - 'total_parameters_sparse': Total number of non-zero values in sparse
            - 'compression_ratio': Ratio of sparse to full parameters

    Example:
        >>> spatial_sims = torch.randn(32, 200, 14, 14)
        >>> sparse_sims = select_top_k_prototypes(spatial_sims, k=5)
        >>> stats = compute_sparsity_statistics(spatial_sims, sparse_sims)
        >>> print(f"Average sparsity: {stats['mean_sparsity']:.2%}")
        >>> print(f"Compression: {stats['compression_ratio']:.1f}x")
    """
    B, M, H, W = spatial_similarities.shape

    # Find selected prototypes per image
    # Reshape to (B, M, H*W), check if any spatial location is non-zero
    sparse_reshaped = sparse_similarities.view(B, M, -1)  # (B, M, H*W)
    has_activation = (sparse_reshaped.abs().sum(dim=2) > 1e-8)  # (B, M), bool

    # Count selected prototypes per image
    num_selected_per_image = has_activation.sum(dim=1).float()  # (B,)

    # Compute sparsity per image (fraction of zero prototypes)
    sparsity_per_image = (M - num_selected_per_image) / M  # (B,)

    # Count total non-zero elements
    total_elements_full = B * M * H * W
    total_nonzero_sparse = (sparse_similarities.abs() > 1e-8).sum().item()

    stats = {
        'mean_sparsity': sparsity_per_image.mean().item(),
        'std_sparsity': sparsity_per_image.std().item(),
        'mean_num_selected': num_selected_per_image.mean().item(),
        'total_parameters_full': total_elements_full,
        'total_parameters_sparse': total_nonzero_sparse,
        'compression_ratio': total_elements_full / max(total_nonzero_sparse, 1),
    }

    return stats


# =============================================================================
# Future Selection Algorithms (Advanced)
# =============================================================================
# These are placeholder ideas for more sophisticated selection strategies
# beyond simple top-k by max activation
# =============================================================================


def select_diverse_prototypes(
    spatial_similarities: torch.Tensor,
    k: int = 5,
    diversity_weight: float = 0.5
) -> torch.Tensor:
    """
    PLACEHOLDER: Select k prototypes balancing activation strength AND diversity.

    Instead of just selecting top-k by activation, this method aims to select
    prototypes that are both highly activated AND semantically diverse (not correlated).

    Algorithm (to be implemented):
        1. Start with top-1 prototype (highest activation)
        2. For remaining k-1 slots:
            - Score = α * activation + β * diversity_from_selected
            - diversity = minimum cosine distance to already selected prototypes
            - Select prototype with highest score
            - Add to selected set

    This encourages selecting prototypes that:
        - Are highly activated (important for the image)
        - Capture different visual concepts (not redundant)

    Args:
        spatial_similarities: (B, M, H, W)
        k: Number of prototypes to select
        diversity_weight: Trade-off between activation (0) and diversity (1)

    Returns:
        sparse_similarities: (B, M, H, W)
    """
    raise NotImplementedError(
        "Diverse prototype selection not yet implemented. "
        "This would require: "
        "1. Access to prototype vectors for computing semantic similarity "
        "2. Iterative selection algorithm (not vectorized) "
        "3. Balance between activation strength and prototype diversity"
    )


def select_prototypes_by_threshold(
    spatial_similarities: torch.Tensor,
    threshold: float = 0.5,
    min_prototypes: int = 1,
    max_prototypes: int = 20
) -> torch.Tensor:
    """
    PLACEHOLDER: Select prototypes whose max activation exceeds a threshold.

    Instead of fixed k, this allows adaptive number of prototypes based on
    activation strength. Images with many strong prototype activations will
    keep more prototypes; images with few will keep fewer.

    Algorithm (to be implemented):
        1. Compute max activation per prototype
        2. Select all prototypes with max activation > threshold
        3. Clip to [min_prototypes, max_prototypes] range

    This allows:
        - Variable sparsity per image (adaptive to image complexity)
        - More interpretable selection (threshold has semantic meaning)

    Args:
        spatial_similarities: (B, M, H, W)
        threshold: Activation threshold (e.g., 0.5)
        min_prototypes: Minimum number to keep (default: 1)
        max_prototypes: Maximum number to keep (default: 20)

    Returns:
        sparse_similarities: (B, M, H, W)
    """
    raise NotImplementedError(
        "Threshold-based selection not yet implemented. "
        "This would require: "
        "1. Per-image variable sparsity (different k per image) "
        "2. Batched operations with variable-size selections "
        "3. Handling edge cases (no prototypes above threshold)"
    )


def select_prototypes_by_class(
    spatial_similarities: torch.Tensor,
    prototype_class_assignments: torch.Tensor,
    target_classes: torch.Tensor,
    k_per_class: int = 2
) -> torch.Tensor:
    """
    PLACEHOLDER: Select top-k prototypes from each relevant class.

    If prototypes are class-specific (e.g., 10 prototypes per class in ProtoPNet),
    this selects the top-k from each class that appears in the image.

    Algorithm (to be implemented):
        1. For each class c in target_classes:
            - Find prototypes belonging to class c
            - Select top-k of these prototypes by activation
        2. Combine selected prototypes across all classes

    This enforces diversity by ensuring multiple classes are represented.

    Args:
        spatial_similarities: (B, M, H, W)
        prototype_class_assignments: (M,) - Class ID for each prototype
        target_classes: (B, num_classes) - Classes present in each image
        k_per_class: Number of prototypes to select per class

    Returns:
        sparse_similarities: (B, M, H, W)
    """
    raise NotImplementedError(
        "Class-based selection not yet implemented. "
        "This would require: "
        "1. Prototype-to-class mapping from ProtoPNet training "
        "2. Multi-class selection logic (handle images with multiple classes) "
        "3. Aggregation when multiple classes selected"
    )
