"""
Mask generation for pruning.

Supports unstructured, structured, and random masking strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal, Tuple
import numpy as np
from .importance import (
    compute_layer_importance,
    compute_global_threshold,
    magnitude_importance
)


class MaskGenerator:
    """
    Generates binary masks for model pruning.
    
    Supports:
    - Unstructured pruning (element-wise)
    - Structured pruning (channel/head pruning)
    - Random pruning
    """
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        method: Literal["unstructured", "structured", "random"] = "unstructured",
        importance_metric: str = "magnitude",
        structured_dim: Optional[int] = None,
    ):
        """
        Initialize mask generator.
        
        Args:
            model: PyTorch model to prune
            sparsity: Target sparsity level (0-1)
            method: Pruning method
            importance_metric: How to compute importance ("magnitude", "gradient", "taylor")
            structured_dim: Which dimension to prune for structured pruning (0 or 1)
        """
        self.model = model
        self.sparsity = sparsity
        self.method = method
        self.importance_metric = importance_metric
        self.structured_dim = structured_dim
        self.masks = {}
    
    def generate_masks(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        loss_fn: Optional[callable] = None,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate masks for all prunable layers.
        
        Args:
            dataloader: DataLoader for gradient-based importance
            loss_fn: Loss function for gradient computation
            device: Device to run on
        
        Returns:
            Dictionary mapping layer names to binary masks
        """
        if self.method == "unstructured":
            masks = compute_unstructured_mask(
                self.model,
                self.sparsity,
                dataloader=dataloader,
                loss_fn=loss_fn,
                importance_metric=self.importance_metric,
                device=device,
            )
        elif self.method == "structured":
            masks = compute_structured_mask(
                self.model,
                self.sparsity,
                dim=self.structured_dim or 0,
                dataloader=dataloader,
                loss_fn=loss_fn,
                importance_metric=self.importance_metric,
                device=device,
            )
        elif self.method == "random":
            masks = compute_random_mask(
                self.model,
                self.sparsity,
            )
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")
        
        self.masks = masks
        return masks


def compute_unstructured_mask(
    model: nn.Module,
    sparsity: float,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    loss_fn: Optional[callable] = None,
    importance_metric: str = "magnitude",
    device: str = "cpu",
    global_pruning: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute unstructured (element-wise) pruning masks.
    
    Args:
        model: PyTorch model
        sparsity: Target sparsity (0-1)
        dataloader: DataLoader for gradient-based importance
        loss_fn: Loss function
        importance_metric: Importance metric ("magnitude", "gradient", "taylor")
        device: Device to run on
        global_pruning: If True, prune globally across all layers. If False, prune per-layer.
    
    Returns:
        Dictionary of binary masks (1 = keep, 0 = prune)
    """
    masks = {}

    # Fast low-memory path for the common magnitude + layer-wise case.
    # This avoids materializing a full-model importance dictionary.
    if not global_pruning and importance_metric.lower() == "magnitude":
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, "weight") and module.weight is not None:
                    # Move one layer at a time to CPU for thresholding to cap peak memory.
                    importance = module.weight.detach().abs().float().cpu()
                    k = int(sparsity * importance.numel())
                    if k >= importance.numel():
                        mask = torch.zeros_like(importance, dtype=torch.bool)
                    elif k == 0:
                        mask = torch.ones_like(importance, dtype=torch.bool)
                    else:
                        threshold = torch.kthvalue(importance.flatten(), k + 1).values
                        mask = importance > threshold
                    masks[name] = mask
        return masks

    # Fallback path for global pruning and gradient/taylor metrics.
    importance_dict = compute_layer_importance(
        model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        method=importance_metric,
        device=device,
    )

    if global_pruning:
        # Global pruning: compute single threshold across all layers
        threshold = compute_global_threshold(importance_dict, sparsity)
        
        for name, importance in importance_dict.items():
            mask = importance > threshold
            masks[name] = mask
    else:
        # Layer-wise pruning: each layer gets same sparsity
        for name, importance in importance_dict.items():
            k = int(sparsity * importance.numel())
            if k >= importance.numel():
                mask = torch.zeros_like(importance, dtype=torch.bool)
            elif k == 0:
                mask = torch.ones_like(importance, dtype=torch.bool)
            else:
                threshold = torch.kthvalue(importance.flatten(), k + 1).values
                mask = importance > threshold
            masks[name] = mask
    
    return masks


def compute_structured_mask(
    model: nn.Module,
    sparsity: float,
    dim: int = 0,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    loss_fn: Optional[callable] = None,
    importance_metric: str = "magnitude",
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Compute structured pruning masks (prune entire channels/filters).
    
    Args:
        model: PyTorch model
        sparsity: Target sparsity (0-1)
        dim: Dimension to prune (0 for output channels, 1 for input channels)
        dataloader: DataLoader for gradient-based importance
        loss_fn: Loss function
        importance_metric: Importance metric
        device: Device to run on
    
    Returns:
        Dictionary of binary masks
    """
    # Compute importance scores
    importance_dict = compute_layer_importance(
        model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        method=importance_metric,
        device=device,
    )
    
    masks = {}
    
    for name, importance in importance_dict.items():
        # Compute channel-wise importance (sum across other dimensions)
        if dim == 0:
            # Prune output channels
            channel_importance = importance.sum(dim=list(range(1, importance.ndim)))
        else:
            # Prune input channels
            channel_importance = importance.sum(dim=[0] + list(range(2, importance.ndim)))
        
        # Determine which channels to keep
        num_channels = channel_importance.numel()
        num_keep = int(num_channels * (1 - sparsity))
        num_keep = max(1, num_keep)  # Keep at least one channel
        
        if num_keep >= num_channels:
            mask = torch.ones_like(importance, dtype=torch.bool)
        else:
            _, indices = torch.topk(channel_importance, num_keep)
            
            # Create mask
            mask = torch.zeros_like(importance, dtype=torch.bool)
            if dim == 0:
                mask[indices] = True
            else:
                mask[:, indices] = True
        
        masks[name] = mask
    
    return masks


def compute_random_mask(
    model: nn.Module,
    sparsity: float,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute random pruning masks.
    
    Args:
        model: PyTorch model
        sparsity: Target sparsity (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of binary masks
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    masks = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                mask = torch.rand_like(weight) > sparsity
                masks[name] = mask
    
    return masks


def apply_masks_to_model(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    inplace: bool = True,
) -> nn.Module:
    """
    Apply masks to model weights.
    
    Args:
        model: PyTorch model
        masks: Dictionary of masks
        inplace: If True, modify model in-place. Otherwise, create a copy.
    
    Returns:
        Masked model
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if name in masks:
            if hasattr(module, 'weight') and module.weight is not None:
                mask = masks[name].to(module.weight.device)
                module.weight.data *= mask
    
    return model
