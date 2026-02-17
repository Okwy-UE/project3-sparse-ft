"""
Importance scoring functions for pruning.

Implements various importance metrics used to determine which weights to prune.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
import numpy as np


def magnitude_importance(
    module: nn.Module,
    name: str = "",
    **kwargs
) -> torch.Tensor:
    """
    Compute importance scores based on weight magnitudes.
    
    Args:
        module: PyTorch module containing weights
        name: Name of the module (for logging)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tensor of importance scores (same shape as module weight)
    """
    if not hasattr(module, 'weight') or module.weight is None:
        return torch.zeros(1)
    
    weight = module.weight.data
    importance = torch.abs(weight)
    
    return importance


def gradient_importance(
    module: nn.Module,
    name: str = "",
    accumulate_steps: int = 100,
    **kwargs
) -> torch.Tensor:
    """
    Compute importance scores based on gradient magnitudes.
    
    Args:
        module: PyTorch module containing weights
        name: Name of the module
        accumulate_steps: Number of steps to accumulate gradients (not used here,
                         assumed gradients are already accumulated)
        **kwargs: Additional arguments
    
    Returns:
        Tensor of importance scores
    """
    if not hasattr(module, 'weight') or module.weight is None:
        return torch.zeros(1)
    
    weight = module.weight
    if weight.grad is None:
        # Fall back to magnitude if no gradient
        return magnitude_importance(module, name)
    
    importance = torch.abs(weight.grad)
    
    return importance


def taylor_importance(
    module: nn.Module,
    name: str = "",
    **kwargs
) -> torch.Tensor:
    """
    Compute importance scores based on first-order Taylor expansion.
    
    Importance = |weight * gradient|
    
    This approximates the change in loss when removing a weight.
    
    Args:
        module: PyTorch module containing weights
        name: Name of the module
        **kwargs: Additional arguments
    
    Returns:
        Tensor of importance scores
    """
    if not hasattr(module, 'weight') or module.weight is None:
        return torch.zeros(1)
    
    weight = module.weight
    if weight.grad is None:
        # Fall back to magnitude if no gradient
        return magnitude_importance(module, name)
    
    importance = torch.abs(weight.data * weight.grad)
    
    return importance


def compute_layer_importance(
    model: nn.Module,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    loss_fn: Optional[Callable] = None,
    method: str = "magnitude",
    device: str = "cpu",
    num_batches: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Compute importance scores for all layers in a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for computing gradient-based importance
        loss_fn: Loss function for computing gradients
        method: Importance metric ("magnitude", "gradient", "taylor")
        device: Device to run on
        num_batches: Number of batches to use for gradient estimation
    
    Returns:
        Dictionary mapping layer names to importance tensors
    """
    importance_dict = {}
    
    # Map method name to function
    method_fn = {
        "magnitude": magnitude_importance,
        "gradient": gradient_importance,
        "taylor": taylor_importance,
    }.get(method.lower(), magnitude_importance)
    
    if method in ["gradient", "taylor"] and dataloader is not None:
        # Accumulate gradients over multiple batches
        model.eval()
        model.zero_grad()
        
        num_processed = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                outputs = model(**batch)
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                outputs = model(*batch)
            else:
                batch = batch.to(device)
                outputs = model(batch)
            
            # Compute loss
            if loss_fn is not None:
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    loss = loss_fn(outputs, batch)
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                raise ValueError("No loss function provided and model output has no 'loss' key")
            
            loss = loss / num_batches  # Normalize
            loss.backward()
            num_processed += 1
        
        if num_processed == 0:
            print(f"Warning: No batches processed for gradient computation, falling back to magnitude")
            method_fn = magnitude_importance
    
    # Compute importance for each layer
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                importance = method_fn(module, name)
                importance_dict[name] = importance.cpu()
    
    return importance_dict


# Utility function for global importance ranking
def compute_global_threshold(
    importance_dict: Dict[str, torch.Tensor],
    sparsity: float,
) -> float:
    """
    Compute global importance threshold for a target sparsity level.
    
    Args:
        importance_dict: Dictionary of importance scores per layer
        sparsity: Target sparsity (fraction of weights to prune, 0-1)
    
    Returns:
        Threshold value: weights with importance below this should be pruned
    """
    # Concatenate all importance scores
    all_importance = torch.cat([
        scores.flatten() for scores in importance_dict.values()
    ])
    
    # Compute threshold
    k = int(sparsity * all_importance.numel())
    if k == 0:
        return float('inf')
    if k >= all_importance.numel():
        return float('-inf')
    
    threshold = torch.kthvalue(all_importance, k + 1).values.item()
    
    return threshold
