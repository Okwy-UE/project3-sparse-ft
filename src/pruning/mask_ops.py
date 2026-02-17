"""
Mask operations: validation, statistics, I/O.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import json
import numpy as np
from pathlib import Path


def apply_mask(
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply binary mask to weight tensor.
    
    Args:
        weight: Weight tensor
        mask: Binary mask (same shape as weight)
    
    Returns:
        Masked weight
    """
    return weight * mask


def validate_mask(
    mask: torch.Tensor,
    expected_sparsity: Optional[float] = None,
    tolerance: float = 0.01,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that a mask is binary and has expected sparsity.
    
    Args:
        mask: Binary mask tensor
        expected_sparsity: Expected sparsity level (0-1)
        tolerance: Tolerance for sparsity check
    
    Returns:
        (is_valid, stats_dict)
    """
    stats = {}
    is_valid = True
    
    # Check that mask is binary
    unique_values = torch.unique(mask)
    is_binary = len(unique_values) <= 2 and all(v in [0.0, 1.0] for v in unique_values.tolist())
    stats['is_binary'] = is_binary
    
    if not is_binary:
        is_valid = False
        stats['error'] = f"Mask is not binary. Unique values: {unique_values.tolist()}"
    
    # Compute actual sparsity
    total_elements = mask.numel()
    zero_elements = (mask == 0).sum().item()
    actual_sparsity = zero_elements / total_elements
    stats['sparsity'] = actual_sparsity
    stats['num_zeros'] = zero_elements
    stats['num_ones'] = (mask == 1).sum().item()
    stats['total_elements'] = total_elements
    
    # Check sparsity if expected value is provided
    if expected_sparsity is not None:
        sparsity_error = abs(actual_sparsity - expected_sparsity)
        stats['sparsity_error'] = sparsity_error
        
        if sparsity_error > tolerance:
            is_valid = False
            stats['error'] = stats.get('error', '') + f" Sparsity {actual_sparsity:.4f} differs from expected {expected_sparsity:.4f} by {sparsity_error:.4f}"
    
    return is_valid, stats


def compute_sparsity_stats(
    masks: Dict[str, torch.Tensor],
    model: Optional[nn.Module] = None,
) -> Dict[str, any]:
    """
    Compute comprehensive sparsity statistics for a set of masks.
    
    Args:
        masks: Dictionary of layer masks
        model: Optional model to compute parameter stats
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'per_layer': {},
        'global': {},
    }
    
    total_zeros = 0
    total_elements = 0
    
    for name, mask in masks.items():
        layer_stats = {}
        
        num_zeros = (mask == 0).sum().item()
        num_elements = mask.numel()
        layer_sparsity = num_zeros / num_elements if num_elements > 0 else 0.0
        
        layer_stats['sparsity'] = layer_sparsity
        layer_stats['num_zeros'] = num_zeros
        layer_stats['num_elements'] = num_elements
        layer_stats['shape'] = list(mask.shape)
        
        stats['per_layer'][name] = layer_stats
        
        total_zeros += num_zeros
        total_elements += num_elements
    
    # Global statistics
    global_sparsity = total_zeros / total_elements if total_elements > 0 else 0.0
    stats['global']['sparsity'] = global_sparsity
    stats['global']['total_zeros'] = total_zeros
    stats['global']['total_elements'] = total_elements
    stats['global']['num_layers'] = len(masks)
    
    # Sparsity distribution
    sparsities = [s['sparsity'] for s in stats['per_layer'].values()]
    stats['global']['mean_layer_sparsity'] = np.mean(sparsities)
    stats['global']['std_layer_sparsity'] = np.std(sparsities)
    stats['global']['min_layer_sparsity'] = np.min(sparsities)
    stats['global']['max_layer_sparsity'] = np.max(sparsities)
    
    return stats


def save_mask(
    masks: Dict[str, torch.Tensor],
    save_path: str,
    metadata: Optional[Dict] = None,
    format: str = "pt",
) -> None:
    """
    Save masks to disk.
    
    Args:
        masks: Dictionary of masks
        save_path: Path to save masks
        metadata: Optional metadata to save with masks
        format: Save format ("pt" for PyTorch, "npz" for NumPy, "safetensors" for Safetensors)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pt":
        # PyTorch format
        save_dict = {
            'masks': masks,
            'metadata': metadata or {},
        }
        torch.save(save_dict, save_path)
    
    elif format == "npz":
        # NumPy format
        np_masks = {k: v.cpu().numpy() for k, v in masks.items()}
        np.savez(save_path, **np_masks)
        
        # Save metadata separately
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    elif format == "safetensors":
        try:
            from safetensors.torch import save_file
            # Safetensors doesn't support nested dicts, so flatten
            flat_masks = {k: v for k, v in masks.items()}
            save_file(flat_masks, save_path, metadata=metadata)
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_mask(
    load_path: str,
    format: str = "pt",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Load masks from disk.
    
    Args:
        load_path: Path to load masks from
        format: Load format ("pt", "npz", "safetensors")
    
    Returns:
        (masks, metadata)
    """
    load_path = Path(load_path)
    
    if format == "pt":
        data = torch.load(load_path, map_location='cpu')
        masks = data['masks']
        metadata = data.get('metadata', {})
    
    elif format == "npz":
        data = np.load(load_path)
        masks = {k: torch.from_numpy(data[k]) for k in data.files}
        
        # Load metadata if available
        metadata_path = load_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
    
    elif format == "safetensors":
        try:
            from safetensors.torch import load_file
            masks = load_file(load_path)
            # Metadata is embedded in safetensors
            metadata = {}
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return masks, metadata


def generate_mask_checksum(mask: torch.Tensor) -> str:
    """
    Generate a checksum for a mask to verify integrity.
    
    Args:
        mask: Binary mask
    
    Returns:
        Checksum string
    """
    # Use MD5 hash of mask bytes
    import hashlib
    mask_bytes = mask.cpu().numpy().tobytes()
    checksum = hashlib.md5(mask_bytes).hexdigest()
    return checksum


def verify_mask_checksum(
    mask: torch.Tensor,
    expected_checksum: str,
) -> bool:
    """
    Verify mask checksum.
    
    Args:
        mask: Binary mask
        expected_checksum: Expected checksum
    
    Returns:
        True if checksums match
    """
    actual_checksum = generate_mask_checksum(mask)
    return actual_checksum == expected_checksum


def plot_sparsity_histogram(
    masks: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-layer sparsity histogram.
    
    Args:
        masks: Dictionary of masks
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping histogram plot.")
        return
    
    stats = compute_sparsity_stats(masks)
    layer_names = list(stats['per_layer'].keys())
    sparsities = [stats['per_layer'][name]['sparsity'] for name in layer_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layer_names)), sparsities)
    plt.xlabel('Layer')
    plt.ylabel('Sparsity')
    plt.title('Per-Layer Sparsity Distribution')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
