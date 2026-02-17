#!/usr/bin/env python3
"""
Compute pruning masks for a pretrained model.

Usage:
    python compute_masks.py \
        --model_path /path/to/model.pt \
        --output_dir ./masks \
        --sparsity 0.5 \
        --method unstructured \
        --importance magnitude
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pruning.mask_generator import MaskGenerator, compute_unstructured_mask, compute_structured_mask, compute_random_mask
from pruning.mask_ops import save_mask, compute_sparsity_stats, validate_mask, plot_sparsity_histogram


def load_model(model_path: str, model_type: str = "auto"):
    """Load a pretrained model."""
    print(f"Loading model from {model_path}")
    
    # Try to load as PyTorch state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"Loaded state dict with {len(state_dict)} keys")
        return state_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_dummy_model_from_state_dict(state_dict: dict) -> nn.Module:
    """
    Create a dummy model structure from a state dict.
    
    This is a simplified version - you may need to adapt based on your actual model.
    """
    class DummyModel(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            # Create modules based on state dict keys
            for key in state_dict.keys():
                if '.weight' in key:
                    module_name = key.replace('.weight', '')
                    weight = state_dict[key]
                    
                    if len(weight.shape) == 2:
                        # Linear layer
                        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                        layer.weight.data = weight
                        
                        # Set as attribute (replace dots with underscores)
                        setattr(self, module_name.replace('.', '_'), layer)
        
        def forward(self, x):
            # Dummy forward
            return x
    
    return DummyModel(state_dict)


def compute_masks_for_state_dict(
    state_dict: dict,
    sparsity: float,
    method: str,
    importance_metric: str,
    seed: int = 42,
) -> dict:
    """
    Compute masks directly from a state dict.
    
    Args:
        state_dict: Model state dict
        sparsity: Target sparsity (0-1)
        method: Pruning method ("unstructured", "structured", "random")
        importance_metric: Importance metric ("magnitude", "gradient", "taylor")
        seed: Random seed
    
    Returns:
        Dictionary of masks
    """
    masks = {}
    
    for key in state_dict.keys():
        if '.weight' in key and 'norm' not in key.lower():
            weight = state_dict[key]
            
            # Skip 1D weights (usually layer norms, embeddings)
            if len(weight.shape) < 2:
                continue
            
            module_name = key.replace('.weight', '')
            
            if method == "unstructured":
                # Magnitude-based unstructured pruning
                importance = torch.abs(weight)
                k = int(sparsity * importance.numel())
                if k >= importance.numel():
                    mask = torch.zeros_like(weight)
                elif k == 0:
                    mask = torch.ones_like(weight)
                else:
                    threshold = torch.kthvalue(importance.flatten(), k + 1).values
                    mask = (importance > threshold).float()
            
            elif method == "structured":
                # Channel-wise structured pruning
                importance = torch.abs(weight).sum(dim=1)  # Sum across input channels
                num_channels = importance.numel()
                num_keep = int(num_channels * (1 - sparsity))
                num_keep = max(1, num_keep)
                
                if num_keep >= num_channels:
                    mask = torch.ones_like(weight)
                else:
                    _, indices = torch.topk(importance, num_keep)
                    mask = torch.zeros_like(weight)
                    mask[indices] = 1.0
            
            elif method == "random":
                torch.manual_seed(seed)
                mask = (torch.rand_like(weight) > sparsity).float()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            masks[module_name] = mask
    
    return masks


def main():
    parser = argparse.ArgumentParser(description="Compute pruning masks")
    
    # Input/output
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save masks")
    parser.add_argument("--model_type", type=str, default="auto", help="Model type (auto, llama, mistral, etc.)")
    
    # Pruning configuration
    parser.add_argument("--sparsity", type=float, nargs="+", default=[0.5],
                       help="Target sparsity levels (can specify multiple)")
    parser.add_argument("--method", type=str, default="unstructured",
                       choices=["unstructured", "structured", "random"],
                       help="Pruning method")
    parser.add_argument("--importance", type=str, default="magnitude",
                       choices=["magnitude", "gradient", "taylor"],
                       help="Importance metric (only for gradient-based methods)")
    
    # Options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "npz", "safetensors"],
                       help="Save format")
    parser.add_argument("--plot", action="store_true", help="Plot sparsity histograms")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load model
    state_dict = load_model(args.model_path, args.model_type)
    if state_dict is None:
        print("Failed to load model")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute masks for each sparsity level
    for sparsity in args.sparsity:
        print(f"\n{'='*60}")
        print(f"Computing masks for sparsity = {sparsity:.2%}")
        print(f"{'='*60}")
        
        # Compute masks
        masks = compute_masks_for_state_dict(
            state_dict,
            sparsity=sparsity,
            method=args.method,
            importance_metric=args.importance,
            seed=args.seed,
        )
        
        # Validate masks
        print(f"\nValidating masks...")
        all_valid = True
        for name, mask in masks.items():
            is_valid, stats = validate_mask(mask, expected_sparsity=sparsity, tolerance=0.05)
            if not is_valid:
                print(f"  ⚠️  {name}: {stats.get('error', 'Invalid')}")
                all_valid = False
        
        if all_valid:
            print("  ✓ All masks valid")
        
        # Compute stats
        stats = compute_sparsity_stats(masks)
        print(f"\nSparsity statistics:")
        print(f"  Global sparsity: {stats['global']['sparsity']:.2%}")
        print(f"  Mean layer sparsity: {stats['global']['mean_layer_sparsity']:.2%} ± {stats['global']['std_layer_sparsity']:.2%}")
        print(f"  Min/Max layer sparsity: {stats['global']['min_layer_sparsity']:.2%} / {stats['global']['max_layer_sparsity']:.2%}")
        print(f"  Total zeros: {stats['global']['total_zeros']:,} / {stats['global']['total_elements']:,}")
        print(f"  Number of layers: {stats['global']['num_layers']}")
        
        # Save masks
        mask_filename = f"masks_sparsity{int(sparsity*100)}_{args.method}.{args.format}"
        mask_path = output_dir / mask_filename
        
        metadata = {
            "sparsity": sparsity,
            "method": args.method,
            "importance": args.importance,
            "seed": args.seed,
            "model_path": str(args.model_path),
            "created_at": datetime.now().isoformat(),
            "stats": stats,
        }
        
        save_mask(masks, str(mask_path), metadata=metadata, format=args.format)
        print(f"\n✓ Masks saved to: {mask_path}")
        
        # Save stats as JSON
        stats_path = output_dir / f"stats_sparsity{int(sparsity*100)}_{args.method}.json"
        with open(stats_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Statistics saved to: {stats_path}")
        
        # Plot histogram
        if args.plot:
            plot_path = output_dir / f"histogram_sparsity{int(sparsity*100)}_{args.method}.png"
            plot_sparsity_histogram(masks, save_path=str(plot_path))
            print(f"✓ Histogram saved to: {plot_path}")
    
    print(f"\n{'='*60}")
    print(f"Done! Masks saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
