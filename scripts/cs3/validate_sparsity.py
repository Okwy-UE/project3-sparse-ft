#!/usr/bin/env python3
"""
Validate sparsity preservation after training.

Checks that:
1. Masks are correctly applied
2. Sparsity is preserved after LoRA merge
3. Zero counts match expected values
"""

import argparse
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pruning.mask_ops import load_mask, validate_mask, compute_sparsity_stats
from pruning.sparse_lora import validate_sparsity_preserved


def load_checkpoint(checkpoint_path: str):
    """Load a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to find checkpoint file
    if checkpoint_path.is_dir():
        # Look for checkpoint files
        ckpt_files = list(checkpoint_path.glob("*.pt")) + list(checkpoint_path.glob("*.pth"))
        if not ckpt_files:
            ckpt_files = list(checkpoint_path.glob("**/*.pt")) + list(checkpoint_path.glob("**/*.pth"))
        
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        
        # Use the latest checkpoint
        ckpt_file = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {ckpt_file}")
    else:
        ckpt_file = checkpoint_path
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    return state_dict


def validate_checkpoint_sparsity(
    checkpoint_path: str,
    mask_path: str,
    tolerance: float = 1e-6,
) -> dict:
    """
    Validate sparsity in a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        mask_path: Path to sparsity masks
        tolerance: Tolerance for zero checking
    
    Returns:
        Validation report
    """
    # Load checkpoint and masks
    state_dict = load_checkpoint(checkpoint_path)
    masks, mask_metadata = load_mask(mask_path)
    
    print(f"Loaded checkpoint with {len(state_dict)} keys")
    print(f"Loaded {len(masks)} masks")
    print(f"Expected sparsity: {mask_metadata.get('sparsity', 'unknown')}")
    
    # Validate each layer
    report = {
        'checkpoint_path': str(checkpoint_path),
        'mask_path': str(mask_path),
        'expected_sparsity': mask_metadata.get('sparsity'),
        'per_layer': {},
        'global': {
            'total_violations': 0,
            'total_masked_elements': 0,
            'all_valid': True,
        }
    }
    
    for mask_name, mask in masks.items():
        # Find corresponding weight in state dict
        weight_key = f"{mask_name}.weight"
        
        # Try alternative naming schemes
        if weight_key not in state_dict:
            # Try without module prefix
            weight_key = weight_key.replace('module.', '')
        if weight_key not in state_dict:
            # Try with module prefix
            weight_key = f"module.{mask_name}.weight"
        if weight_key not in state_dict:
            print(f"⚠️  Weight not found for mask: {mask_name}")
            continue
        
        weight = state_dict[weight_key]
        
        # Check that masked weights are zero
        masked_indices = (mask == 0)
        masked_weights = weight[masked_indices]
        violations = torch.abs(masked_weights) > tolerance
        
        num_violations = violations.sum().item()
        num_masked = masked_indices.sum().item()
        max_violation = torch.abs(masked_weights).max().item() if num_masked > 0 else 0.0
        
        # Compute actual sparsity
        actual_sparsity = (weight == 0).sum().item() / weight.numel()
        
        layer_report = {
            'num_violations': num_violations,
            'num_masked': num_masked,
            'violation_rate': num_violations / max(num_masked, 1),
            'max_violation': max_violation,
            'actual_sparsity': actual_sparsity,
            'expected_sparsity': mask_metadata.get('sparsity'),
            'valid': num_violations == 0,
        }
        
        report['per_layer'][mask_name] = layer_report
        report['global']['total_violations'] += num_violations
        report['global']['total_masked_elements'] += num_masked
        
        if num_violations > 0:
            report['global']['all_valid'] = False
            print(f"⚠️  {mask_name}: {num_violations} violations (max: {max_violation:.2e})")
    
    # Compute global statistics
    if report['global']['total_masked_elements'] > 0:
        report['global']['violation_rate'] = (
            report['global']['total_violations'] / report['global']['total_masked_elements']
        )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total violations: {report['global']['total_violations']}")
    print(f"Total masked elements: {report['global']['total_masked_elements']}")
    if report['global']['all_valid']:
        print("✓ All layers valid - sparsity preserved!")
    else:
        print("✗ Sparsity NOT preserved")
        print(f"  Violation rate: {report['global']['violation_rate']:.2%}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate sparsity preservation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask file")
    parser.add_argument("--output", type=str, help="Path to save validation report (JSON)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for zero check")
    
    args = parser.parse_args()
    
    # Validate
    report = validate_checkpoint_sparsity(
        args.checkpoint,
        args.mask_path,
        tolerance=args.tolerance,
    )
    
    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ Validation report saved to: {output_path}")
    
    # Exit with error code if validation failed
    if not report['global']['all_valid']:
        sys.exit(1)


if __name__ == "__main__":
    main()
