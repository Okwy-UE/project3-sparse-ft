#!/usr/bin/env python3
"""
Generate unstructured magnitude-based pruning masks for all model-task pairs.

This script runs on GPU nodes to generate masks that will be used on Cerebras CS-3.
Focuses on unstructured pruning with magnitude-based importance at 25%, 50%, and 75% sparsity.

Usage:
    python generate_all_masks_unstructured.py
"""

import sys
import argparse
from pathlib import Path
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import traceback

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pruning.mask_generator import compute_unstructured_mask
from pruning.mask_ops import save_mask, compute_sparsity_stats, validate_mask, plot_sparsity_histogram


# ============================================================================
# Configuration
# ============================================================================

# Models to process
MODELS = [
    # "llama3", 
    "mistral", 
    "mixtral"]

# Tasks for each model
TASKS = ["boolq", "hellaswag", "gsm8k"]

# Sparsity levels (as requested: 25%, 50%, 75%)
SPARSITIES = [0.25, 0.50, 0.75]

# Pruning method (fixed for this script)
METHOD = "unstructured"
IMPORTANCE = "magnitude"

# Hugging Face model mappings
HF_MODEL_NAMES = {
    "llama3": "meta-llama/Llama-3.1-8B",  # Use cached version (3.1 instead of 3.0)
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
}

# Alternative open models (no authentication required)
HF_MODEL_ALTERNATIVES = {
    "llama3": "openlm-research/open_llama_7b",  # Open alternative to LLaMA
}

# Default model cache directory (huggingface_hub stores repos under HF_HUB_CACHE or HF_HOME/hub)
_hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DEFAULT_CACHE_DIR = os.environ.get("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))


# ============================================================================
# Helper Functions
# ============================================================================

def load_model_from_hf(
    model_name: str,
    device: str = "cpu",
    use_alternative: bool = False,
    local_files_only: bool = False,
    mixtral_4bit: bool = False,
) -> torch.nn.Module:
    """
    Load a model from Hugging Face (or local cache).
    
    Args:
        model_name: Model identifier (e.g., "llama3", "mistral")
        device: Device to load on
        use_alternative: Use alternative open model if available
        local_files_only: Only use local cached models (don't download)
    
    Returns:
        Loaded model
    """
    try:
        from transformers import AutoModelForCausalLM
        from huggingface_hub.errors import GatedRepoError
        
        hf_name = HF_MODEL_NAMES.get(model_name)
        if not hf_name:
            raise ValueError(f"Unknown model: {model_name}")
        
        # If use_alternative is True and we have an alternative, use it
        if use_alternative and model_name in HF_MODEL_ALTERNATIVES:
            hf_name = HF_MODEL_ALTERNATIVES[model_name]
            print(f"Using alternative model: {hf_name}")
        
        cache_status = "from cache" if local_files_only else "from Hugging Face (or cache if available)"
        print(f"Loading {hf_name} {cache_status}...")
        
        # Use the same cache directory as gpu_dense_sft.py
        cache_dir = DEFAULT_CACHE_DIR
        
        try:
            # Pin to a single logical device to avoid ambiguous accelerator mapping.
            # When CUDA_VISIBLE_DEVICES is set to one GPU, that GPU is always index 0.
            if str(device).startswith("cuda"):
                resolved_device_map = {"": 0}
            elif str(device) == "cpu":
                resolved_device_map = {"": "cpu"}
            else:
                resolved_device_map = device

            # Use reduced precision on GPU to lower memory pressure.
            model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

            load_kwargs = dict(
                dtype=model_dtype,
                device_map=resolved_device_map,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=False,
            )

            if mixtral_4bit and model_name == "mixtral" and str(device).startswith("cuda"):
                try:
                    from transformers import BitsAndBytesConfig
                except Exception as exc:
                    raise RuntimeError(
                        "Mixtral 4-bit requested, but bitsandbytes support is unavailable. "
                        "Install bitsandbytes and ensure CUDA compatibility."
                    ) from exc
                print("Using 4-bit quantization for Mixtral (bitsandbytes).")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                **load_kwargs,
            )
            
            print(f"✓ Model loaded: {model.__class__.__name__}")
            if local_files_only:
                print(f"✓ Loaded from local cache: {cache_dir}")
            return model
            
        except (GatedRepoError, OSError) as e:
            error_str = str(e).lower()
            
            # If local_files_only failed, suggest trying without it
            if local_files_only and ("not found" in error_str or "no such file" in error_str):
                print(f"⚠️  Model {hf_name} not found in local cache.")
                print(f"   Cache location: {cache_dir}")
                print(f"   Try running without --local-only flag to download.")
                raise
            
            if "gated repo" in error_str or "restricted" in error_str:
                print(f"⚠️  Model {hf_name} requires authentication.")
                print(f"")
                print(f"To access gated models, you need to:")
                print(f"  1. Visit https://huggingface.co/{hf_name}")
                print(f"  2. Accept the terms and conditions")
                print(f"  3. Login with: huggingface-cli login")
                print(f"")
                
                # Try alternative if available
                if model_name in HF_MODEL_ALTERNATIVES and not use_alternative:
                    alt_name = HF_MODEL_ALTERNATIVES[model_name]
                    print(f"Trying alternative model: {alt_name}")
                    return load_model_from_hf(model_name, device, use_alternative=True, local_files_only=local_files_only)
                
                raise
            else:
                raise
        
    except Exception as e:
        print(f"Error loading model from HF: {e}")
        raise


def compute_masks_for_model(
    model: torch.nn.Module,
    model_name: str,
    sparsity: float,
    method: str = "unstructured",
    importance_metric: str = "magnitude",
    device: str = "cpu",
    global_pruning: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute masks for a model using specified method.
    
    Args:
        model: PyTorch model
        model_name: Model name (for logging)
        sparsity: Target sparsity (0-1)
        method: Pruning method
        importance_metric: Importance metric
        device: Device to run on
    
    Returns:
        Dictionary of masks
    """
    print(f"\nComputing {method} masks (sparsity={sparsity:.0%}, importance={importance_metric})...")
    
    # For magnitude-based pruning, we don't need a dataloader
    masks = compute_unstructured_mask(
        model,
        sparsity=sparsity,
        dataloader=None,
        loss_fn=None,
        importance_metric=importance_metric,
        device=device,
        global_pruning=global_pruning,
    )
    
    return masks


def save_masks_with_metadata(
    masks: Dict[str, torch.Tensor],
    model_name: str,
    task_name: str,
    sparsity: float,
    method: str,
    importance: str,
    output_dir: Path,
    plot: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    Save masks with comprehensive metadata.
    
    Returns:
        (mask_path, stats_path, plot_path)
    """
    # Create output directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    stats = compute_sparsity_stats(masks)
    
    # Validate masks
    all_valid = True
    validation_results = {}
    for name, mask in masks.items():
        is_valid, mask_stats = validate_mask(mask, expected_sparsity=sparsity, tolerance=0.05)
        validation_results[name] = {
            "valid": is_valid,
            "sparsity": mask_stats["sparsity"],
        }
        if not is_valid:
            all_valid = False
            print(f"  ⚠️  {name}: {mask_stats.get('error', 'Invalid')}")
    
    if all_valid:
        print(f"  ✓ All masks valid")
    
    # Prepare metadata
    metadata = {
        "model": model_name,
        "task": task_name,
        "sparsity": sparsity,
        "method": method,
        "importance": importance,
        "created_at": datetime.now().isoformat(),
        "stats": stats,
        "validation": {
            "all_valid": all_valid,
            "per_layer": validation_results,
        },
        "phoenix_compatible": True,
        "format_version": "1.0",
    }
    
    # File paths
    sparsity_int = int(sparsity * 100)
    mask_filename = f"masks_sparsity{sparsity_int}_{method}.pt"
    mask_path = model_dir / mask_filename
    
    # Save masks (PyTorch format for Cerebras compatibility)
    save_mask(masks, str(mask_path), metadata=metadata, format="pt")
    print(f"  ✓ Masks saved: {mask_path}")
    
    # Also save as NPZ for GPU compatibility
    mask_npz_path = model_dir / f"masks_sparsity{sparsity_int}_{method}.npz"
    save_mask(masks, str(mask_npz_path), metadata=metadata, format="npz")
    print(f"  ✓ Masks saved (NPZ): {mask_npz_path}")
    
    # Save stats as JSON
    stats_path = model_dir / f"stats_sparsity{sparsity_int}_{method}.json"
    with open(stats_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Statistics saved: {stats_path}")
    
    # Plot histogram
    plot_path = None
    if plot:
        plot_path = model_dir / f"histogram_sparsity{sparsity_int}_{method}.png"
        try:
            plot_sparsity_histogram(masks, save_path=str(plot_path))
            print(f"  ✓ Histogram saved: {plot_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to generate plot: {e}")
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Global sparsity: {stats['global']['sparsity']:.2%}")
    print(f"    Mean layer sparsity: {stats['global']['mean_layer_sparsity']:.2%} ± {stats['global']['std_layer_sparsity']:.2%}")
    print(f"    Layers: {stats['global']['num_layers']}")
    print(f"    Total zeros: {stats['global']['total_zeros']:,} / {stats['global']['total_elements']:,}")
    
    return mask_path, stats_path, plot_path


# ============================================================================
# Main Processing
# ============================================================================

def process_model_task_pair(
    model_name: str,
    task_name: str,
    sparsities: List[float],
    output_dir: Path,
    device: str = "cpu",
    plot: bool = True,
    skip_gated: bool = False,
    use_alternatives: bool = False,
    local_only: bool = False,
    global_pruning: bool = False,
    mixtral_4bit: bool = True,
) -> bool:
    """
    Process a single model-task pair for all sparsity levels.
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"Processing: {model_name} + {task_name}")
    print("="*80)
    
    try:
        # Load model
        model = load_model_from_hf(
            model_name, 
            device=device, 
            use_alternative=use_alternatives,
            local_files_only=local_only,
            mixtral_4bit=mixtral_4bit,
        )
        
        # Process each sparsity level
        for sparsity in sparsities:
            print(f"\n{'-'*60}")
            print(f"Sparsity level: {sparsity:.0%}")
            print(f"{'-'*60}")
            
            # Compute masks
            masks = compute_masks_for_model(
                model,
                model_name,
                sparsity=sparsity,
                method=METHOD,
                importance_metric=IMPORTANCE,
                device=device,
                global_pruning=global_pruning,
            )
            
            # Save masks with metadata
            save_masks_with_metadata(
                masks,
                model_name,
                task_name,
                sparsity,
                METHOD,
                IMPORTANCE,
                output_dir,
                plot=plot,
            )
        
        print(f"\n✓ Successfully processed {model_name} + {task_name}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a gated repo error
        if "gated" in error_msg.lower() or "restricted" in error_msg.lower():
            if skip_gated:
                print(f"\n⚠️  Skipped {model_name} + {task_name} (gated model, --skip-gated enabled)")
                return False
            else:
                print(f"\n✗ Failed to process {model_name} + {task_name} (gated model)")
                print(f"  Hint: Use --skip-gated to skip gated models, or --use-alternatives for open alternatives")
                return False
        
        print(f"\n✗ Failed to process {model_name} + {task_name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate unstructured magnitude-based masks for all model-task pairs"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "masks"),
        help="Output directory for masks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Models to process",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Tasks to process",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=SPARSITIES,
        help="Sparsity levels to use",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram plotting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip models that require authentication",
    )
    parser.add_argument(
        "--use-alternatives",
        action="store_true",
        help="Use alternative open models when available",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="Only use locally cached models (don't download from HF) [default: True]",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading models from HuggingFace if not in cache",
    )
    parser.add_argument(
        "--global-pruning",
        action="store_true",
        help="Use global pruning across all layers (higher memory use)",
    )
    parser.add_argument(
        "--mixtral-4bit",
        action="store_true",
        help="Load Mixtral in 4-bit (bitsandbytes) to reduce GPU memory",
    )
    
    args = parser.parse_args()
    
    # If --allow-download is set, disable local_only
    if args.allow_download:
        args.local_only = False
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Unstructured Magnitude-Based Mask Generation")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Sparsities: {', '.join(f'{s:.0%}' for s in args.sparsities)}")
    print(f"Method: {METHOD}")
    print(f"Importance: {IMPORTANCE}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Cache mode: {'local only' if args.local_only else 'allow download'}")
    print(f"Pruning scope: {'global' if args.global_pruning else 'layer-wise'}")
    print(f"Mixtral 4-bit: {'enabled' if args.mixtral_4bit else 'disabled'}")
    print("="*80)
    
    if args.dry_run:
        print("\nDRY RUN - no masks will be generated")
        total_runs = len(args.models) * len(args.tasks) * len(args.sparsities)
        print(f"Would generate masks for {total_runs} configurations:")
        for model in args.models:
            for task in args.tasks:
                for sparsity in args.sparsities:
                    print(f"  - {model} + {task} @ {sparsity:.0%} sparsity")
        return
    
    # Process all model-task pairs
    results = {}
    total_pairs = len(args.models) * len(args.tasks)
    current_pair = 0
    
    for model_name in args.models:
        for task_name in args.tasks:
            current_pair += 1
            print(f"\n{'='*80}")
            print(f"Progress: {current_pair}/{total_pairs}")
            print(f"{'='*80}")
            
            success = process_model_task_pair(
                model_name,
                task_name,
                args.sparsities,
                output_dir,
                device=args.device,
                plot=not args.no_plot,
                skip_gated=args.skip_gated,
                use_alternatives=args.use_alternatives,
                local_only=args.local_only,
                global_pruning=args.global_pruning,
                mixtral_4bit=args.mixtral_4bit,
            )
            
            results[f"{model_name}_{task_name}"] = success
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print(f"Total pairs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed pairs:")
        for pair, success in results.items():
            if not success:
                print(f"  ✗ {pair}")
    
    print("\n" + "="*80)
    print(f"Masks saved to: {output_dir}")
    print("="*80)
    
    # Create registry
    registry_path = output_dir / "mask_registry.json"
    registry = {
        "created_at": datetime.now().isoformat(),
        "method": METHOD,
        "importance": IMPORTANCE,
        "sparsities": args.sparsities,
        "models": args.models,
        "tasks": args.tasks,
        "results": results,
        "output_dir": str(output_dir),
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nRegistry saved: {registry_path}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
