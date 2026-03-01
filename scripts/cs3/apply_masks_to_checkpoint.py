#!/usr/bin/env python3
"""
Apply pre-computed sparsity masks to a HuggingFace checkpoint.

Reads sharded safetensors (or a single pytorch_model.bin), zeros out
weights where the mask is 0, and writes the modified checkpoint to a
new directory.  The resulting checkpoint can then be converted to
Cerebras CS-2.5 format with ``cszoo checkpoint convert``.

Mask keys (from generate_all_masks_unstructured.py) look like:
    model.layers.0.self_attn.q_proj

Corresponding HF state-dict keys are:
    model.layers.0.self_attn.q_proj.weight

Usage:
    python apply_masks_to_checkpoint.py \\
        --model llama3 \\
        --sparsity 25 \\
        --mask_path masks/llama3/masks_sparsity25_unstructured.pt \\
        --hf_dir checkpoints/hf/llama3p1_8b \\
        --output_dir checkpoints/hf_sparse/llama3_s25
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pruning.mask_ops import load_mask, compute_sparsity_stats


# ======================================================================
# Key mapping helpers
# ======================================================================

def mask_key_to_weight_key(mask_key: str) -> str:
    """Convert a module-name mask key to a state-dict weight key."""
    return mask_key + ".weight"


def _try_match(weight_key, masks):
    """Try to find a mask for *weight_key*, returning the mask-dict key."""
    if not weight_key.endswith(".weight"):
        return None
    module_key = weight_key[: -len(".weight")]
    if module_key in masks:
        return module_key
    return None


# ======================================================================
# Safetensors helpers
# ======================================================================

def _load_safetensors_index(hf_dir):
    idx_path = hf_dir / "model.safetensors.index.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text())
    return None


def _shard_files(hf_dir: Path) -> list[Path]:
    """Return list of safetensors shard files (sorted)."""
    return sorted(hf_dir.glob("model-*.safetensors"))


# ======================================================================
# Core masking
# ======================================================================

def apply_masks_safetensors(
    hf_dir: Path,
    output_dir: Path,
    masks: dict,
    metadata: dict,
) -> dict:
    """Apply masks to a sharded safetensors checkpoint."""
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        sys.exit("safetensors package required.  pip install safetensors")

    stats = {"matched": 0, "unmatched_mask_keys": [], "total_params_masked": 0}

    shards = _shard_files(hf_dir)
    if not shards:
        single = hf_dir / "model.safetensors"
        if single.exists():
            shards = [single]
        else:
            sys.exit(f"No safetensors files found in {hf_dir}")

    mask_keys_used = set()

    for shard in shards:
        print(f"  Processing shard: {shard.name}")
        sd = load_file(str(shard))
        modified = False

        for wk in list(sd.keys()):
            mk = _try_match(wk, masks)
            if mk is not None:
                mask_tensor = masks[mk]
                w = sd[wk]
                if mask_tensor.shape != w.shape:
                    print(f"    SKIP shape mismatch: {wk} "
                          f"weight={w.shape} mask={mask_tensor.shape}")
                    continue
                sd[wk] = w * mask_tensor.to(dtype=w.dtype)
                mask_keys_used.add(mk)
                stats["matched"] += 1
                stats["total_params_masked"] += int((mask_tensor == 0).sum().item())
                modified = True

        out_path = output_dir / shard.name
        save_file(sd, str(out_path))
        del sd

    stats["unmatched_mask_keys"] = sorted(set(masks.keys()) - mask_keys_used)
    return stats


def apply_masks_pytorch(
    hf_dir: Path,
    output_dir: Path,
    masks: dict,
    metadata: dict,
) -> dict:
    """Apply masks to a pytorch_model.bin (or sharded .bin) checkpoint."""
    stats = {"matched": 0, "unmatched_mask_keys": [], "total_params_masked": 0}

    bin_files = sorted(hf_dir.glob("pytorch_model*.bin"))
    if not bin_files:
        sys.exit(f"No pytorch_model*.bin found in {hf_dir}")

    mask_keys_used = set()

    for bf in bin_files:
        print(f"  Processing: {bf.name}")
        sd = torch.load(str(bf), map_location="cpu", weights_only=False)

        for wk in list(sd.keys()):
            mk = _try_match(wk, masks)
            if mk is not None:
                mask_tensor = masks[mk]
                w = sd[wk]
                if mask_tensor.shape != w.shape:
                    print(f"    SKIP shape mismatch: {wk}")
                    continue
                sd[wk] = w * mask_tensor.to(dtype=w.dtype)
                mask_keys_used.add(mk)
                stats["matched"] += 1
                stats["total_params_masked"] += int((mask_tensor == 0).sum().item())

        out_path = output_dir / bf.name
        torch.save(sd, str(out_path))
        del sd

    stats["unmatched_mask_keys"] = sorted(set(masks.keys()) - mask_keys_used)
    return stats


# ======================================================================
# Copy auxiliary files
# ======================================================================

AUX_EXTENSIONS = {
    ".json", ".txt", ".model", ".vocab",
    ".tiktoken", ".py", ".md",
}
AUX_NAMES = {
    "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "config.json",
    "generation_config.json", "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def copy_aux_files(src: Path, dst: Path):
    """Symlink config, tokenizer, and other non-weight files (saves quota)."""
    for f in src.iterdir():
        if f.is_dir():
            continue
        if f.name in AUX_NAMES or f.suffix in AUX_EXTENSIONS:
            target = dst / f.name
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                target.symlink_to(f.resolve())
                print(f"  Linked {f.name}")
            except OSError:
                shutil.copy2(f, target)
                print(f"  Copied {f.name}")


# ======================================================================
# Main
# ======================================================================

MODEL_DIRS = {
    "llama3": "llama3p1_8b",
    "mistral": "mistral_7b",
    "mixtral": "mixtral_8x7b",
}


def main():
    ap = argparse.ArgumentParser(
        description="Apply pruning masks to an HF checkpoint")
    ap.add_argument("--model", required=True, choices=list(MODEL_DIRS))
    ap.add_argument("--sparsity", type=int, required=True,
                    help="Sparsity percentage (e.g. 25, 50, 75)")
    ap.add_argument("--mask_path", type=str, default="",
                    help="Path to mask .pt file (auto-resolved if empty)")
    ap.add_argument("--hf_dir", type=str, default="",
                    help="HF checkpoint dir (auto-resolved if empty)")
    ap.add_argument("--output_dir", type=str, default="",
                    help="Where to write the masked HF checkpoint")
    args = ap.parse_args()

    proj = PROJECT_ROOT
    sparsity = args.sparsity

    mask_path = Path(args.mask_path) if args.mask_path else (
        proj / "masks" / args.model / f"masks_sparsity{sparsity}_unstructured.pt"
    )
    hf_dir = Path(args.hf_dir) if args.hf_dir else (
        proj / "checkpoints" / "hf" / MODEL_DIRS[args.model]
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        proj / "checkpoints" / "hf_sparse" / f"{args.model}_s{sparsity}"
    )

    print("=" * 70)
    print(f"Applying masks to HF checkpoint")
    print(f"  Model    : {args.model}")
    print(f"  Sparsity : {sparsity}%")
    print(f"  Mask     : {mask_path}")
    print(f"  HF dir   : {hf_dir}")
    print(f"  Output   : {output_dir}")
    print("=" * 70)

    if not mask_path.exists():
        sys.exit(f"Mask file not found: {mask_path}")
    if not hf_dir.exists():
        sys.exit(f"HF checkpoint dir not found: {hf_dir}")

    masks, meta = load_mask(str(mask_path))
    mask_stats = compute_sparsity_stats(masks)
    print(f"  Loaded {len(masks)} mask tensors, "
          f"global sparsity = {mask_stats['global']['sparsity']:.2%}")

    output_dir.mkdir(parents=True, exist_ok=True)

    has_safetensors = bool(list(hf_dir.glob("*.safetensors")))
    if has_safetensors:
        stats = apply_masks_safetensors(hf_dir, output_dir, masks, meta)
    else:
        stats = apply_masks_pytorch(hf_dir, output_dir, masks, meta)

    copy_aux_files(hf_dir, output_dir)

    print(f"\nResults:")
    print(f"  Layers masked : {stats['matched']}")
    print(f"  Params zeroed : {stats['total_params_masked']:,}")
    if stats["unmatched_mask_keys"]:
        print(f"  Unmatched mask keys ({len(stats['unmatched_mask_keys'])}):")
        for k in stats["unmatched_mask_keys"][:10]:
            print(f"    {k}")

    report_path = output_dir / "masking_report.json"
    report = {
        "model": args.model,
        "sparsity": sparsity,
        "mask_path": str(mask_path),
        "hf_dir": str(hf_dir),
        "output_dir": str(output_dir),
        "stats": stats,
        "mask_stats_global": mask_stats["global"],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
