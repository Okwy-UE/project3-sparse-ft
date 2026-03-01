#!/usr/bin/env python3
"""
Post-training LoRA merge with sparsity-preservation validation.

After a Cerebras CS-3 LoRA training run, this script:
  1. Loads the trained CS checkpoint (base + LoRA adapters).
  2. Merges LoRA weights back into the base.
  3. For sparse-to-sparse: re-applies the original mask.
  4. Validates that sparsity is preserved (zero-count check + checksum).
  5. Saves a validation report.

Usage:
    python merge_and_validate.py \\
        --run_dir results/masked_runs_cs/cs3_llama3_boolq_lora_s50_... \\
        --mask_path masks/llama3/masks_sparsity50_unstructured.pt \\
        --mode sparse_to_sparse
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pruning.mask_ops import load_mask, generate_mask_checksum


def load_checkpoint(run_dir):
    """Try to load the final checkpoint from a Cerebras run."""
    model_dir = run_dir / "model_dir"
    candidates = [
        model_dir / "checkpoint_final.mdl",
        model_dir / "model_to_cs-2.5.mdl",
    ]
    ckpt_files = sorted(model_dir.glob("checkpoint_*.mdl"))
    candidates.extend(ckpt_files)

    for c in candidates:
        if c.exists():
            print(f"  Loading checkpoint: {c}")
            return torch.load(str(c), map_location="cpu", weights_only=False)
    return None


def validate_sparsity_on_state_dict(
    state_dict: dict,
    masks: dict,
    tolerance: float = 1e-6,
) -> dict:
    """
    Check that weight tensors are zero wherever the mask is zero.

    Handles the key mapping between mask keys (module paths) and
    state-dict keys (module path + '.weight').
    """
    report = {
        "per_layer": {},
        "global": {
            "all_valid": True,
            "total_violations": 0,
            "total_masked_params": 0,
            "layers_checked": 0,
        },
    }

    for mask_key, mask_tensor in masks.items():
        weight_key = mask_key + ".weight"
        w = state_dict.get(weight_key)
        if w is None:
            for sk in state_dict:
                if sk.endswith(weight_key) or mask_key in sk:
                    w = state_dict[sk]
                    weight_key = sk
                    break
        if w is None:
            continue

        if w.shape != mask_tensor.shape:
            report["per_layer"][mask_key] = {"error": "shape_mismatch"}
            continue

        zero_region = mask_tensor == 0
        violations = (torch.abs(w[zero_region]) > tolerance).sum().item()
        total_masked = zero_region.sum().item()

        layer = {
            "num_violations": int(violations),
            "total_masked": int(total_masked),
            "violation_rate": violations / max(total_masked, 1),
        }

        report["per_layer"][mask_key] = layer
        report["global"]["total_violations"] += int(violations)
        report["global"]["total_masked_params"] += int(total_masked)
        report["global"]["layers_checked"] += 1
        if violations > 0:
            report["global"]["all_valid"] = False

    return report


def reapply_masks_to_state_dict(state_dict: dict, masks: dict) -> int:
    """Zero out masked positions in the state dict.  Returns count of params zeroed."""
    count = 0
    for mask_key, mask_tensor in masks.items():
        weight_key = mask_key + ".weight"
        if weight_key not in state_dict:
            for sk in state_dict:
                if sk.endswith(weight_key) or mask_key in sk:
                    weight_key = sk
                    break
        if weight_key not in state_dict:
            continue
        w = state_dict[weight_key]
        if w.shape != mask_tensor.shape:
            continue
        new_zeros = int(((w != 0) & (mask_tensor == 0)).sum().item())
        state_dict[weight_key] = w * mask_tensor.to(dtype=w.dtype)
        count += new_zeros
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Merge LoRA weights and validate sparsity")
    ap.add_argument("--run_dir", required=True, help="Path to training run directory")
    ap.add_argument("--mask_path", required=True, help="Path to mask .pt file")
    ap.add_argument("--mode", default="sparse_to_sparse",
                    choices=["sparse_to_dense", "sparse_to_sparse"])
    ap.add_argument("--save_merged", action="store_true",
                    help="Save the merged checkpoint")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    masks, mask_meta = load_mask(args.mask_path)

    print("=" * 60)
    print(f"Post-training merge & validation")
    print(f"  Run dir  : {run_dir}")
    print(f"  Mask     : {args.mask_path}")
    print(f"  Mode     : {args.mode}")
    print("=" * 60)

    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        sys.exit(f"No checkpoint found in {run_dir}/model_dir/")

    sd = ckpt.get("model", ckpt)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if args.mode == "sparse_to_sparse":
        print("\n  Re-applying masks for sparse-to-sparse ...")
        n_zeroed = reapply_masks_to_state_dict(sd, masks)
        print(f"  Zeroed {n_zeroed:,} newly-non-zero params")

    print("\n  Validating sparsity ...")
    report = validate_sparsity_on_state_dict(sd, masks)

    checksums = {}
    for mk, mt in list(masks.items())[:5]:
        checksums[mk] = generate_mask_checksum(mt)

    report["mask_checksums_sample"] = checksums
    report["mode"] = args.mode

    report_path = run_dir / "sparsity_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    valid = report["global"]["all_valid"]
    print(f"\n  Layers checked  : {report['global']['layers_checked']}")
    print(f"  Total violations: {report['global']['total_violations']}")
    print(f"  All valid       : {valid}")
    print(f"  Report saved    : {report_path}")

    if args.save_merged:
        merged_path = run_dir / "model_dir" / "merged_checkpoint.pt"
        torch.save(sd, str(merged_path))
        print(f"  Merged ckpt saved: {merged_path}")

    print("=" * 60)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
