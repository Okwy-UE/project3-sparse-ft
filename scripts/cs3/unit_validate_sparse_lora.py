#!/usr/bin/env python3
"""
Unit validation for dense LoRA vs masked LoRA behaviour.

Checks:
1) Dense LoRA baseline modifies unmasked AND masked coordinates.
2) Masked LoRA update enforces  (B @ A) ⊙ M  (zero in masked coords).
3) Merge with preserve_sparsity keeps all masked coordinates at zero.
4) Mask checksum remains stable through the whole process.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pruning.mask_ops import generate_mask_checksum
from pruning.sparse_lora import SparseLoRAConfig, SparseLoRALayer


def _build_mask(out_features: int, in_features: int, sparsity: float) -> torch.Tensor:
    numel = out_features * in_features
    num_zero = int(numel * sparsity)
    flat = torch.ones(numel, dtype=torch.float32)
    flat[:num_zero] = 0.0
    return flat.view(out_features, in_features)


def run_validation(seed: int = 7) -> dict:
    torch.manual_seed(seed)

    in_features, out_features = 16, 12
    mask = _build_mask(out_features, in_features, sparsity=0.5)
    checksum_before = generate_mask_checksum(mask)

    # ----- Dense LoRA baseline (no mask on the delta) -----
    dense_cfg = SparseLoRAConfig(
        r=4, alpha=8, dropout=0.0,
        masked_lora_update=False,
    )
    dense_base = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        dense_base.weight.fill_(0.01)
    dense_layer = SparseLoRALayer(dense_base, mask=mask.clone(), config=dense_cfg)
    with torch.no_grad():
        dense_layer.lora_B.fill_(0.1)
        dense_layer.lora_A.fill_(0.2)
    dense_delta = dense_layer._compute_lora_delta()

    # ----- Masked LoRA variant (delta masked by M) -----
    masked_cfg = SparseLoRAConfig(
        r=4, alpha=8, dropout=0.0,
        masked_lora_update=True,
    )
    masked_base = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        masked_base.weight.fill_(0.01)
    masked_layer = SparseLoRALayer(masked_base, mask=mask.clone(), config=masked_cfg)
    with torch.no_grad():
        masked_layer.lora_B.fill_(0.1)
        masked_layer.lora_A.fill_(0.2)
    masked_delta = masked_layer._compute_lora_delta()

    zero_region = mask == 0
    nonzero_region = mask == 1

    dense_nonzero_in_zero = int((torch.abs(dense_delta[zero_region]) > 0).sum().item())
    masked_nonzero_in_zero = int((torch.abs(masked_delta[zero_region]) > 0).sum().item())
    masked_nonzero_in_nonzero = int((torch.abs(masked_delta[nonzero_region]) > 0).sum().item())

    # ----- Merge check -----
    masked_layer.merge_weights(preserve_sparsity=True)
    merged_w = masked_layer.base_layer.weight.detach()
    merge_violations = int((torch.abs(merged_w[zero_region]) > 1e-8).sum().item())
    expected_zero_count = int(zero_region.sum().item())
    observed_zero_count = int((torch.abs(merged_w[zero_region]) <= 1e-8).sum().item())

    checksum_after = generate_mask_checksum(masked_layer.mask)

    # ----- Forward-pass smoke test -----
    x = torch.randn(2, in_features)
    dense_out = dense_layer(x)
    assert dense_out.shape == (2, out_features), "Dense forward shape mismatch"

    masked_cfg2 = SparseLoRAConfig(r=4, alpha=8, dropout=0.0, masked_lora_update=True)
    fwd_base = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        fwd_base.weight.fill_(0.01)
    fwd_layer = SparseLoRALayer(fwd_base, mask=mask.clone(), config=masked_cfg2)
    masked_out = fwd_layer(x)
    assert masked_out.shape == (2, out_features), "Masked forward shape mismatch"

    passed = (
        dense_nonzero_in_zero > 0
        and masked_nonzero_in_zero == 0
        and masked_nonzero_in_nonzero > 0
        and merge_violations == 0
        and expected_zero_count == observed_zero_count
        and checksum_before == checksum_after
    )

    return {
        "passed": passed,
        "dense_baseline_nonzero_in_masked_coords": dense_nonzero_in_zero,
        "masked_update_nonzero_in_masked_coords": masked_nonzero_in_zero,
        "masked_update_nonzero_in_unmasked_coords": masked_nonzero_in_nonzero,
        "merge_violations_in_masked_coords": merge_violations,
        "expected_zero_count_in_masked_coords": expected_zero_count,
        "observed_zero_count_in_masked_coords": observed_zero_count,
        "mask_checksum_before": checksum_before,
        "mask_checksum_after": checksum_after,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Unit validation for sparse LoRA")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    report = run_validation(seed=args.seed)
    print(json.dumps(report, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved report: {out}")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
