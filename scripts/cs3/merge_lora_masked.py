#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, hashlib
import torch

from src.pruning.apply_mask import count_zeros

def sha256_tensor(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_state_dict", required=True, help="path to merged state_dict.pt (W+BA)")
    ap.add_argument("--mask_pt", required=True, help="mask.pt")
    ap.add_argument("--out_state_dict", required=True)
    ap.add_argument("--report_json", required=True)
    args = ap.parse_args()

    sd = torch.load(args.merged_state_dict, map_location="cpu")
    masks = torch.load(args.mask_pt, map_location="cpu")

    report = {"checked": [], "global": {}}
    total_zeros_before = 0
    total_zeros_after = 0
    total_numel = 0

    for k, v in sd.items():
        if k in masks and isinstance(v, torch.Tensor) and v.shape == masks[k].shape:
            z0, n0 = count_zeros(v)
            v2 = v * masks[k].to(v.dtype)
            z1, n1 = count_zeros(v2)
            sd[k] = v2
            report["checked"].append({
                "param": k,
                "zeros_before": z0,
                "zeros_after": z1,
                "numel": n0,
                "mask_sha256": sha256_tensor(masks[k].float()),
            })
            total_zeros_before += z0
            total_zeros_after += z1
            total_numel += n0

    report["global"] = {
        "zeros_before": total_zeros_before,
        "zeros_after": total_zeros_after,
        "numel": total_numel,
        "sparsity_before": (total_zeros_before / max(total_numel, 1)),
        "sparsity_after": (total_zeros_after / max(total_numel, 1)),
    }

    os.makedirs(os.path.dirname(args.out_state_dict), exist_ok=True)
    torch.save(sd, args.out_state_dict)
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[OK] wrote masked-merged state_dict:", args.out_state_dict)
    print("[OK] report:", args.report_json)
    print("[OK] global sparsity_after:", report["global"]["sparsity_after"])

if __name__ == "__main__":
    main()
