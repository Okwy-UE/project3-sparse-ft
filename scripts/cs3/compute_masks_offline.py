#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.phoenix_sft_tasks import build_sft_dataset, make_tokenize_fn
from src.pruning.mask_types import MaskSpec
from src.pruning.importance import collect_linear_input_l2, importance_unstructured_task
from src.pruning.validate_masks import validate_masks

def nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Structured N:M along input dimension (groups of size m in last dim).
    Keep top-n magnitudes per group for each output row.
    weight: [out, in]
    """
    w = weight.detach().abs()
    out, inn = w.shape
    pad = (m - (inn % m)) % m
    if pad:
        w = torch.nn.functional.pad(w, (0, pad), value=-1.0)
    w = w.view(out, -1, m)  # [out, groups, m]
    topk = torch.topk(w, k=n, dim=-1).indices
    mask = torch.zeros_like(w, dtype=torch.float32)
    mask.scatter_(-1, topk, 1.0)
    mask = mask.view(out, -1)
    if pad:
        mask = mask[:, :inn]
    return mask

def random_mask(weight: torch.Tensor, sparsity: float, g: torch.Generator) -> torch.Tensor:
    keep = 1.0 - sparsity
    return (torch.rand_like(weight, dtype=torch.float32, generator=g) < keep).float()

def topk_mask_from_scores(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    # keep fraction = 1 - sparsity
    keep = 1.0 - sparsity
    numel = scores.numel()
    k = int(round(keep * numel))
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.float32)
    if k >= numel:
        return torch.ones_like(scores, dtype=torch.float32)

    flat = scores.flatten()
    thresh = torch.kthvalue(flat, numel - k + 1).values  # kth smallest of descending => threshold
    mask = (scores >= thresh).float()
    # tie-breaker: adjust exact count if needed
    cur = int(mask.sum().item())
    if cur > k:
        # drop some ties
        eq = (scores == thresh).flatten().nonzero(as_tuple=False).flatten()
        drop = cur - k
        if drop > 0 and eq.numel() >= drop:
            mask_f = mask.flatten()
            mask_f[eq[:drop]] = 0.0
            mask = mask_f.view_as(scores)
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", required=True, help="HF model id or local path")
    ap.add_argument("--model_tag", required=True, choices=["llama3", "mistral", "mixtral"])
    ap.add_argument("--task", required=True, choices=["boolq", "hellaswag", "gsm8k"])
    ap.add_argument("--mask_type", required=True, choices=["unstructured_task", "random", "structured_nm"])
    ap.add_argument("--sparsity", required=True, type=float, help="fraction zeros, e.g. 0.5")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_samples", type=int, default=2048)
    ap.add_argument("--max_batches", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--nm_n", type=int, default=2)
    ap.add_argument("--nm_m", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    ds = build_sft_dataset(args.task, split="train", max_samples=args.max_samples)
    ds = ds.map(make_tokenize_fn(tok, max_seq_len=args.max_seq_len), remove_columns=ds.column_names)
    dl = DataLoader(ds, batch_size=2, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.float32)
    model.eval()

    in_l2 = None
    if args.mask_type == "unstructured_task":
        in_l2 = collect_linear_input_l2(model, dl, device=device, max_batches=args.max_batches)

    g = torch.Generator(device="cpu").manual_seed(args.seed)

    masks: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if args.mask_type == "structured_nm":
            masks[name] = nm_mask(p.data.cpu(), n=args.nm_n, m=args.nm_m)
        elif args.mask_type == "random":
            masks[name] = random_mask(p.data.cpu(), sparsity=args.sparsity, g=g)
        else:
            # unstructured task-aware
            # try to match Linear param names to module names for in_l2 lookup (best-effort)
            # fallback to magnitude-only if not found
            lin_name = name.rsplit(".", 1)[0]
            scores = importance_unstructured_task(p.data.cpu(), in_l2.get(lin_name) if in_l2 else None)
            masks[name] = topk_mask_from_scores(scores, sparsity=args.sparsity)

    spec = MaskSpec(
        model=args.model_tag,
        task=args.task,
        mask_type=args.mask_type,
        sparsity=float(args.sparsity),
        seed=int(args.seed),
        n=(args.nm_n if args.mask_type == "structured_nm" else None),
        m=(args.nm_m if args.mask_type == "structured_nm" else None),
    )

    torch.save(masks, os.path.join(args.out_dir, "mask.pt"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(spec.to_dict(), f, indent=2, sort_keys=True)

    stats = validate_masks(masks, os.path.join(args.out_dir, "layer_sparsity.csv"))
    with open(os.path.join(args.out_dir, "mask_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print("[OK] wrote:", args.out_dir)
    print("[OK] global_weighted_sparsity:", stats["global_weighted_sparsity"])

if __name__ == "__main__":
    main()
