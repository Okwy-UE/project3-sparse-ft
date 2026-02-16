#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pruning.importance import (
    ImportanceConfig,
    compute_importance_scores,
    scores_to_mask_random,
    scores_to_mask_structured_block,
    scores_to_mask_unstructured,
)
from src.pruning.mask_utils import MaskSpec, default_exclude_for_llms, save_mask, mask_sparsity_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--task", required=True, choices=["boolq", "hellaswag", "gsm8k"])
    ap.add_argument("--sparsity", required=True, type=float)
    ap.add_argument("--sparsity-type", required=True, choices=["unstructured", "structured_block", "random"])
    ap.add_argument("--importance", required=True, choices=["magnitude", "wanda_like"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    icfg = ImportanceConfig(
        importance=args.importance,
        include_regex=r".*weight$",
        exclude_regex=default_exclude_for_llms(),
    )

    scores = compute_importance_scores(model, device=device, cfg=icfg)

    if args.sparsity_type == "unstructured":
        masks = scores_to_mask_unstructured(scores, sparsity=args.sparsity)
    elif args.sparsity_type == "random":
        masks = scores_to_mask_random(scores, sparsity=args.sparsity, seed=args.seed)
    else:
        masks = scores_to_mask_structured_block(scores, sparsity=args.sparsity)

    spec = MaskSpec(
        model_id=args.model_id,
        task=args.task,
        sparsity=args.sparsity,
        sparsity_type=args.sparsity_type,
        importance=args.importance,
        seed=args.seed,
        created_at=datetime.now().isoformat(),
    )

    out = Path(args.out or "mask.pt")
    save_mask(out, masks, spec)

    rep = mask_sparsity_report(masks)
    print(f"[OK] wrote mask -> {out}")
    print(f"[INFO] tensors masked: {len(rep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
