#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pruning.apply_mask import apply_mask_to_state_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_in", required=True, help="HF model id or local dir")
    ap.add_argument("--mask_pt", required=True)
    ap.add_argument("--out_dir", required=True, help="output HF-style directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_in, use_fast=True)
    tok.save_pretrained(args.out_dir)

    model = AutoModelForCausalLM.from_pretrained(args.hf_in, torch_dtype=torch.float32)
    masks = torch.load(args.mask_pt, map_location="cpu")

    sd = model.state_dict()
    sd2 = apply_mask_to_state_dict(sd, masks)
    model.load_state_dict(sd2, strict=False)

    model.save_pretrained(args.out_dir)
    print("[OK] masked checkpoint written:", args.out_dir)

if __name__ == "__main__":
    main()
