#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.pruning.mask_utils import load_mask, mask_sparsity_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    masks, meta = load_mask(args.mask)
    rep = mask_sparsity_report(masks)

    out = Path(args.out) if args.out else Path(args.mask).with_suffix(".report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "numel", "nnz", "sparsity"])
        for p, r in sorted(rep.items()):
            w.writerow([p, r["numel"], r["nnz"], f"{r['sparsity']:.6f}"])

    print(f"[OK] wrote report -> {out}")
    print(f"[META] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
