#!/usr/bin/env python3
import argparse
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt_path", required=True, help="masked CS .mdl path")
    ap.add_argument("--mask_type", required=True)
    ap.add_argument("--sparsity", required=True, type=float)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    trainer = cfg.setdefault("trainer", {})
    fit = trainer.setdefault("fit", {})
    fit["ckpt_path"] = args.ckpt_path

    # also store in config for provenance (harmless)
    init = trainer.setdefault("init", {})
    init.setdefault("logging", {}).setdefault("tags", {})
    init["logging"]["tags"].update({
        "mask_type": args.mask_type,
        "sparsity": float(args.sparsity),
        "sparse_notes": args.notes,
    })

    with open(args.yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("[OK] patched:", args.yaml)

if __name__ == "__main__":
    main()
