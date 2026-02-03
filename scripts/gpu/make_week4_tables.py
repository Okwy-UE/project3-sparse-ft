# scripts/gpu/make_week4_tables.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

def read_registry(p: Path) -> List[Dict[str, Any]]:
    with p.open("r", newline="") as f:
        return list(csv.DictReader(f))

def fnum(x: str):
    try:
        return float(x)
    except Exception:
        return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/run_registry.csv")
    ap.add_argument("--out_dir", default="results/tables")
    args = ap.parse_args()

    reg = read_registry(Path(args.registry))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group_id -> {bench_points, train, eval}
    grouped = defaultdict(lambda: {"bench": [], "train": None, "eval": None, "meta": None})

    for r in reg:
        if str(r.get("week", "")).strip() != "4":
            continue
        gid = r.get("group_id", "")
        kind = r.get("run_kind", "")
        grouped[gid]["meta"] = grouped[gid]["meta"] or r

        if kind == "bench":
            grouped[gid]["bench"].append(r)
        elif kind == "train":
            grouped[gid]["train"] = r
        else:
            # eval run copies meta; run_kind might be "train" if copied—detect by presence of eval.accuracy
            if r.get("eval.accuracy", "") != "":
                grouped[gid]["eval"] = r

    rows_out = []
    for gid, pack in grouped.items():
        meta = pack["meta"] or {}
        model = meta.get("model_name", "")
        task = meta.get("task", "")

        # bench points (bsz -> tokens/s)
        bench = pack["bench"]
        bench_pairs = []
        for b in bench:
            bsz = b.get("per_device_train_batch_size", "")
            tps = b.get("throughput.tokens_per_s", "")
            bench_pairs.append((int(bsz) if bsz else -1, fnum(tps)))
        bench_pairs = sorted([x for x in bench_pairs if x[0] > 0 and x[1] is not None], key=lambda x: x[0])
        bench_str = ", ".join([f"bsz{bsz}:{tps:.1f}" for bsz, tps in bench_pairs])

        train = pack["train"] or {}
        evalr = pack["eval"] or {}

        rows_out.append({
            "group_id": gid,
            "model": model,
            "task": task,
            "bench_tokens_per_s": bench_str,
            "train_effective_global_bsz": train.get("train.effective_global_bsz", ""),
            "train_peak_mem_gb": train.get("train.peak_mem_gb", ""),
            "train_loss": train.get("train.train_loss", ""),
            "eval_accuracy": evalr.get("eval.accuracy", ""),
            "gpu_name": meta.get("gpu_name", ""),
            "world_size": meta.get("world_size", ""),
            "seq_len": meta.get("seq_len", ""),
        })

    # Write CSV
    out_csv = out_dir / "week4_h100_dense_baseline.csv"
    cols = list(rows_out[0].keys()) if rows_out else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)

    # Write Markdown
    out_md = out_dir / "week4_h100_dense_baseline.md"
    with out_md.open("w") as f:
        f.write("# Week 4 Dense GPU Baseline (H100/H200)\n\n")
        if not rows_out:
            f.write("_No Week 4 rows found in registry._\n")
            return
        f.write("| Model | Task | Bench tokens/s (per-dev bsz) | Train peak GB | Eval acc |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows_out:
            f.write(f"| {r['model']} | {r['task']} | {r['bench_tokens_per_s']} | {r['train_peak_mem_gb']} | {r['eval_accuracy']} |\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")

if __name__ == "__main__":
    main()
