from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def discover_runs(runs_dir: Path) -> List[Path]:
    return [p for p in runs_dir.glob("*") if p.is_dir()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="results/runs")
    ap.add_argument("--out_csv", type=str, default="results/tables/week4_gpu_baselines.csv")
    ap.add_argument("--out_md", type=str, default="results/tables/week4_gpu_baselines.md")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []

    for r in discover_runs(runs_dir):
        meta_p = r / "run_meta.json"
        if not meta_p.exists():
            continue
        meta = load_json(meta_p)
        model = (r / "model_id.txt").read_text().strip() if (r / "model_id.txt").exists() else meta.get("model_name")
        task = (r / "task_id.txt").read_text().strip() if (r / "task_id.txt").exists() else meta.get("task")

        row = {
            "run_dir": str(r),
            "run_kind": meta.get("run_kind"),
            "model": model,
            "task": task,
            "gpu": meta.get("gpu_name"),
            "world_size": meta.get("world_size"),
            "seq_len": meta.get("seq_len"),
            "micro_bsz": meta.get("per_device_train_batch_size"),
            "grad_accum": meta.get("gradient_accumulation_steps"),
        }

        tp = r / "throughput.json"
        if tp.exists():
            t = load_json(tp)
            row.update({
                "samples_per_s": t.get("samples_per_s"),
                "tokens_per_s": t.get("tokens_per_s"),
                "peak_mem_gb": t.get("peak_mem_gb"),
                "step_time_s": t.get("step_time_s"),
                "effective_global_bsz": t.get("effective_global_bsz"),
            })

        tr = r / "train_summary.json"
        if tr.exists():
            t = load_json(tr)
            row.update({
                "train_loss": t.get("train_loss"),
                "train_peak_mem_gb": t.get("peak_mem_gb"),
                "train_effective_global_bsz": t.get("effective_global_bsz"),
                "global_steps": t.get("global_steps"),
            })

        ev = r / "eval_metrics.json"
        if ev.exists():
            e = load_json(ev)
            row.update({
                "eval_n": e.get("n"),
                "eval_accuracy": e.get("accuracy"),
                "eval_elapsed_s": e.get("elapsed_s"),
            })

        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order
    cols = [
        "run_kind","model","task","gpu","world_size","seq_len","micro_bsz","grad_accum",
        "effective_global_bsz","samples_per_s","tokens_per_s","step_time_s","peak_mem_gb",
        "train_loss","global_steps","train_peak_mem_gb","train_effective_global_bsz",
        "eval_accuracy","eval_n","eval_elapsed_s","run_dir",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    # Minimal markdown view
    out_md = Path(args.out_md)
    lines = []
    lines.append("| kind | model | task | micro_bsz | samples/s | tokens/s | peak_mem_gb | train_loss | eval_acc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("run_kind") not in ("bench","train"):
            continue
        lines.append(
            f"| {r.get('run_kind')} | {r.get('model')} | {r.get('task')} | {r.get('micro_bsz')} | "
            f"{r.get('samples_per_s')} | {r.get('tokens_per_s')} | {r.get('peak_mem_gb') or r.get('train_peak_mem_gb')} | "
            f"{r.get('train_loss')} | {r.get('eval_accuracy')} |"
        )
    out_md.write_text("\n".join(lines))

    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
