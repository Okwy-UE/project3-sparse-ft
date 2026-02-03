import glob
import json
import os
from collections import defaultdict

RUNS_GLOB = "results/runs/week4-gpu-dense-*/run_result.json"

def load_run_results():
    rows = []
    for p in glob.glob(RUNS_GLOB):
        with open(p, "r") as f:
            r = json.load(f)
        run_dir = os.path.dirname(p)
        spec_path = os.path.join(run_dir, "run_spec.json")
        spec = {}
        if os.path.exists(spec_path):
            with open(spec_path, "r") as f:
                spec = json.load(f)

        rows.append({
            "run_dir": run_dir,
            "model_id": r.get("model_id"),
            "model_alias": spec.get("model_alias", ""),
            "task": r.get("task"),
            "seq_len": r.get("seq_len"),
            "micro_batch": r.get("micro_batch_final"),
            "train_tok_s": r.get("train_tokens_per_s"),
            "train_time_s": r.get("train_time_s"),
            "loss_last": r.get("loss_last"),
            "tp_points": r.get("throughput_points", []),
            "lm_eval": r.get("lm_eval", {}),
        })
    return rows

def format_tp(points):
    # show as "bs: tok/s" list
    out = []
    for p in points:
        out.append(f'{p.get("micro_batch")}:{p.get("tokens_per_s"):.0f}')
    return ", ".join(out)

def extract_metric(lm_eval_blob):
    # lm-eval format: {"results": {"task": {...}}}
    try:
        res = lm_eval_blob.get("results", {})
        if not res:
            return ("", "")
        task_name = list(res.keys())[0]
        metrics = res[task_name]
        # pick first scalar metric
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                return (k, v)
        return (task_name, "")
    except Exception:
        return ("", "")

def main():
    rows = load_run_results()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/notes", exist_ok=True)

    # ---- H100 baseline table
    lines = []
    lines.append("# Week 4 GPU Dense Baselines (H100/H200)\n")
    lines.append("| Model | Task | Seq | MicroBS | Train tok/s | Train time (s) | Loss(last) | Eval metric | Eval | TP scaling (bs:tok/s) | Run dir |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---|---|")
    for r in sorted(rows, key=lambda x: (x["model_alias"], x["task"])):
        mname, mval = extract_metric(r["lm_eval"])
        tp = format_tp(r["tp_points"])
        lines.append(
            f'| {r["model_alias"]} | {r["task"]} | {r["seq_len"]} | {r["micro_batch"]} | '
            f'{(r["train_tok_s"] or 0):.0f} | {(r["train_time_s"] or 0):.0f} | {(r["loss_last"] or 0):.4f} | '
            f'{mname} | {mval if mval != "" else ""} | {tp} | {r["run_dir"]} |'
        )

    with open("results/tables/week4_h100_dense_baselines.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- Hyperparam alignment notes stub (auto-generated)
    notes = []
    notes.append("# Week 4 Hyperparameter Alignment Notes\n")
    notes.append("- Fixed: max_seq_len=2048, LoRA baseline, bf16, cosine schedule + warmup_ratio.\n")
    notes.append("- Variable: micro-batch (auto-found by OOM probing), DeepSpeed ZeRO settings (needed esp. for Mixtral).\n")
    notes.append("- Eval: lm-eval-harness with batch_size=auto; adapter evaluated via `peft=` model_args.\n")
    with open("results/notes/week4_hyperparam_alignment.md", "w") as f:
        f.write("\n".join(notes) + "\n")

    print("[OK] wrote results/tables/week4_h100_dense_baselines.md")
    print("[OK] wrote results/notes/week4_hyperparam_alignment.md")

if __name__ == "__main__":
    main()
