#!/usr/bin/env python3
"""
Throughput benchmarking for CS-3 sparse LoRA runs.

Sweeps over batch sizes, runs a short validation probe against a trained
``checkpoint_200.mdl`` via ``cszoo validate``, and parses eval GlobalRate
from the log.

Outputs a CSV row per batch size for downstream analysis.
"""

import argparse
import copy
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    import yaml
except ImportError:
    sys.exit("Missing PyYAML. Install with: pip install pyyaml")

EVAL_LINE_RE = re.compile(
    r"Eval .*?GlobalRate=([0-9]*\.?[0-9]+)\s+samples/sec"
)


def deep_get(d, path):
    cur = d
    for k in path:
        cur = cur[k]
    return cur


def deep_set(d, path, value):
    cur = d
    for k in path[:-1]:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def parse_last_global_rate(log_path):
    last = None
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = EVAL_LINE_RE.search(line)
            if m:
                last = float(m.group(1))
    return last


def run_cmd(cmd, log_path, env=None):
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_yaml", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--mode", default="sparse_to_dense",
                    choices=["sparse_to_dense", "sparse_to_sparse"])
    ap.add_argument("--num_csx", type=int, default=int(os.environ.get("NUM_CSX", "1")))
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_sizes", default="8,16,32,64,128,256,512")
    ap.add_argument("--eval_steps", type=int, default=3)
    ap.add_argument("--results_root", default="results/runs")
    ap.add_argument("--out_csv", default="results/tables/cs3_throughput_sweep.csv")
    ap.add_argument("--git_sha", default=os.environ.get("GIT_SHA", "nogit"))
    args = ap.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.base_yaml) as f:
        base_cfg = yaml.safe_load(f)

    def make_probe_cfg(bs):
        cfg = copy.deepcopy(base_cfg)
        deep_set(cfg, ["trainer", "fit", "train_dataloader", "batch_size"], bs)

        # Update all validation dataloaders to probe this batch size.
        try:
            fit_vdl = deep_get(cfg, ["trainer", "fit", "val_dataloader"])
            if isinstance(fit_vdl, list) and fit_vdl:
                fit_vdl[0]["batch_size"] = bs
        except Exception:
            pass

        try:
            validate_vdl = deep_get(cfg, ["trainer", "validate", "val_dataloader"])
            if isinstance(validate_vdl, dict):
                validate_vdl["batch_size"] = bs
        except Exception:
            pass

        try:
            validate_all_vdls = deep_get(cfg, ["trainer", "validate_all", "val_dataloaders"])
            if isinstance(validate_all_vdls, list) and validate_all_vdls:
                validate_all_vdls[0]["batch_size"] = bs
        except Exception:
            pass

        deep_set(cfg, ["trainer", "init", "logging", "log_steps"], 1)
        deep_set(cfg, ["trainer", "init", "checkpoint", "steps"], 1_000_000)
        deep_set(cfg, ["trainer", "init", "loop", "eval_steps"], args.eval_steps)
        cbs = deep_get(cfg, ["trainer", "init", "callbacks"])
        if isinstance(cbs, list):
            for cb in cbs:
                if isinstance(cb, dict):
                    for flag_key in ("ScopedTrainFlags", "ScopedValidateFlags"):
                        if flag_key in cb and "csx.performance.micro_batch_size" in cb[flag_key]:
                            cb[flag_key]["csx.performance.micro_batch_size"] = "auto"
        return cfg

    fieldnames = [
        "timestamp", "system", "model", "task", "sparsity", "mode",
        "seq_len", "num_csx", "batch_size",
        "global_rate_samples_s", "tokens_s",
        "status", "run_id", "log_path", "yaml_path",
    ]
    write_header = not os.path.exists(args.out_csv)

    with open(args.out_csv, "a", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        best = {"batch_size": None, "global_rate": -1.0}

        for bs in batch_sizes:
            ts = datetime.now().isoformat()
            run_id = (f"cs3_bench_{args.model}_{args.task}"
                      f"_s{int(args.sparsity*100)}_{args.mode}"
                      f"_bs{bs}_{args.git_sha}"
                      f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            out_dir = os.path.join(args.results_root, run_id)
            os.makedirs(out_dir, exist_ok=True)

            probe_yaml_path = os.path.join(out_dir, "probe.yaml")
            log_path = os.path.join(out_dir, "train.log")

            probe_cfg = make_probe_cfg(bs)
            with open(probe_yaml_path, "w") as yf:
                yaml.safe_dump(probe_cfg, yf, sort_keys=False)

            mode_short = "s2d" if args.mode == "sparse_to_dense" else "s2s"
            job_label = f"bench.{args.model}.{args.task}.bs{bs}.{mode_short}"[:63]
            cmd = [
                "cszoo", "validate", probe_yaml_path,
                f"--num_csx={args.num_csx}",
                f"--job_labels=name={job_label}",
                "--checkpoint_path", args.checkpoint_path,
                "--model_dir", os.path.join(out_dir, "model_dir"),
                "--mount_dirs", os.getcwd(),
                "--python_paths", os.path.join(os.getcwd(), "src"),
            ]

            rc = run_cmd(cmd, log_path)
            rate = parse_last_global_rate(log_path) if rc == 0 else None

            status = "ok" if (rc == 0 and rate is not None) else f"fail_rc={rc}"
            tokens_s = (rate * args.seq_len) if rate is not None else None

            if rate is not None and rate > best["global_rate"]:
                best = {"batch_size": bs, "global_rate": rate}

            w.writerow({
                "timestamp": ts,
                "system": "CS-3",
                "model": args.model,
                "task": args.task,
                "sparsity": args.sparsity,
                "mode": args.mode,
                "seq_len": args.seq_len,
                "num_csx": args.num_csx,
                "batch_size": bs,
                "global_rate_samples_s": f"{rate:.4f}" if rate else "",
                "tokens_s": f"{tokens_s:.2f}" if tokens_s else "",
                "status": status,
                "run_id": run_id,
                "log_path": log_path,
                "yaml_path": probe_yaml_path,
            })
            cf.flush()

        print(f"[bench] DONE. Results appended to: {args.out_csv}")
        if best["batch_size"] is not None:
            print(f"[bench] BEST batch_size={best['batch_size']}  "
                  f"GlobalRate={best['global_rate']:.4f} samples/s")
        else:
            print("[bench] No successful runs; check logs.")


if __name__ == "__main__":
    main()
