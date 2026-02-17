#!/usr/bin/env python3
import argparse, copy, csv, os, re, subprocess, sys, time
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

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
    ap.add_argument("--base_yaml", required=True, help="Base trainer YAML (CSX)")
    ap.add_argument("--model", required=True, help="llama3 | mistral | mixtral (label)")
    ap.add_argument("--task", required=True, help="boolq | hellaswag | gsm8k (label)")
    ap.add_argument("--num_csx", type=int, default=int(os.environ.get("NUM_CSX", "2")))
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_sizes", default="8,16,32,64,128,256,512")
    ap.add_argument("--warmup_eval_steps", type=int, default=3, help="extra eval steps before measuring")
    ap.add_argument("--results_root", default="results/runs")
    ap.add_argument("--out_csv", default="results/tables/cs3_throughput_sweep.csv")
    ap.add_argument("--git_sha", default=os.environ.get("GIT_SHA", "nogit"))
    args = ap.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.base_yaml, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Minimal “probe mode” patch knobs
    # - 1 train step (so fit path works reliably)
    # - short eval loop, measure last eval step
    # - log every step
    def make_probe_cfg(bs):
        cfg = copy.deepcopy(base_cfg)

        deep_set(cfg, ["trainer","fit","train_dataloader","batch_size"], bs)
        # val_dataloader is a list in your YAML
        vdl = deep_get(cfg, ["trainer","fit","val_dataloader"])
        if isinstance(vdl, list) and len(vdl) > 0:
            vdl[0]["batch_size"] = bs

        deep_set(cfg, ["trainer","init","logging","log_steps"], 1)
        deep_set(cfg, ["trainer","init","checkpoint","steps"], 1000000)
        deep_set(cfg, ["trainer","init","loop","max_steps"], 1)
        deep_set(cfg, ["trainer","init","loop","eval_frequency"], 1)
        deep_set(cfg, ["trainer","init","loop","eval_steps"], args.warmup_eval_steps + 1)

        # Make sure we aren’t artificially throttling microbatching
        # Keep change minimal: use auto
        cbs = deep_get(cfg, ["trainer","init","callbacks"])
        for cb in cbs:
            if "ScopedTrainFlags" in cb and "csx.performance.micro_batch_size" in cb["ScopedTrainFlags"]:
                cb["ScopedTrainFlags"]["csx.performance.micro_batch_size"] = "auto"
            if "ScopedValidateFlags" in cb and "csx.performance.micro_batch_size" in cb["ScopedValidateFlags"]:
                cb["ScopedValidateFlags"]["csx.performance.micro_batch_size"] = "auto"

        return cfg

    fieldnames = [
        "timestamp","system","model","task","seq_len","num_csx","batch_size",
        "global_rate_samples_s","tokens_s","status","run_id","log_path","yaml_path"
    ]
    write_header = not os.path.exists(args.out_csv)

    with open(args.out_csv, "a", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        best = {"batch_size": None, "global_rate": -1.0}

        for bs in batch_sizes:
            ts = datetime.now().isoformat()
            run_id = f"cs3_bench_{args.model}_{args.task}_bs{bs}_{args.git_sha}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = os.path.join(args.results_root, run_id)
            os.makedirs(out_dir, exist_ok=True)

            probe_yaml_path = os.path.join(out_dir, "probe.yaml")
            log_path = os.path.join(out_dir, "train.log")

            probe_cfg = make_probe_cfg(bs)
            with open(probe_yaml_path, "w") as yf:
                yaml.safe_dump(probe_cfg, yf, sort_keys=False)

            cmd = [
                "cszoo","fit", probe_yaml_path,
                f"--num_csx={args.num_csx}",
                f"--job_labels=name={run_id}",
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
                "seq_len": args.seq_len,
                "num_csx": args.num_csx,
                "batch_size": bs,
                "global_rate_samples_s": f"{rate:.4f}" if rate is not None else "",
                "tokens_s": f"{tokens_s:.2f}" if tokens_s is not None else "",
                "status": status,
                "run_id": run_id,
                "log_path": log_path,
                "yaml_path": probe_yaml_path,
            })
            cf.flush()

        print(f"[bench] DONE. Results appended to: {args.out_csv}")
        if best["batch_size"] is not None:
            print(f"[bench] BEST batch_size={best['batch_size']}  GlobalRate={best['global_rate']:.4f} samples/s")
        else:
            print("[bench] No successful runs; check logs.")

if __name__ == "__main__":
    main()
