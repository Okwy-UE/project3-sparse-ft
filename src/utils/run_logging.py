# src/utils/run_logging.py
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

REGISTRY_DEFAULT = Path("results/run_registry.csv")

def _try(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        return out.strip()
    except Exception as e:
        return f"<err:{e}>"

def snapshot_env(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "timestamp": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": _try(["python", "--version"]),
        "git_rev": _try(["git", "rev-parse", "HEAD"]),
        "git_status": _try(["git", "status", "--porcelain"]),
        "nvidia_smi": _try(["nvidia-smi"]),
        "pip_freeze": _try(["python", "-m", "pip", "freeze"]),
        "env": {k: os.environ.get(k) for k in [
            "SLURM_JOB_ID","SLURM_JOB_NODELIST","SLURM_NTASKS","SLURM_GPUS",
            "CUDA_VISIBLE_DEVICES","HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE",
            "NCCL_DEBUG","NCCL_IB_DISABLE","NCCL_SOCKET_IFNAME"
        ]},
    }
    (run_dir / "env_snapshot.json").write_text(json.dumps(info, indent=2, sort_keys=True))

def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    return json.loads(p.read_text())

def collect_record(run_dir: Path) -> Dict[str, Any]:
    meta = load_json(run_dir / "run_meta.json") or {}
    throughput = load_json(run_dir / "throughput.json") or {}
    train = load_json(run_dir / "train_summary.json") or {}
    evalm = load_json(run_dir / "eval_metrics.json") or {}

    record: Dict[str, Any] = {}
    record.update(meta)

    # Merge known metric payloads (presence depends on run kind)
    for k, v in throughput.items():
        record[f"throughput.{k}"] = v
    for k, v in train.items():
        record[f"train.{k}"] = v
    for k, v in evalm.items():
        record[f"eval.{k}"] = v

    record["run_dir"] = str(run_dir)
    record["registered_at"] = time.time()
    return record

def append_registry(record: Dict[str, Any], registry_path: Path = REGISTRY_DEFAULT) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()

    # Stable column order: union of existing header + new keys
    if exists:
        with registry_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        cols = list(header)
        for k in record.keys():
            if k not in cols:
                cols.append(k)
    else:
        cols = list(record.keys())

    # If expanding columns, rewrite file with new header
    if exists:
        with registry_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for c in cols:
                r.setdefault(c, "")
        for c in cols:
            record.setdefault(c, "")
        with registry_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
            w.writerow(record)
    else:
        for c in cols:
            record.setdefault(c, "")
        with registry_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow(record)

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Append a run_dir into results/run_registry.csv")
    p_reg.add_argument("--run_dir", required=True)
    p_reg.add_argument("--registry", default=str(REGISTRY_DEFAULT))

    p_snap = sub.add_parser("snapshot", help="Write env snapshot into run_dir")
    p_snap.add_argument("--run_dir", required=True)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    if args.cmd == "snapshot":
        snapshot_env(run_dir)
        return

    if args.cmd == "register":
        rec = collect_record(run_dir)
        append_registry(rec, Path(args.registry))
        return

if __name__ == "__main__":
    main()
