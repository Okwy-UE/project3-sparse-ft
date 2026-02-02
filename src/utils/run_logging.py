from __future__ import annotations
import csv, json, os, platform, subprocess, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""

def git_snapshot(repo_root: str) -> Dict[str, str]:
    return {
        "git_commit": _run(["git", "-C", repo_root, "rev-parse", "HEAD"]),
        "git_status": _run(["git", "-C", repo_root, "status", "--porcelain"]),
    }

def hw_snapshot() -> Dict[str, Any]:
    snap = {
        "host": platform.node(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    snap["nvidia_smi"] = _run(["bash", "-lc", "nvidia-smi -L"])
    snap["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    snap["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
    snap["slurm_nodelist"] = os.environ.get("SLURM_NODELIST", "")
    return snap

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def append_run_registry(
    csv_path: str,
    row: Dict[str, Any],
    fieldnames: Optional[list[str]] = None,
) -> None:
    ensure_dir(os.path.dirname(csv_path))
    exists = os.path.exists(csv_path)
    if fieldnames is None:
        fieldnames = [
            "run_id","date","hardware","system","model","task",
            "method","sparsity","notes","artifact_path"
        ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

@dataclass
class RunMeta:
    run_id: str
    system: str               # e.g., "DGX-H100"
    hardware: str             # e.g., "h100"
    model: str
    task: str
    method: str               # e.g., "dense_lora"
    sparsity: float           # 0 for dense
    notes: str
    artifacts_dir: str

def init_run_dir(repo_root: str, run_id: str) -> str:
    out = os.path.join(repo_root, "results", "runs", run_id)
    ensure_dir(out)
    return out
