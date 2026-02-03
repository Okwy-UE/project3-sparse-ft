import csv
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import fcntl  # linux-only, which is fine for HPC
except Exception:
    fcntl = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _run(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"<error: {e}>"


def get_host_meta() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", ""),
        "cwd": os.getcwd(),
        "time": _now_iso(),
    }


def get_git_meta() -> Dict[str, Any]:
    return {
        "git_commit": _run("git rev-parse HEAD"),
        "git_status": _run("git status --porcelain"),
        "git_branch": _run("git rev-parse --abbrev-ref HEAD"),
    }


def get_gpu_meta() -> Dict[str, Any]:
    # works on DGX nodes
    q = "name,memory.total,driver_version"
    cmd = f"nvidia-smi --query-gpu={q} --format=csv,noheader"
    raw = _run(cmd)
    gpus = []
    if raw and not raw.startswith("<error:"):
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "mem_total": parts[1], "driver": parts[2]})
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": gpus,
        "nvidia_smi": raw,
    }


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_id(prefix: str) -> str:
    # stable + sortable
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def make_run_dir(root: str, run_id: str) -> str:
    d = os.path.join(root, run_id)
    ensure_dir(d)
    return d


def snapshot_env(run_dir: str) -> None:
    meta = {
        "host": get_host_meta(),
        "git": get_git_meta(),
        "gpu": get_gpu_meta(),
        "pip_freeze": _run("python -m pip freeze | sort"),
        "python": _run("python -V"),
        "torch": _run("python - << 'PY'\nimport torch\nprint(torch.__version__)\nprint(torch.version.cuda)\nPY"),
    }
    write_json(os.path.join(run_dir, "env_snapshot.json"), meta)


def _lock_file(f):
    if fcntl is None:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f):
    if fcntl is None:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_run_registry(csv_path: str, row: Dict[str, Any]) -> None:
    """
    Appends a row to results/run_registry.csv without assuming a fixed schema.
    - If the file exists, we preserve existing columns and fill what we can.
    - If new keys appear, we expand the header (rewrite file) safely.
    """
    ensure_dir(os.path.dirname(csv_path))

    # Normalize values to strings for CSV
    row_str = {k: ("" if v is None else str(v)) for k, v in row.items()}

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row_str.keys()))
            w.writeheader()
            w.writerow(row_str)
        return

    # Read existing rows + header, expand if needed
    with open(csv_path, "r", newline="") as f:
        _lock_file(f)
        try:
            r = csv.DictReader(f)
            existing_fields = list(r.fieldnames or [])
            existing_rows = list(r)
        finally:
            _unlock_file(f)

    new_fields = existing_fields[:]
    for k in row_str.keys():
        if k not in new_fields:
            new_fields.append(k)

    # If no schema change, append directly
    if new_fields == existing_fields:
        with open(csv_path, "a", newline="") as f:
            _lock_file(f)
            try:
                w = csv.DictWriter(f, fieldnames=existing_fields)
                w.writerow({k: row_str.get(k, "") for k in existing_fields})
            finally:
                _unlock_file(f)
        return

    # Otherwise rewrite with expanded header
    existing_rows.append({k: row_str.get(k, "") for k in new_fields})
    with open(csv_path, "w", newline="") as f:
        _lock_file(f)
        try:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            for rr in existing_rows:
                w.writerow({k: rr.get(k, "") for k in new_fields})
        finally:
            _unlock_file(f)
