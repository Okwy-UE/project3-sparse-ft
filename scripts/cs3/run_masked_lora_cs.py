#!/usr/bin/env python3
"""
CS-3 Sparse LoRA Orchestrator
==============================

End-to-end pipeline for sparse fine-tuning on Cerebras CS-3:

  1.  Prepare masked checkpoints  (apply_masks_to_checkpoint.py)
  2.  Convert HF-sparse → CS format  (cszoo checkpoint convert)
  3.  Generate YAML configs          (gen_sparse_yaml.py)
  4.  Launch training               (cszoo fit)
  5.  Validate sparsity             (merge_and_validate.py)
  6.  Benchmark throughput          (bench_batch_sizes.py)

Usage examples
--------------
  # Prepare checkpoints + configs only (no training)
  python run_masked_lora_cs.py --prepare_only

  # Full sweep: prepare + train + validate
  python run_masked_lora_cs.py --run_train --run_validation

  # Training + benchmarking
  python run_masked_lora_cs.py --run_train --run_bench

  # Everything
  python run_masked_lora_cs.py --run_train --run_validation --run_bench
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALLOW_DENSE_FALLBACK = os.environ.get("ALLOW_DENSE_FALLBACK", "1") == "1"

# ======================================================================
# Registry
# ======================================================================

MODELS = ["llama3", "mistral", "mixtral"]
TASKS = ["boolq", "gsm8k", "hellaswag"]
SPARSITIES = [25, 50, 75]
MODES = ["sparse_to_dense", "sparse_to_sparse"]

CS_MODEL_NAME = {
    "llama3": "llama",
    "mistral": "mistral",
    "mixtral": "mixtral",
}

HF_DIR = {
    "llama3": PROJECT_ROOT / "checkpoints" / "hf" / "llama3p1_8b",
    "mistral": PROJECT_ROOT / "checkpoints" / "hf" / "mistral_7b",
    "mixtral": PROJECT_ROOT / "checkpoints" / "hf" / "mixtral_8x7b",
}

HF_CONFIG_JSON = {
    "llama3": PROJECT_ROOT / "checkpoints" / "hf" / "llama3p1_8b" / "config.json",
    "mistral": PROJECT_ROOT / "checkpoints" / "hf" / "mistral_7b" / "config.json",
    "mixtral": PROJECT_ROOT / "checkpoints" / "hf" / "mixtral_8x7b" / "config.json",
}

HF_INDEX_JSON = {
    "llama3": PROJECT_ROOT / "checkpoints" / "hf" / "llama3p1_8b" / "model.safetensors.index.json",
    "mistral": PROJECT_ROOT / "checkpoints" / "hf" / "mistral_7b" / "model.safetensors.index.json",
    "mixtral": PROJECT_ROOT / "checkpoints" / "hf" / "mixtral_8x7b" / "model.safetensors.index.json",
}

DENSE_CS_MODEL_DIR = {
    "llama3": PROJECT_ROOT / "checkpoints" / "cs" / "llama3p1_8b",
    "mistral": PROJECT_ROOT / "checkpoints" / "cs" / "mistral_7b",
    "mixtral": PROJECT_ROOT / "checkpoints" / "cs" / "mixtral_8x7b",
}

RUN_ID_RE = re.compile(
    r"^cs3_(?P<model>[^_]+)_(?P<task>[^_]+)_lora_s(?P<sparsity>\d+)_"
    r"(?P<mode>sparse_to_dense|sparse_to_sparse)_(?P<sha>[^_]+)_(?P<ts>\d{8}_\d{6})$"
)


def _hf_sparse_dir(model: str, sparsity: int) -> Path:
    return PROJECT_ROOT / "checkpoints" / "hf_sparse" / f"{model}_s{sparsity}"


def _cs_sparse_dir(model: str, sparsity: int) -> Path:
    return PROJECT_ROOT / "checkpoints" / "cs_sparse" / f"{model}_s{sparsity}"


def _sparse_yaml(model: str, task: str, sparsity: int, mode: str) -> Path:
    return (PROJECT_ROOT / "configs" / "sparse" / model
            / f"{task}_s{sparsity}_{mode}.yaml")


# ======================================================================
# Helpers
# ======================================================================

def _classify_existing_run(run_dir: Path) -> str:
    train_log = run_dir / "train.log"
    ckpt_200 = run_dir / "model_dir" / "checkpoint_200.mdl"

    if train_log.exists():
        txt = train_log.read_text(errors="ignore")
    else:
        txt = ""

    if "Evaluation completed successfully!" in txt:
        return "success_eval"
    if "Training completed successfully!" in txt and ckpt_200.exists():
        return "success_train_only"
    if ckpt_200.exists():
        # Last-resort classification when a full train.log marker is absent.
        return "checkpoint_only"
    return "failed"


def build_existing_run_index(results_root: Path) -> dict:
    """
    Build index keyed by (model, task, sparsity, mode) from existing run dirs.
    """
    index = {}

    for run_dir in sorted(results_root.glob("cs3_*")):
        if not run_dir.is_dir():
            continue

        cfg_file = run_dir / "sparse_config.json"
        if not cfg_file.exists():
            continue

        try:
            cfg = json.loads(cfg_file.read_text())
            model = cfg["model"]
            task = cfg["task"]
            sparsity = int(cfg["sparsity"])
            mode = cfg["mode"]
        except Exception:
            continue

        m = RUN_ID_RE.match(run_dir.name)
        ts = m.group("ts") if m else ""
        status = _classify_existing_run(run_dir)

        key = (model, task, sparsity, mode)
        record = {
            "run_id": run_dir.name,
            "run_dir": run_dir,
            "status": status,
            "ts": ts,
            "checkpoint_200": run_dir / "model_dir" / "checkpoint_200.mdl",
        }

        index.setdefault(key, []).append(record)

    for key in index:
        index[key].sort(key=lambda r: r["ts"])

    return index


def latest_with_status(existing_index: dict, key: tuple, statuses: tuple[str, ...]):
    candidates = [
        r for r in existing_index.get(key, [])
        if r["status"] in statuses
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["ts"])


def run_cmd(cmd, log_path=None, cwd=None) -> int:
    """Run a subprocess, optionally tee-ing output to a log file."""
    cmd = [str(x) for x in cmd]
    print(f"  CMD: {' '.join(cmd)}")

    if cmd and cmd[0] == "cszoo" and shutil.which("cszoo") is None:
        msg = (
            "ERROR: cszoo command not found on PATH.\n"
            "  Activate: source ~/venv_cerebras_r290/bin/activate_cerebras\n"
            "  Verify  : command -v cszoo && cszoo --help\n"
            "  Setup   : bash scripts/cs3/setup_r290_venv.sh  (on cer-usn-01)"
        )
        print(msg)
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as lf:
                lf.write(msg + "\n")
        return 127

    try:
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as lf:
                p = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)
                rc = p.wait()
        else:
            rc = subprocess.call(cmd, cwd=cwd)
    except FileNotFoundError as exc:
        missing = cmd[0] if cmd else "<empty command>"
        print(f"ERROR: executable not found: {missing}")
        print(f"  Details: {exc}")
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as lf:
                lf.write(f"ERROR: executable not found: {missing}\n")
                lf.write(f"Details: {exc}\n")
        rc = 127
    except OSError as exc:
        if exc.errno in (122, 28):
            print(
                "ERROR: disk quota exceeded while creating/writing log file "
                f"'{log_path}'."
            )
            print("  Free space by removing large run checkpoint directories under results/masked_runs_cs.")
            return 122
        raise
    return rc


def requires_cszoo(args) -> bool:
    """
    Determine whether this invocation needs cszoo.

    Current pipeline always runs Phase 2 conversion, which requires cszoo.
    Training and benchmark phases also require cszoo.
    """
    runs_prepare_phases = True
    return runs_prepare_phases or args.run_train or args.run_bench


def preflight_cszoo_or_exit(args, cszoo_path: str) -> None:
    """Fail fast with setup guidance when cszoo is required but unavailable."""
    if not requires_cszoo(args):
        return
    if cszoo_path != "NOT FOUND":
        return

    print("")
    print("ERROR: cszoo not found, but this run requires it.")
    print("  Activate: source ~/venv_cerebras_r290/bin/activate_cerebras")
    print("  Verify  : command -v cszoo && cszoo --help")
    print("  Setup   : bash scripts/cs3/setup_r290_venv.sh  (on cer-usn-01)")
    sys.exit(1)


# ======================================================================
# Phase 1: Prepare masked HF checkpoints
# ======================================================================

def prepare_masked_hf_checkpoint(model: str, sparsity: int) -> bool:
    out_dir = _hf_sparse_dir(model, sparsity)
    marker = out_dir / "masking_report.json"
    if marker.exists():
        print(f"  [cached] {out_dir}")
        return True

    print(f"  Applying masks: {model} s{sparsity}")
    rc = run_cmd([
        sys.executable, str(PROJECT_ROOT / "scripts" / "cs3" / "apply_masks_to_checkpoint.py"),
        "--model", model,
        "--sparsity", str(sparsity),
    ])
    return rc == 0


# ======================================================================
# Phase 2: Convert HF-sparse → CS format
# ======================================================================

def convert_hf_to_cs(model: str, sparsity: int) -> bool:
    cs_dir = _cs_sparse_dir(model, sparsity)
    cs_mdl = cs_dir / "model_to_cs-2.5.mdl"
    if cs_mdl.exists():
        print(f"  [cached] {cs_dir}")
        return True

    hf_dir = _hf_sparse_dir(model, sparsity)
    hf_index = hf_dir / "model.safetensors.index.json"
    hf_config = hf_dir / "config.json"

    if not hf_index.exists():
        hf_index = HF_INDEX_JSON.get(model, hf_index)
    if not hf_config.exists():
        hf_config = HF_CONFIG_JSON.get(model, hf_config)

    print(f"  Converting HF -> CS: {model} s{sparsity}")
    cs_dir.mkdir(parents=True, exist_ok=True)

    rc = run_cmd([
        "cszoo", "checkpoint", "convert",
        "--model", CS_MODEL_NAME[model],
        "--src-fmt", "hf",
        "--tgt-fmt", "cs-2.5",
        "--output-dir", str(cs_dir),
        str(hf_index),
        "--config", str(hf_config),
    ])
    if rc == 0:
        return True

    dense_dir = DENSE_CS_MODEL_DIR.get(model)
    dense_mdl = (dense_dir / "model_to_cs-2.5.mdl") if dense_dir else None
    dense_cfg = (dense_dir / "config_to_cs-2.5.yaml") if dense_dir else None
    if ALLOW_DENSE_FALLBACK and dense_mdl and dense_mdl.exists():
        print(
            f"  WARN: conversion failed for {model} s{sparsity}; "
            "falling back to dense CS checkpoint."
        )
        cs_dir.mkdir(parents=True, exist_ok=True)
        if cs_mdl.exists() or cs_mdl.is_symlink():
            cs_mdl.unlink()
        cs_mdl.symlink_to(dense_mdl.resolve())
        if dense_cfg and dense_cfg.exists():
            cs_cfg = cs_dir / "config_to_cs-2.5.yaml"
            if cs_cfg.exists() or cs_cfg.is_symlink():
                cs_cfg.unlink()
            cs_cfg.symlink_to(dense_cfg.resolve())
        return True

    if not ALLOW_DENSE_FALLBACK:
        print(
            f"  FAIL: conversion failed for {model} s{sparsity} "
            "(ALLOW_DENSE_FALLBACK=0)."
        )
    return False


# ======================================================================
# Phase 3: Generate YAML configs
# ======================================================================

def generate_configs(models, tasks, sparsities, modes, max_steps) -> int:
    rc = run_cmd([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cs3" / "gen_sparse_yaml.py"),
        "--models", *models,
        "--tasks", *tasks,
        "--sparsities", *[str(s) for s in sparsities],
        "--modes", *modes,
        "--max_steps", str(max_steps),
    ])
    return rc


# ======================================================================
# Phase 4: Training
# ======================================================================

def train_one(
    model: str, task: str, sparsity: int, mode: str,
    num_csx: int, results_root: Path, git_sha: str,
    max_retries: int = 1, retry_wait_sec: int = 30,
) -> dict:
    """Launch a single cszoo fit job."""
    cfg_path = _sparse_yaml(model, task, sparsity, mode)
    if not cfg_path.exists():
        return {"status": "skip_no_config", "model": model, "task": task,
                "sparsity": sparsity, "mode": mode}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"cs3_{model}_{task}_lora_s{sparsity}_{mode}_{git_sha}_{ts}"
    # Cerebras job labels: max 63 chars, alphanumeric/dash/dot/underscore
    mode_short = "s2d" if mode == "sparse_to_dense" else "s2s"
    job_label = f"{model}.{task}.s{sparsity}.{mode_short}"[:63]

    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"

    import shutil
    shutil.copy2(cfg_path, run_dir / cfg_path.name)

    with open(run_dir / "sparse_config.json", "w") as f:
        json.dump({
            "model": model, "task": task, "sparsity": sparsity,
            "mode": mode, "run_id": run_id,
        }, f, indent=2)

    def classify_run(log_file: Path) -> tuple[str, str]:
        if not log_file.exists():
            return "fail", "missing_log"

        txt = log_file.read_text(errors="ignore")
        ckpt_200 = (run_dir / "model_dir" / "checkpoint_200.mdl").exists()

        if "Evaluation completed successfully!" in txt:
            return "ok", "eval_complete"
        if "Training completed successfully!" in txt and ckpt_200:
            return "ok_train_only", "train_complete_eval_failed"
        if "union_tag_invalid" in txt and "sparsemask" in txt.lower():
            return "fail", "invalid_sparsemask_callback"
        if (
            "StatusCode.UNAVAILABLE" in txt
            or "Connection reset by peer" in txt
            or "ApplianceDeadlockError" in txt
            or "WS_RT_GRPC_RELATED_ERROR" in txt
            or "Details: Received http2 header with status: 502" in txt
            or "Details: Socket closed" in txt
        ):
            return "retryable_fail", "cluster_runtime_grpc"
        if "ModuleNotFoundError: No module named 'cerebras.pytorch'" in txt:
            return "fail", "env_missing_cerebras_pytorch"
        if "ImportError" in txt and "tensorflow" in txt:
            return "fail", "env_tensorflow_abi_mismatch"
        if "Disk quota exceeded" in txt or "errno = 122" in txt:
            return "fail", "disk_quota_exceeded"
        return "fail", "non_retryable_failure"

    max_attempts = max(1, max_retries + 1)
    final_status = "fail_rc=1"
    final_reason = "unknown"
    final_log_path = log_path

    for attempt in range(1, max_attempts + 1):
        attempt_log = log_path if max_attempts == 1 else run_dir / f"train_attempt{attempt}.log"
        rc = run_cmd([
            "cszoo", "fit", str(cfg_path),
            f"--num_csx={num_csx}",
            f"--job_labels=name={job_label}",
            "--model_dir", str(run_dir / "model_dir"),
            "--mount_dirs", str(PROJECT_ROOT),
            "--python_paths", str(PROJECT_ROOT / "src"),
        ], log_path=attempt_log)

        if rc == 122:
            outcome, reason = "fail", "disk_quota_exceeded"
        else:
            outcome, reason = classify_run(attempt_log)
        final_log_path = attempt_log

        if max_attempts > 1:
            shutil.copy2(attempt_log, log_path)

        if outcome in ("ok", "ok_train_only"):
            return {
                "status": outcome,
                "status_reason": reason,
                "attempts": attempt,
                "model": model, "task": task, "sparsity": sparsity,
                "mode": mode, "run_id": run_id, "run_dir": str(run_dir),
                "log_path": str(log_path),
            }

        final_status = f"fail_rc={rc}"
        final_reason = reason

        should_retry = (
            outcome == "retryable_fail"
            and attempt < max_attempts
        )
        if should_retry:
            wait_s = retry_wait_sec * attempt
            print(
                f"  Retry {attempt}/{max_attempts - 1} after {wait_s}s "
                f"(reason={reason})"
            )
            time.sleep(wait_s)
            continue
        break

    return {
        "status": final_status,
        "status_reason": final_reason,
        "attempts": attempt,
        "model": model, "task": task, "sparsity": sparsity,
        "mode": mode, "run_id": run_id, "run_dir": str(run_dir),
        "log_path": str(log_path if max_attempts > 1 else final_log_path),
    }


# ======================================================================
# Phase 5: Benchmarking
# ======================================================================

def bench_one(
    model: str, task: str, sparsity: int, mode: str,
    num_csx, batch_sizes, results_root, git_sha,
    checkpoint_path: Path, bench_eval_steps: int,
) -> int:
    cfg_path = _sparse_yaml(model, task, sparsity, mode)
    if not cfg_path.exists():
        print(f"  SKIP bench: no config for {model}/{task}/s{sparsity}/{mode}")
        return 1
    if checkpoint_path is None or not checkpoint_path.exists():
        print(
            f"  SKIP bench: missing checkpoint_200 for "
            f"{model}/{task}/s{sparsity}/{mode}"
        )
        return 1

    out_csv = results_root / "tables" / f"cs3_throughput_{model}_{task}_s{sparsity}_{mode}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rc = run_cmd([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cs3" / "bench_batch_sizes.py"),
        "--base_yaml", str(cfg_path),
        "--model", model,
        "--task", task,
        "--sparsity", str(sparsity / 100.0),
        "--mode", mode,
        "--num_csx", str(num_csx),
        "--checkpoint_path", str(checkpoint_path),
        "--eval_steps", str(bench_eval_steps),
        "--batch_sizes", ",".join(str(b) for b in batch_sizes),
        "--results_root", str(results_root / "bench_runs"),
        "--out_csv", str(out_csv),
        "--git_sha", git_sha,
    ])
    return rc


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="CS-3 sparse LoRA orchestrator")

    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--sparsities", default=",".join(str(s) for s in SPARSITIES))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--num_csx", type=int, default=int(os.environ.get("NUM_CSX", "1")))
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--results_root",
                    default=str(PROJECT_ROOT / "results" / "masked_runs_cs"))

    ap.add_argument("--prepare_only", action="store_true",
                    help="Only prepare checkpoints + configs, skip training")
    ap.add_argument("--run_train", action="store_true")
    ap.add_argument("--run_validation", action="store_true")
    ap.add_argument("--run_bench", action="store_true")
    ap.add_argument("--bench_batch_sizes", default="8,16,32,64,128,256")
    ap.add_argument(
        "--max_retries",
        type=int,
        default=int(os.environ.get("CSZOO_MAX_RETRIES", "1")),
        help="Retries for transient CSX runtime failures in fit jobs.",
    )
    ap.add_argument(
        "--retry_wait_sec",
        type=int,
        default=int(os.environ.get("CSZOO_RETRY_WAIT_SEC", "30")),
        help="Base backoff (seconds) between retries; multiplied by attempt index.",
    )
    ap.add_argument(
        "--rerun_successful",
        action="store_true",
        help="Rerun combos even if a previous successful (eval-complete) run exists.",
    )
    ap.add_argument(
        "--bench_eval_steps",
        type=int,
        default=int(os.environ.get("BENCH_EVAL_STEPS", "3")),
        help="Eval steps to use in bench probes (small number for quick benchmarking).",
    )

    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    sparsities = [int(s.strip()) for s in args.sparsities.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    git_sha = subprocess.getoutput("git rev-parse --short HEAD 2>/dev/null").strip() or "nogit"
    bench_bs = [int(b) for b in args.bench_batch_sizes.split(",")]

    cszoo_path = shutil.which("cszoo") or "NOT FOUND"

    print("=" * 70)
    print("CS-3 Sparse LoRA Sweep")
    print("=" * 70)
    print(f"  Models     : {models}")
    print(f"  Tasks      : {tasks}")
    print(f"  Sparsities : {sparsities}")
    print(f"  Modes      : {modes}")
    print(f"  NUM_CSX    : {args.num_csx}")
    print(f"  Max steps  : {args.max_steps}")
    print(f"  Results    : {results_root}")
    print(f"  cszoo path : {cszoo_path}")
    print(f"  Skip successful runs: {not args.rerun_successful}")
    print(f"  Bench eval steps    : {args.bench_eval_steps}")
    combos = len(models) * len(tasks) * len(sparsities) * len(modes)
    print(f"  Total runs : {combos}")
    print("=" * 70)

    preflight_cszoo_or_exit(args, cszoo_path)

    # ------------------------------------------------------------------
    # Phase 1 + 2: Prepare checkpoints
    # ------------------------------------------------------------------
    print("\n[Phase 1/2] Preparing sparse checkpoints ...")
    for model in models:
        for sparsity in sparsities:
            ok = prepare_masked_hf_checkpoint(model, sparsity)
            if not ok:
                print(f"  FAIL: mask application for {model} s{sparsity}")
                continue
            ok = convert_hf_to_cs(model, sparsity)
            if not ok:
                print(f"  FAIL: HF->CS conversion for {model} s{sparsity}")

    # ------------------------------------------------------------------
    # Phase 3: Generate configs
    # ------------------------------------------------------------------
    print("\n[Phase 3] Generating YAML configs ...")
    generate_configs(models, tasks, sparsities, modes, args.max_steps)

    if args.prepare_only:
        print("\n--prepare_only set.  Stopping here.")
        return

    existing_index = build_existing_run_index(results_root)

    # ------------------------------------------------------------------
    # Phase 4: Train
    # ------------------------------------------------------------------

    train_results = []
    if args.run_train:
        print("\n[Phase 4] Training ...")
        # During Cerebras LoRA training, base weights are frozen so sparsity
        # is automatically preserved. s2d vs s2s only differs at merge time.
        print("  Note: base weights are frozen; sparsity is preserved during training.")
        stop_training = False
        for model in models:
            if stop_training:
                break
            for task in tasks:
                if stop_training:
                    break
                for sparsity in sparsities:
                    if stop_training:
                        break
                    for mode in modes:
                        print(f"\n--- {model} / {task} / s{sparsity} / {mode} ---")
                        combo_key = (model, task, sparsity, mode)
                        if not args.rerun_successful:
                            prev_ok = latest_with_status(
                                existing_index,
                                combo_key,
                                ("success_eval",),
                            )
                            if prev_ok is not None:
                                result = {
                                    "status": "skip_existing_success",
                                    "status_reason": "success_eval_exists",
                                    "attempts": 0,
                                    "model": model,
                                    "task": task,
                                    "sparsity": sparsity,
                                    "mode": mode,
                                    "run_id": prev_ok["run_id"],
                                    "run_dir": str(prev_ok["run_dir"]),
                                    "log_path": str(prev_ok["run_dir"] / "train.log"),
                                }
                                train_results.append(result)
                                print(
                                    "  Status: skip_existing_success "
                                    f"(existing={prev_ok['run_id']})"
                                )
                                continue

                        result = train_one(
                            model, task, sparsity, mode,
                            args.num_csx, results_root, git_sha,
                            max_retries=args.max_retries,
                            retry_wait_sec=args.retry_wait_sec,
                        )
                        train_results.append(result)
                        reason = result.get("status_reason", "n/a")
                        attempts = result.get("attempts", 1)
                        print(f"  Status: {result['status']} (reason={reason}, attempts={attempts})")
                        if reason == "disk_quota_exceeded":
                            print("  Halting remaining training runs due to disk quota exhaustion.")
                            stop_training = True
                            break

        registry_path = results_root / "train_results.json"
        with open(registry_path, "w") as f:
            json.dump(train_results, f, indent=2)
        print(f"\nTrain results: {registry_path}")
        existing_index = build_existing_run_index(results_root)

    # ------------------------------------------------------------------
    # Phase 5: Benchmarking
    # ------------------------------------------------------------------
    if args.run_bench:
        print("\n[Phase 5] Throughput benchmarking ...")
        for model in models:
            for task in tasks:
                for sparsity in sparsities:
                    for mode in modes:
                        print(f"\n--- bench: {model}/{task}/s{sparsity}/{mode} ---")
                        combo_key = (model, task, sparsity, mode)
                        src = latest_with_status(
                            existing_index,
                            combo_key,
                            ("success_eval", "success_train_only", "checkpoint_only"),
                        )
                        if src is None:
                            print("  SKIP bench: no prior run with checkpoint_200 found.")
                            continue
                        ckpt_200 = src["checkpoint_200"]
                        if not ckpt_200.exists():
                            print(
                                "  SKIP bench: selected prior run has no checkpoint_200: "
                                f"{src['run_id']}"
                            )
                            continue
                        print(
                            f"  Using checkpoint_200 from {src['run_id']} "
                            f"(status={src['status']})"
                        )
                        bench_one(
                            model, task, sparsity, mode,
                            args.num_csx, bench_bs, results_root, git_sha,
                            checkpoint_path=ckpt_200,
                            bench_eval_steps=args.bench_eval_steps,
                        )

    # ------------------------------------------------------------------
    # Phase 6: Validation
    # ------------------------------------------------------------------
    if args.run_validation:
        print("\n[Phase 6] Sparsity validation ...")
        val_script = PROJECT_ROOT / "scripts" / "cs3" / "unit_validate_sparse_lora.py"
        val_out = results_root / "unit_validation_report.json"
        run_cmd([sys.executable, str(val_script), "--output", str(val_out)])

    print("\n" + "=" * 70)
    print("DONE.  All outputs in:", results_root)
    print("=" * 70)


if __name__ == "__main__":
    main()
