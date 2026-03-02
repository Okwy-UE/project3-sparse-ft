#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys

import yaml


DEFAULT_MODELS = ["llama3.1-8b", "mistral-7b"]


def load_tasks_from_contract(contract_path: str) -> list[str]:
    with open(contract_path, "r") as f:
        obj = yaml.safe_load(f)
    task_steps = obj.get("max_steps", {})
    if not isinstance(task_steps, dict) or not task_steps:
        raise ValueError(f"Could not read task list from contract: {contract_path}")
    return list(task_steps.keys())


def has_completed_run(
    registry_csv: str,
    model: str,
    task: str,
    mask_sparsity: int,
    sparse_mode: str,
) -> bool:
    if not os.path.exists(registry_csv):
        return False

    try:
        with open(registry_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("system") != "gpu":
                    continue
                if row.get("model") != model:
                    continue
                if row.get("task") != task:
                    continue
                if row.get("sparse_mode", "none") != sparse_mode:
                    continue
                if row.get("mask_sparsity", "") != str(mask_sparsity):
                    continue
                run_dir = row.get("run_dir", "")
                if run_dir and os.path.exists(os.path.join(run_dir, "run_result.json")):
                    return True
    except Exception:
        return False

    return False


def main():
    ap = argparse.ArgumentParser(
        description="Run masked week4 GPU pretraining/eval sweeps for llama+mistral."
    )
    ap.add_argument("--contract", default="configs/contracts/week4_dense_h100.yaml")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="+", default=None, help="Defaults to tasks from contract.max_steps keys.")
    ap.add_argument("--sparsities", nargs="+", type=int, default=[25, 50, 75], choices=[25, 50, 75])
    ap.add_argument("--deepspeed", default="off", choices=["auto", "on", "off"])
    ap.add_argument("--runs_root", default="results/runs")
    ap.add_argument("--registry_csv", default="results/run_registry.csv")
    ap.add_argument("--masks_root", default="masks")
    ap.add_argument(
        "--sparse_modes",
        nargs="+",
        default=["sparse_to_dense", "sparse_to_sparse"],
        choices=["sparse_to_dense", "sparse_to_sparse"],
        help="Masked finetuning modes to run for each configuration.",
    )
    ap.add_argument("--notes", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--fail_fast", action="store_true", help="Stop at first failed configuration.")
    args = ap.parse_args()

    tasks = args.tasks or load_tasks_from_contract(args.contract)

    commands = []
    for model in args.models:
        for task in tasks:
            for sparsity in args.sparsities:
                for sparse_mode in args.sparse_modes:
                    if has_completed_run(args.registry_csv, model, task, sparsity, sparse_mode):
                        print(
                            "[SKIP] completed run already exists for "
                            f"model={model} task={task} mask_sparsity={sparsity} "
                            f"sparse_mode={sparse_mode}"
                        )
                        continue
                    cmd = [
                        sys.executable,
                        "scripts/gpu/run_week4_dense.py",
                        "--model",
                        model,
                        "--task",
                        task,
                        "--contract",
                        args.contract,
                        "--deepspeed",
                        args.deepspeed,
                        "--runs_root",
                        args.runs_root,
                        "--registry_csv",
                        args.registry_csv,
                        "--masks_root",
                        args.masks_root,
                        "--mask_sparsity",
                        str(sparsity),
                        "--sparse_mode",
                        sparse_mode,
                        "--notes",
                        args.notes,
                    ]
                    commands.append(cmd)

    failures = []
    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx}/{len(commands)}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append({"index": idx, "returncode": result.returncode, "cmd": cmd})
            print(f"[FAIL] idx={idx} returncode={result.returncode}")
            if args.fail_fast:
                break

    if failures:
        print(f"[SUMMARY] {len(failures)}/{len(commands)} runs failed.")
        for f in failures:
            print(f"[FAILED CMD {f['index']}] {' '.join(f['cmd'])}")
        if args.fail_fast:
            raise subprocess.CalledProcessError(
                failures[0]["returncode"], failures[0]["cmd"]
            )


if __name__ == "__main__":
    main()
