#!/usr/bin/env python3
"""
Parse Cerebras dense FT run logs and write a single CSV summary:

  ~/project3-sparse-ft/cerebras_dense_results.csv

For each run directory under results/runs matching *_YYYYMMDD_HHMMSS, this script:
- infers (base_model, task) from the folder name
- reads train.log (if present)
- extracts:
    train_batch_size            (Effective batch size is N.)
    train_global_rate           (GlobalRate from the last Train line)
    eval_loss                   (Avg Eval Loss)
    eval_perplexity             (eval/lm_perplexity)
    eval_acc                    (eval/accuracy)
    eval_global_rate            (GlobalRate from the last Eval line)

Usage:
  python collect_dense_results.py
  python collect_dense_results.py --runs-root results/runs --out ~/project3-sparse-ft/cerebras_dense_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List


TASKS_DEFAULT = {"boolq", "gsm8k", "hellaswag"}


RUN_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")

# Training line example:
# | Train Device=CSX, Step=200, Loss=..., Rate=... samples/sec, GlobalRate=55.24 samples/sec, ...
TRAIN_LINE_RE = re.compile(r"\|\s*Train\b.*?\bGlobalRate=([0-9]*\.?[0-9]+)\s+samples/sec")

# Eval line example:
# | Eval Device=CSX, GlobalStep=200, Batch=3, Loss=0.24839, Rate=..., GlobalRate=183.54 samples/sec, ...
EVAL_LINE_RE = re.compile(r"\|\s*Eval\b.*?\bGlobalRate=([0-9]*\.?[0-9]+)\s+samples/sec")

# Effective batch size:
EBS_RE = re.compile(r"Effective batch size is\s+(\d+)\.")

# Avg eval loss:
AVG_EVAL_LOSS_RE = re.compile(r"Avg Eval Loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")

# Metrics:
PPL_RE = re.compile(r"eval/lm_perplexity\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
ACC_RE = re.compile(r"eval/accuracy\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


@dataclass
class Row:
    base_model: str
    task: str
    train_batch_size: Optional[int]
    train_global_rate: Optional[float]
    eval_loss: Optional[float]
    eval_perplexity: Optional[float]
    eval_acc: Optional[float]
    eval_global_rate: Optional[float]


def expanduser(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def parse_run_name(run_dir_name: str, tasks: set[str]) -> Optional[Tuple[str, str, datetime]]:
    """
    Extract (base_model, task, timestamp) from a run dir name like:
      cs3_mistral_boolq_lora_f2ae45c_20260203_175242
      CS-3_llama3_boolq_20260201_200939
      cs3_llama3_8b_boolq_dense_lora_20260201_172710

    We split on "_" and "-" and interpret:
      [prefix] [model tokens...] [task] [rest...] _YYYYMMDD_HHMMSS
    """
    m = RUN_TS_RE.search(run_dir_name)
    if not m:
        return None

    date_s, time_s = m.group(1), m.group(2)
    try:
        ts = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    base = run_dir_name[: m.start()]
    parts = [p for p in re.split(r"[_-]+", base) if p]
    if len(parts) < 3:
        return None

    task_idx = next((i for i, p in enumerate(parts) if p.lower() in tasks), None)
    if task_idx is None or task_idx < 2:
        return None

    model = "_".join(parts[1:task_idx])
    task = parts[task_idx].lower()
    if not model:
        return None
    return model, task, ts


def find_runs(runs_root: Path, tasks: set[str]) -> List[Tuple[str, str, datetime, Path]]:
    runs: List[Tuple[str, str, datetime, Path]] = []
    if not runs_root.exists():
        return runs

    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        parsed = parse_run_name(child.name, tasks)
        if not parsed:
            continue
        model, task, ts = parsed
        runs.append((model, task, ts, child))
    return runs


def parse_train_log(log_path: Path) -> Dict[str, Optional[float]]:
    """
    Parse the train.log to extract:
      train_batch_size (int)
      train_global_rate (float)  from last Train line
      eval_global_rate (float)   from last Eval line
      eval_loss (float)          from Avg Eval Loss
      eval_perplexity (float)    from eval/lm_perplexity
      eval_acc (float)           from eval/accuracy
    """
    train_batch_size: Optional[int] = None
    train_global_rate: Optional[float] = None
    eval_global_rate: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_ppl: Optional[float] = None
    eval_acc: Optional[float] = None

    # Read line by line to keep memory low on huge logs.
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Effective batch size (keep the last one if repeated)
            m = EBS_RE.search(line)
            if m:
                try:
                    train_batch_size = int(m.group(1))
                except ValueError:
                    pass

            # Last train GlobalRate
            m = TRAIN_LINE_RE.search(line)
            if m:
                try:
                    train_global_rate = float(m.group(1))
                except ValueError:
                    pass

            # Last eval GlobalRate
            m = EVAL_LINE_RE.search(line)
            if m:
                try:
                    eval_global_rate = float(m.group(1))
                except ValueError:
                    pass

            # Avg eval loss
            m = AVG_EVAL_LOSS_RE.search(line)
            if m:
                try:
                    eval_loss = float(m.group(1))
                except ValueError:
                    pass

            # Perplexity
            m = PPL_RE.search(line)
            if m:
                try:
                    eval_ppl = float(m.group(1))
                except ValueError:
                    pass

            # Accuracy
            m = ACC_RE.search(line)
            if m:
                try:
                    eval_acc = float(m.group(1))
                except ValueError:
                    pass

    return {
        "train_batch_size": train_batch_size,
        "train_global_rate": train_global_rate,
        "eval_loss": eval_loss,
        "eval_perplexity": eval_ppl,
        "eval_acc": eval_acc,
        "eval_global_rate": eval_global_rate,
    }


def write_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model",
        "task",
        "train_batch_size",
        "train_global_rate",
        "eval_loss",
        "eval_perplexity",
        "eval_acc",
        "eval_global rate",  # keep exactly as you asked (space in header)
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["eval_global rate"] = d.pop("eval_global_rate")
            w.writerow(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="results/runs", help="Root folder containing run directories")
    ap.add_argument(
        "--out",
        default="~/project3-sparse-ft/cerebras_dense_results.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--tasks",
        default=",".join(sorted(TASKS_DEFAULT)),
        help="Comma-separated task tokens to recognize in folder names (e.g. boolq,gsm8k,hellaswag)",
    )
    ap.add_argument(
        "--skip-missing-log",
        action="store_true",
        help="If set, silently skip runs without train.log (default: include row with empty metrics).",
    )
    args = ap.parse_args()

    runs_root = expanduser(args.runs_root)
    out_path = expanduser(args.out)
    tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}

    runs = find_runs(runs_root, tasks)

    # Keep only latest per (model, task) if multiple exist (defensive in case cleanup missed some)
    latest: Dict[Tuple[str, str], Tuple[datetime, Path, str, str]] = {}
    for model, task, ts, path in runs:
        key = (model.lower(), task.lower())
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, path, model, task)

    rows: List[Row] = []
    for (_m, _t), (ts, run_path, model, task) in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        log_path = run_path / "train.log"
        if not log_path.exists():
            if args.skip_missing_log:
                continue
            rows.append(
                Row(
                    base_model=model,
                    task=task,
                    train_batch_size=None,
                    train_global_rate=None,
                    eval_loss=None,
                    eval_perplexity=None,
                    eval_acc=None,
                    eval_global_rate=None,
                )
            )
            continue

        metrics = parse_train_log(log_path)
        rows.append(
            Row(
                base_model=model,
                task=task,
                train_batch_size=metrics["train_batch_size"],
                train_global_rate=metrics["train_global_rate"],
                eval_loss=metrics["eval_loss"],
                eval_perplexity=metrics["eval_perplexity"],
                eval_acc=metrics["eval_acc"],
                eval_global_rate=metrics["eval_global_rate"],
            )
        )

    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} row(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
