#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "tables"

TASKS = ("boolq", "gsm8k", "hellaswag")
TARGET_BATCH_SIZES = (32, 64, 128, 256)


DENSE_RE = re.compile(
    r"^gpu-dense-(?P<model>.+)-(?P<task>boolq|gsm8k|hellaswag)-(?P<timestamp>\d{8}-\d{6}-\d+)$"
)
SPARSE_RE = re.compile(
    r"^gpu-week4_(?P<mode>sparse_to_dense|sparse_to_sparse)-(?P<model>.+)-(?P<task>boolq|gsm8k|hellaswag)-s(?P<sparsity>\d+)-(?P<timestamp>\d{8}-\d{6}-\d+)$"
)


ACC_COLUMNS = [
    "Model Name",
    "Task Name",
    "Sparsity Percentage",
    "Training Mode",
    "Run Timestamp",
    "Run Status",
    "Evaluation Accuracy",
    "Evaluation Language Model Perplexity",
    "Observed Batch Size",
    "Observed Training Throughput at Observed Batch Size (Samples Per Second)",
    "Run Directory",
]

TP_COLUMNS = [
    "Model Name",
    "Task Name",
    "Sparsity Percentage",
    "Training Mode",
    "Run Timestamp",
    "Run Status",
    "Evaluation Accuracy",
    "Evaluation Language Model Perplexity",
    "Observed Batch Size",
    "Estimated Training Throughput at Batch Size 32 (Samples Per Second, Linear Estimate)",
    "Estimated Training Throughput at Batch Size 64 (Samples Per Second, Linear Estimate)",
    "Estimated Training Throughput at Batch Size 128 (Samples Per Second, Linear Estimate)",
    "Estimated Training Throughput at Batch Size 256 (Samples Per Second, Linear Estimate)",
    "Run Directory",
]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def clean_number(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def parse_run_dir(run_dir: Path):
    name = run_dir.name
    dense_match = DENSE_RE.match(name)
    if dense_match:
        meta = dense_match.groupdict()
        return {
            "kind": "dense",
            "model": meta["model"],
            "task": meta["task"],
            "sparsity": 0,
            "training_mode": "dense",
            "timestamp": meta["timestamp"],
            "run_dir": run_dir,
        }

    sparse_match = SPARSE_RE.match(name)
    if sparse_match:
        meta = sparse_match.groupdict()
        return {
            "kind": "sparse",
            "model": meta["model"],
            "task": meta["task"],
            "sparsity": int(meta["sparsity"]),
            "training_mode": meta["mode"].replace("_", " "),
            "timestamp": meta["timestamp"],
            "run_dir": run_dir,
        }

    return None


def extract_result_blob(run_dir: Path):
    for candidate in ("run_result.json", "eval_results.json", "lm_eval.json"):
        data = load_json(run_dir / candidate)
        if not data:
            continue
        if "lm_eval" in data and isinstance(data["lm_eval"], dict):
            nested = data["lm_eval"]
            if "results" in nested:
                return nested
        if "results" in data:
            return data
    return None


def extract_eval_metrics(run_dir: Path, task: str):
    blob = extract_result_blob(run_dir)
    if not blob:
        return None, None, False

    results = blob.get("results") or {}
    task_metrics = results.get(task)
    if not isinstance(task_metrics, dict):
        return None, None, False

    accuracy = None
    for key in (
        "acc,none",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "acc_norm,none",
    ):
        value = clean_number(task_metrics.get(key))
        if value is not None:
            accuracy = value
            break

    perplexity = None
    for key in ("word_perplexity,none", "perplexity,none"):
        value = clean_number(task_metrics.get(key))
        if value is not None:
            perplexity = value
            break

    return accuracy, perplexity, True


def extract_status(run_dir: Path, has_eval_blob: bool):
    if (run_dir / "run_error.json").exists():
        return "failed"
    if (run_dir / "deepspeed_disabled.json").exists():
        return "blocked"
    if has_eval_blob:
        return "success eval"
    if any(
        (run_dir / candidate).exists()
        for candidate in ("train_summary.json", "train_progress.json", "throughput_scaling.json")
    ):
        return "missing eval"
    return "unknown"


def extract_throughput_summary(run_dir: Path):
    tp = load_json(run_dir / "throughput_scaling.json")
    points = (tp or {}).get("points") or []
    normalized = []

    for point in points:
        seq_len = clean_number(point.get("seq_len"))
        world_size = clean_number(point.get("world_size"))
        micro_batch = clean_number(point.get("micro_batch"))
        tokens_per_s = clean_number(point.get("tokens_per_s"))
        if not seq_len or not world_size or not micro_batch or tokens_per_s is None:
            continue
        batch_size = int(world_size * micro_batch)
        samples_per_s = tokens_per_s / seq_len
        normalized.append((batch_size, samples_per_s))

    if not normalized:
        summary = load_json(run_dir / "train_summary.json")
        if summary:
            seq_len = clean_number(summary.get("seq_len"))
            world_size = clean_number(summary.get("world_size"))
            micro_batch = clean_number(summary.get("micro_batch"))
            train_tokens_per_s = clean_number(summary.get("train_tokens_per_s"))
            if seq_len and world_size and micro_batch and train_tokens_per_s is not None:
                batch_size = int(world_size * micro_batch)
                samples_per_s = train_tokens_per_s / seq_len
                normalized.append((batch_size, samples_per_s))

    if not normalized:
        return None

    observed_batch_size, observed_samples_per_s = max(normalized, key=lambda item: item[0])
    estimates = {
        batch_size: round(observed_samples_per_s * batch_size / observed_batch_size, 2)
        for batch_size in TARGET_BATCH_SIZES
    }
    return {
        "observed_batch_size": observed_batch_size,
        "observed_samples_per_s": round(observed_samples_per_s, 2),
        "estimates": estimates,
    }


def build_rows(kind: str):
    acc_rows = []
    tp_rows = []

    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        meta = parse_run_dir(run_dir)
        if not meta or meta["kind"] != kind:
            continue

        accuracy, perplexity, has_eval_blob = extract_eval_metrics(run_dir, meta["task"])
        status = extract_status(run_dir, has_eval_blob)
        throughput = extract_throughput_summary(run_dir)

        base = {
            "Model Name": meta["model"],
            "Task Name": meta["task"],
            "Sparsity Percentage": meta["sparsity"],
            "Training Mode": meta["training_mode"],
            "Run Timestamp": meta["timestamp"],
            "Run Status": status,
            "Evaluation Accuracy": accuracy,
            "Evaluation Language Model Perplexity": perplexity,
            "Observed Batch Size": throughput["observed_batch_size"] if throughput else None,
            "Run Directory": str(run_dir.relative_to(ROOT)),
        }

        acc_rows.append(
            {
                **base,
                "Observed Training Throughput at Observed Batch Size (Samples Per Second)": (
                    throughput["observed_samples_per_s"] if throughput else None
                ),
            }
        )

        tp_rows.append(
            {
                **base,
                "Estimated Training Throughput at Batch Size 32 (Samples Per Second, Linear Estimate)": (
                    throughput["estimates"][32] if throughput else None
                ),
                "Estimated Training Throughput at Batch Size 64 (Samples Per Second, Linear Estimate)": (
                    throughput["estimates"][64] if throughput else None
                ),
                "Estimated Training Throughput at Batch Size 128 (Samples Per Second, Linear Estimate)": (
                    throughput["estimates"][128] if throughput else None
                ),
                "Estimated Training Throughput at Batch Size 256 (Samples Per Second, Linear Estimate)": (
                    throughput["estimates"][256] if throughput else None
                ),
            }
        )

    sort_key = lambda row: (
        row["Model Name"],
        row["Task Name"],
        row["Sparsity Percentage"],
        row["Training Mode"],
        row["Run Timestamp"],
        row["Run Directory"],
    )
    acc_rows.sort(key=sort_key)
    tp_rows.sort(key=sort_key)
    return acc_rows, tp_rows


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_number(row.get(key)) for key in fieldnames})


def main():
    dense_acc_rows, dense_tp_rows = build_rows("dense")
    sparse_acc_rows, sparse_tp_rows = build_rows("sparse")

    write_csv(TABLES_DIR / "run_gpu_dense_acc.csv", ACC_COLUMNS, dense_acc_rows)
    write_csv(TABLES_DIR / "run_gpu_dense_tp_scale.csv", TP_COLUMNS, dense_tp_rows)
    write_csv(TABLES_DIR / "run_gpu_sparse_acc.csv", ACC_COLUMNS, sparse_acc_rows)
    write_csv(TABLES_DIR / "run_gpu_sparse_tp_scale.csv", TP_COLUMNS, sparse_tp_rows)


if __name__ == "__main__":
    main()
