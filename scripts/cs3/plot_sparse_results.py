#!/usr/bin/env python3
"""
Generate Phoenix-comparable plots and tables from sparse LoRA results.

Reads training logs and benchmark CSVs produced by the sweep and outputs:
  1. Sparsity vs validation loss (per model × task, both modes)
  2. Throughput vs batch size (per sparsity level)
  3. Summary comparison table (CSV)  — formatted like Phoenix Table 3
  4. Accuracy-recovery heatmap

Usage:
    python plot_sparse_results.py \\
        --results_dir results/masked_runs_cs \\
        --output_dir  results/masked_runs_cs/analysis
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ======================================================================
# Log parsers
# ======================================================================

_LOSS_RE = re.compile(r"(?:Train|train).*?[Ll]oss\s*[=:]\s*([0-9]+\.?[0-9]*)")
_EVAL_LOSS_RE = re.compile(r"(?:Eval|Val).*?[Ll]oss\s*[=:]\s*([0-9]+\.?[0-9]*)")
_THROUGHPUT_RE = re.compile(
    r"GlobalRate\s*=\s*([0-9]+\.?[0-9]*)\s*samples/sec", re.I
)


def parse_train_log(log_path: Path) -> dict:
    metrics: dict = {}
    if not log_path.exists():
        return metrics
    text = log_path.read_text(errors="ignore")

    losses = _LOSS_RE.findall(text)
    if losses:
        metrics["final_train_loss"] = float(losses[-1])

    eval_losses = _EVAL_LOSS_RE.findall(text)
    if eval_losses:
        metrics["final_eval_loss"] = float(eval_losses[-1])

    throughputs = _THROUGHPUT_RE.findall(text)
    if throughputs:
        metrics["throughput_samples_s"] = float(throughputs[-1])

    return metrics


# ======================================================================
# Collect results
# ======================================================================

def collect_run_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    success_states = {"success_eval", "success_train_only", "checkpoint_only"}

    def classify_run_state(run_dir: Path) -> str:
        log_path = run_dir / "train.log"
        ckpt_200 = run_dir / "model_dir" / "checkpoint_200.mdl"
        if not log_path.exists():
            return "failed"
        text = log_path.read_text(errors="ignore")
        if "Evaluation completed successfully!" in text:
            return "success_eval"
        if "Training completed successfully!" in text and ckpt_200.exists():
            return "success_train_only"
        if ckpt_200.exists():
            return "checkpoint_only"
        return "failed"

    for rd in sorted(results_dir.iterdir()):
        if not rd.is_dir() or not rd.name.startswith("cs3_"):
            continue

        meta_path = rd / "sparse_config.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())

        run_state = classify_run_state(rd)
        if run_state not in success_states:
            continue

        log_metrics = parse_train_log(rd / "train.log")
        meta.update(log_metrics)
        meta["run_state"] = run_state
        meta["run_dir"] = str(rd)
        rows.append(meta)

    return pd.DataFrame(rows)


def collect_bench_results(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(glob.glob(str(results_dir / "tables" / "cs3_throughput_*.csv")))
    if not csvs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csvs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ======================================================================
# Tables
# ======================================================================

def make_phoenix_comparison_table(df: pd.DataFrame, out: Path):
    """
    Create a CSV table analogous to Phoenix paper Table 3:
    rows = (model, task), columns = sparsity levels,
    cells = eval loss (or train loss as proxy).
    """
    if df.empty:
        return
    loss_col = "final_eval_loss" if "final_eval_loss" in df.columns else "final_train_loss"
    if loss_col not in df.columns:
        return

    pivot = df.pivot_table(
        values=loss_col,
        index=["model", "task", "mode"],
        columns="sparsity",
        aggfunc="mean",
    )
    pivot.to_csv(out)
    print(f"  Phoenix-style table: {out}")


# ======================================================================
# Plots
# ======================================================================

def plot_sparsity_vs_loss(df: pd.DataFrame, out_dir: Path):
    if not HAS_PLOT or df.empty:
        return
    loss_col = "final_eval_loss" if "final_eval_loss" in df.columns else "final_train_loss"
    if loss_col not in df.columns:
        return

    sns.set_style("whitegrid")

    for (model, task), grp in df.groupby(["model", "task"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, mg in grp.groupby("mode"):
            agg = mg.groupby("sparsity")[loss_col].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["sparsity"], agg["mean"], yerr=agg["std"],
                marker="o", capsize=4, label=mode,
            )
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model} – {task}: Sparsity vs Loss (CS-3)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fpath = out_dir / f"sparsity_vs_loss_{model}_{task}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fpath}")


def plot_throughput_vs_batch(bench_df: pd.DataFrame, out_dir: Path):
    if not HAS_PLOT or bench_df.empty:
        return
    for col in ["batch_size", "global_rate_samples_s"]:
        if col in bench_df.columns:
            bench_df[col] = pd.to_numeric(bench_df[col], errors="coerce")
    bench_df = bench_df.dropna(subset=["batch_size", "global_rate_samples_s"])
    if bench_df.empty:
        return

    sns.set_style("whitegrid")

    group_cols = [c for c in ["model", "task"] if c in bench_df.columns]
    if not group_cols:
        return

    for keys, grp in bench_df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        label = "_".join(str(k) for k in keys)
        fig, ax = plt.subplots(figsize=(8, 5))

        sp_col = "sparsity" if "sparsity" in grp.columns else None
        if sp_col:
            for sp, sg in grp.groupby(sp_col):
                ax.plot(sg["batch_size"], sg["global_rate_samples_s"],
                        marker="o", label=f"s{sp}")
        else:
            ax.plot(grp["batch_size"], grp["global_rate_samples_s"], marker="o")

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title(f"CS-3 Throughput: {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fpath = out_dir / f"throughput_vs_batch_{label}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fpath}")


def plot_heatmap(df: pd.DataFrame, out_dir: Path):
    if not HAS_PLOT or df.empty:
        return
    loss_col = "final_eval_loss" if "final_eval_loss" in df.columns else "final_train_loss"
    if loss_col not in df.columns:
        return

    sns.set_style("whitegrid")

    for mode, mg in df.groupby("mode"):
        pivot = mg.pivot_table(
            values=loss_col,
            index=["model", "task"],
            columns="sparsity",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"Loss Heatmap – {mode}")
        fpath = out_dir / f"heatmap_{mode}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fpath}")


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Plot sparse LoRA results")
    ap.add_argument("--results_dir",
                    default=str(PROJECT_ROOT / "results" / "masked_runs_cs"))
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting training results ...")
    df = collect_run_results(results_dir)
    print(f"  Found {len(df)} runs")

    if not df.empty:
        df.to_csv(out_dir / "all_runs.csv", index=False)
        make_phoenix_comparison_table(df, out_dir / "phoenix_comparison.csv")

    print("\nCollecting benchmark results ...")
    bench_df = collect_bench_results(results_dir)
    print(f"  Found {len(bench_df)} bench rows")

    if not bench_df.empty:
        bench_df.to_csv(out_dir / "all_bench.csv", index=False)

    if not args.no_plot:
        print("\nGenerating plots ...")
        plot_sparsity_vs_loss(df, out_dir)
        plot_throughput_vs_batch(bench_df, out_dir)
        plot_heatmap(df, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
