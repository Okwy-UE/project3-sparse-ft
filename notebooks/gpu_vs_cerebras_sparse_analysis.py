# %% [markdown]
# # GPU vs Cerebras Analysis Cells
#
# This file is organized with `# %%` cell markers so you can run it directly in an
# editor that supports notebook-style Python cells, or copy each section into a
# Jupyter notebook with minimal cleanup.

# %%
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter


# %%
# Global plotting style chosen to match the reference figures.
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "axes.facecolor": "#e6e6e6",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#9a9a9a",
        "grid.alpha": 0.65,
        "grid.linestyle": "-",
        "axes.edgecolor": "#444444",
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.framealpha": 0.9,
        "legend.facecolor": "#efefef",
        "legend.edgecolor": "#bbbbbb",
        "font.size": 11,
        "lines.markersize": 5,
    }
)

MODEL_COLORS = {
    "llama3": "blue",
    "mistral": "green",
    "mixtral": "red",
}

MODEL_MARKERS = {
    "llama3": "o",
    "mistral": "s",
    "mixtral": "D",
}

DEVICE_COLORS = {
    "GPU": "darkorange",
    "Cerebras": "green",
}

MODE_STYLES = {
    "dense": {"linestyle": "--", "linewidth": 1.6},
    "sparse to dense": {"linestyle": "-", "linewidth": 2.0},
    "sparse to sparse": {"linestyle": ":", "linewidth": 2.4},
}


# %%
# Paths. This script supports both direct execution and interactive cell execution.
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()

TABLES = ROOT / "tables"
RUNS = ROOT / "results" / "runs"

GPU_SPARSE_ACC = TABLES / "run_gpu_sparse_acc.csv"
GPU_DENSE_ACC = TABLES / "run_gpu_dense_acc.csv"
GPU_SPARSE_TP = TABLES / "run_gpu_sparse_tp_scale.csv"
GPU_DENSE_TP = TABLES / "run_gpu_dense_tp_scale.csv"
CS_SPARSE_ACC = TABLES / "run_cerebras_sparse_results_by_task.csv"
CS_SPARSE_TP = TABLES / "run_cerebras_sparse_throughput_estimates_by_batch_size.csv"
CS_DENSE = TABLES / "cerebras_dense_results.csv"


# %%
def require_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_model_name(value: str) -> str:
    mapping = {
        "llama3": "llama3",
        "llama3.1-8b": "llama3",
        "mistral": "mistral",
        "mistral-7b": "mistral",
        "mixtral": "mixtral",
        "mixtral-8x7b": "mixtral",
    }
    return mapping.get(str(value).strip(), str(value).strip())


def normalize_mode_name(value: str) -> str:
    mapping = {
        "dense": "dense",
        "sparse to dense": "sparse to dense",
        "sparse_to_dense": "sparse to dense",
        "sparse to sparse": "sparse to sparse",
        "sparse_to_sparse": "sparse to sparse",
    }
    return mapping.get(str(value).strip(), str(value).strip())


def parse_gpu_run_dir_name(run_name: str):
    tasks = ("boolq", "gsm8k", "hellaswag")

    if run_name.startswith("gpu-dense-"):
        for task in tasks:
            marker = f"-{task}-"
            if marker in run_name:
                prefix, suffix = run_name.split(marker, 1)
                model = prefix[len("gpu-dense-") :]
                return {
                    "model": normalize_model_name(model),
                    "task": task,
                    "mode": "dense",
                    "sparsity": 0,
                    "timestamp": suffix,
                }

    if run_name.startswith("gpu-week4_sparse_to_"):
        for task in tasks:
            marker = f"-{task}-s"
            if marker in run_name:
                left, right = run_name.split(marker, 1)
                prefix = "gpu-week4_"
                payload = left[len(prefix) :]
                for mode_prefix in ("sparse_to_dense-", "sparse_to_sparse-"):
                    if payload.startswith(mode_prefix):
                        mode = mode_prefix[:-1]
                        model = payload[len(mode_prefix) :]
                        break
                else:
                    return None
                sparsity_str, timestamp = right.split("-", 1)
                return {
                    "model": normalize_model_name(model),
                    "task": task,
                    "mode": normalize_mode_name(mode),
                    "sparsity": int(sparsity_str),
                    "timestamp": timestamp,
                }

    return None


def title_model_name(value: str) -> str:
    mapping = {
        "llama3": "Llama-3-8B",
        "mistral": "Mistral-7B-v3",
        "mixtral": "Mixtral-8x7B",
    }
    return mapping.get(value, value)


def clean_status(df: pd.DataFrame, status_col: str = "Run Status") -> pd.DataFrame:
    if status_col not in df.columns:
        return df.copy()
    out = df.copy()
    status = out[status_col].fillna("").astype(str).str.lower()
    return out[status != "blocked"].copy()


def format_percent(v: float) -> str:
    return f"{v:.2f}%"


def format_rate(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def build_log2_ticks(values) -> list[int]:
    values = sorted({int(v) for v in values if pd.notna(v) and v > 0})
    if not values:
        return []
    lo = int(np.floor(np.log2(min(values))))
    hi = int(np.ceil(np.log2(max(values))))
    ticks = {2**k for k in range(lo, hi + 1)}
    ticks.update(values)
    return sorted(ticks)


def add_in_axes_legend(ax, handles, loc="lower left"):
    if not handles:
        return
    ax.legend(handles=handles, loc=loc, fontsize=10)


def add_compact_figure_legend(fig, handles_by_label: dict[str, Line2D], max_cols: int = 3):
    if not handles_by_label:
        return

    handles = list(handles_by_label.values())
    ncols = min(max_cols, max(1, len(handles)))

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=ncols,
        fontsize=9,
        frameon=True,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.45,
    )


def annotation_offset(mode: str, dense_xy: tuple[int, int] = (0, 8)) -> tuple[int, int]:
    if mode == "sparse to sparse":
        return (8, 8)
    if mode == "sparse to dense":
        return (-8, -16)
    return dense_xy


# %%
# Load the current tables.
gpu_sparse_acc_raw = clean_status(require_table(GPU_SPARSE_ACC))
gpu_dense_acc_raw = clean_status(require_table(GPU_DENSE_ACC))
gpu_sparse_tp_raw = clean_status(require_table(GPU_SPARSE_TP))
gpu_dense_tp_raw = clean_status(require_table(GPU_DENSE_TP))
cs_sparse_acc_raw = clean_status(require_table(CS_SPARSE_ACC))
cs_sparse_tp_raw = clean_status(require_table(CS_SPARSE_TP))
cs_dense_raw = require_table(CS_DENSE)

print("Loaded tables:")
for name, df in [
    ("gpu_sparse_acc", gpu_sparse_acc_raw),
    ("gpu_dense_acc", gpu_dense_acc_raw),
    ("gpu_sparse_tp", gpu_sparse_tp_raw),
    ("gpu_dense_tp", gpu_dense_tp_raw),
    ("cs_sparse_acc", cs_sparse_acc_raw),
    ("cs_sparse_tp", cs_sparse_tp_raw),
    ("cs_dense", cs_dense_raw),
]:
    print(f"  {name}: {len(df)} rows")


# %% [markdown]
# ## 0) Hardware Reference Table

# %%
hardware_specs = pd.DataFrame(
    [
        {
            "Device": "Cerebras CS-3",
            "Processor": "Wafer-Scale Engine 3 (WSE-3)",
            "On-device memory": "44 GB SRAM",
            "Compute summary": "~125 PFLOPS AI compute (sparse, vendor-cited)",
            "Core / compute units": "~900,000 AI-optimized cores",
            "Memory bandwidth": "~21 PB/s on-wafer bandwidth",
        },
        {
            "Device": "NVIDIA A100 40 GB",
            "Processor": "Ampere GA100",
            "On-device memory": "40 GB HBM2",
            "Compute summary": "Up to ~312 TFLOPS Tensor Core (FP16/BF16 with sparsity)",
            "Core / compute units": "6912 CUDA cores, 432 Tensor Cores",
            "Memory bandwidth": "Up to ~1.6 TB/s",
        },
    ]
)

hardware_specs


# %% [markdown]
# ## 1) Build Unified Accuracy Frame
#
# Dense runs are treated as 0% sparsity.
# Mixtral is kept on the Cerebras side even if GPU sparse runs are missing.

# %%
def build_accuracy_frame() -> pd.DataFrame:
    gpu_sparse = gpu_sparse_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Evaluation Accuracy": "accuracy",
        }
    ).copy()
    gpu_sparse["device"] = "GPU"

    gpu_dense = gpu_dense_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Evaluation Accuracy": "accuracy",
        }
    ).copy()
    gpu_dense["device"] = "GPU"

    cs_sparse = cs_sparse_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Evaluation Accuracy": "accuracy",
        }
    ).copy()
    cs_sparse["device"] = "Cerebras"

    cs_dense = cs_dense_raw.rename(
        columns={
            "base_model": "model",
            "task": "task",
            "eval_acc": "accuracy",
        }
    ).copy()
    cs_dense["device"] = "Cerebras"
    cs_dense["sparsity"] = 0
    cs_dense["mode"] = "dense"
    cs_dense["Run Status"] = "success eval"

    cols = ["device", "model", "task", "sparsity", "mode", "accuracy", "Run Status"]
    combined = pd.concat(
        [gpu_sparse[cols], gpu_dense[cols], cs_sparse[cols], cs_dense[cols]],
        ignore_index=True,
    )
    combined["model"] = combined["model"].map(normalize_model_name)
    combined["mode"] = combined["mode"].map(normalize_mode_name)
    combined["task"] = combined["task"].astype(str).str.lower()
    combined["sparsity"] = pd.to_numeric(combined["sparsity"], errors="coerce")
    combined["accuracy_pct"] = pd.to_numeric(combined["accuracy"], errors="coerce") * 100.0
    combined = combined.dropna(subset=["sparsity", "accuracy_pct"]).copy()
    return combined.sort_values(["task", "device", "model", "mode", "sparsity"])


accuracy_df = build_accuracy_frame()
accuracy_df.head()


# %% [markdown]
# ## 2) Accuracy Plots
#
# One figure per task. Two columns per device (`GPU`, `Cerebras`).
# Colors encode model. Line style encodes training mode.
# Dense baselines are rendered as horizontal lines for the 0% sparsity reference.

# %%
def plot_accuracy_panels(acc_df: pd.DataFrame):
    tasks = sorted(acc_df["task"].unique())
    devices = ["GPU", "Cerebras"]

    for task in tasks:
        task_df = acc_df[acc_df["task"] == task].copy()
        if task_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
        figure_legend_handles = {}
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        y_min = task_df["accuracy_pct"].min()
        y_max = task_df["accuracy_pct"].max()
        pad = max(1.0, 0.08 * (y_max - y_min if y_max > y_min else 5.0))

        for ax, device in zip(axes, devices):
            subset = task_df[task_df["device"] == device].copy()
            ax.set_facecolor("#e6e6e6")
            ax.set_xlabel("Sparsity (%)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(device)
            ax.set_xlim(-2, max(80, subset["sparsity"].max() + 5 if not subset.empty else 80))
            ax.set_ylim(y_min - pad, y_max + pad)

            legend_handles = {}

            for model in sorted(subset["model"].unique(), key=lambda x: ["llama3", "mistral", "mixtral"].index(x) if x in ["llama3", "mistral", "mixtral"] else 99):
                model_subset = subset[subset["model"] == model].copy()
                color = MODEL_COLORS.get(model, "black")
                marker = MODEL_MARKERS.get(model, "o")

                dense_rows = model_subset[model_subset["mode"] == "dense"].sort_values("sparsity")
                if not dense_rows.empty:
                    baseline_y = dense_rows["accuracy_pct"].iloc[-1]
                    ax.axhline(
                        baseline_y,
                        color=color,
                        linestyle=MODE_STYLES["dense"]["linestyle"],
                        linewidth=MODE_STYLES["dense"]["linewidth"],
                        alpha=0.9,
                    )
                    ax.annotate(
                        format_percent(baseline_y),
                        xy=(2, baseline_y),
                        xytext=annotation_offset("dense", dense_xy=(0, 6)),
                        textcoords="offset points",
                        color=color,
                        fontsize=11,
                    )
                    label = f"{title_model_name(model)} Baseline (0% sparsity)"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.6,
                        label=label,
                    )

                for mode in ["sparse to dense", "sparse to sparse"]:
                    mode_subset = model_subset[model_subset["mode"] == mode].sort_values("sparsity")
                    if mode_subset.empty:
                        continue
                    style = MODE_STYLES[mode]
                    ax.plot(
                        mode_subset["sparsity"],
                        mode_subset["accuracy_pct"],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker=marker,
                    )
                    for _, row in mode_subset.iterrows():
                        dx, dy = annotation_offset(mode)
                        ax.annotate(
                            format_percent(row["accuracy_pct"]),
                            (row["sparsity"], row["accuracy_pct"]),
                            textcoords="offset points",
                            xytext=(dx, dy),
                            ha="center",
                            color=color,
                            fontsize=11,
                        )
                    label = f"{title_model_name(model)} ({mode})"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker=marker,
                        label=label,
                    )

            figure_legend_handles.update(legend_handles)

        add_compact_figure_legend(fig, figure_legend_handles, max_cols=3)
        fig.suptitle(f"Model Accuracy on {task.upper()} vs. Sparsity Level", y=1.05, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


plot_accuracy_panels(accuracy_df)


# %% [markdown]
# ## 3) Build Unified Final Throughput Frame
#
# This is throughput at the observed batch size for each run, with dense treated as 0% sparsity.

# %%
def build_final_throughput_frame() -> pd.DataFrame:
    gpu_sparse = gpu_sparse_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Observed Batch Size": "batch_size",
            "Observed Training Throughput at Observed Batch Size (Samples Per Second)": "throughput",
        }
    ).copy()
    gpu_sparse["device"] = "GPU"

    gpu_dense = gpu_dense_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Observed Batch Size": "batch_size",
            "Observed Training Throughput at Observed Batch Size (Samples Per Second)": "throughput",
        }
    ).copy()
    gpu_dense["device"] = "GPU"

    cs_sparse = cs_sparse_acc_raw.rename(
        columns={
            "Model Name": "model",
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "mode",
            "Observed Batch Size": "batch_size",
            "Observed Training Throughput at Batch Size 256 (Samples Per Second)": "throughput",
        }
    ).copy()
    cs_sparse["device"] = "Cerebras"

    cs_dense = cs_dense_raw.rename(
        columns={
            "base_model": "model",
            "task": "task",
            "train_batch_size": "batch_size",
            "train_global_rate": "throughput",
        }
    ).copy()
    cs_dense["device"] = "Cerebras"
    cs_dense["sparsity"] = 0
    cs_dense["mode"] = "dense"

    cols = ["device", "model", "task", "sparsity", "mode", "batch_size", "throughput"]
    combined = pd.concat(
        [gpu_sparse[cols], gpu_dense[cols], cs_sparse[cols], cs_dense[cols]],
        ignore_index=True,
    )
    combined["model"] = combined["model"].map(normalize_model_name)
    combined["mode"] = combined["mode"].map(normalize_mode_name)
    combined["task"] = combined["task"].astype(str).str.lower()
    combined["sparsity"] = pd.to_numeric(combined["sparsity"], errors="coerce")
    combined["batch_size"] = pd.to_numeric(combined["batch_size"], errors="coerce")
    combined["throughput"] = pd.to_numeric(combined["throughput"], errors="coerce")
    combined = combined.dropna(subset=["sparsity", "throughput"]).copy()
    combined["batch_label"] = combined["batch_size"].apply(
        lambda v: f"BS {int(v)}" if pd.notna(v) else "BS ?"
    )
    return combined.sort_values(["task", "model", "device", "mode", "sparsity"])


final_tp_df = build_final_throughput_frame()
final_tp_df.head()


# %% [markdown]
# ## 4) Final Throughput vs. Sparsity
#
# One figure per task. Each column is a model. Colors encode device, and line style
# encodes training mode. Batch-size labels are drawn at each point. The legend stays
# inside each subplot.

# %%
def plot_final_throughput_panels(tp_df: pd.DataFrame):
    tasks = sorted(tp_df["task"].unique())

    for task in tasks:
        task_df = tp_df[tp_df["task"] == task].copy()
        models = sorted(
            task_df["model"].unique(),
            key=lambda x: ["llama3", "mistral", "mixtral"].index(x) if x in ["llama3", "mistral", "mixtral"] else 99,
        )
        if not models:
            continue

        fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.9), sharey=True)
        figure_legend_handles = {}
        if len(models) == 1:
            axes = np.array([axes])

        for ax, model in zip(axes, models):
            subset = task_df[task_df["model"] == model].copy()
            ax.set_facecolor("#e6e6e6")
            ax.set_title(title_model_name(model))
            ax.set_xlabel("Sparsity (%)")
            ax.set_ylabel("Throughput (samples/s)")

            legend_handles = {}
            for device in ["Cerebras", "GPU"]:
                device_subset = subset[subset["device"] == device].copy()
                color = DEVICE_COLORS.get(device, "black")

                dense_rows = device_subset[device_subset["mode"] == "dense"].sort_values("sparsity")
                if not dense_rows.empty:
                    baseline = dense_rows.iloc[-1]
                    ax.axhline(
                        baseline["throughput"],
                        color=color,
                        linestyle=MODE_STYLES["dense"]["linestyle"],
                        linewidth=MODE_STYLES["dense"]["linewidth"],
                    )
                    ax.annotate(
                        f"{format_rate(baseline['throughput'])}\n{baseline['batch_label']}",
                        (2, baseline["throughput"]),
                        textcoords="offset points",
                        xytext=annotation_offset("dense", dense_xy=(0, 6)),
                        color=color,
                        fontsize=10,
                    )
                    label = f"{device} baseline (0%)"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.6,
                        label=label,
                    )

                for mode in ["sparse to dense", "sparse to sparse"]:
                    mode_subset = device_subset[device_subset["mode"] == mode].sort_values("sparsity")
                    if mode_subset.empty:
                        continue
                    style = MODE_STYLES[mode]
                    ax.plot(
                        mode_subset["sparsity"],
                        mode_subset["throughput"],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker="o",
                    )
                    for _, row in mode_subset.iterrows():
                        dx, dy = annotation_offset(mode)
                        ax.annotate(
                            f"{format_rate(row['throughput'])}\n{row['batch_label']}",
                            (row["sparsity"], row["throughput"]),
                            textcoords="offset points",
                            xytext=(dx, dy),
                            ha="center",
                            color=color,
                            fontsize=10,
                        )
                    label = f"{device} ({mode})"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker="o",
                        label=label,
                    )

            figure_legend_handles.update(legend_handles)

        add_compact_figure_legend(fig, figure_legend_handles, max_cols=3)
        fig.suptitle(f"Training Throughput on {task.upper()} vs. Sparsity Level", y=1.05, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        plt.show()


plot_final_throughput_panels(final_tp_df)


# %% [markdown]
# ## 5) Build Throughput Scaling Frame
#
# This expands the estimated-throughput columns into long form. Dense is treated as
# 0% sparsity. Cerebras dense contributes one observed point at its training batch.

# %%
TP_BATCH_COLUMNS = {
    32: "Estimated Training Throughput at Batch Size 32 (Samples Per Second, Linear Estimate)",
    64: "Estimated Training Throughput at Batch Size 64 (Samples Per Second, Linear Estimate)",
    128: "Estimated Training Throughput at Batch Size 128 (Samples Per Second, Linear Estimate)",
    256: "Estimated Training Throughput at Batch Size 256 (Samples Per Second, Linear Estimate)",
}


def melt_estimated_tp(df: pd.DataFrame, device: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for batch_size, col in TP_BATCH_COLUMNS.items():
            if col not in row.index:
                continue
            rows.append(
                {
                    "device": device,
                    "model": normalize_model_name(row["Model Name"]),
                    "task": str(row["Task Name"]).lower(),
                    "sparsity": pd.to_numeric(row["Sparsity Percentage"], errors="coerce"),
                    "mode": normalize_mode_name(row["Training Mode"]),
                    "batch_size": batch_size,
                    "throughput": pd.to_numeric(row[col], errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def build_gpu_actual_scaling_frame() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_gpu_run_dir_name(run_dir.name)
        if meta is None:
            continue
        tp_path = run_dir / "throughput_scaling.json"
        if not tp_path.exists():
            continue

        payload = json.loads(tp_path.read_text())
        for point in payload.get("points", []):
            seq_len = pd.to_numeric(point.get("seq_len"), errors="coerce")
            tokens_per_s = pd.to_numeric(point.get("tokens_per_s"), errors="coerce")
            micro_batch = pd.to_numeric(point.get("micro_batch"), errors="coerce")
            world_size = pd.to_numeric(point.get("world_size"), errors="coerce")
            if pd.isna(seq_len) or pd.isna(tokens_per_s) or pd.isna(micro_batch) or pd.isna(world_size):
                continue
            rows.append(
                {
                    "device": "GPU",
                    "model": meta["model"],
                    "task": meta["task"],
                    "sparsity": meta["sparsity"],
                    "mode": meta["mode"],
                    "batch_size": int(micro_batch * world_size),
                    "throughput": float(tokens_per_s) / float(seq_len),
                }
            )

    return pd.DataFrame(rows)


def build_cerebras_downward_scaling_frame() -> pd.DataFrame:
    rows = []

    # Sparse: keep observed/estimated 32..256 points and extrapolate only downward to 1..16 from batch-32.
    for _, row in cs_sparse_tp_raw.iterrows():
        base = {
            "device": "Cerebras",
            "model": normalize_model_name(row["Model Name"]),
            "task": str(row["Task Name"]).lower(),
            "sparsity": pd.to_numeric(row["Sparsity Percentage"], errors="coerce"),
            "mode": normalize_mode_name(row["Training Mode"]),
        }
        if pd.isna(base["sparsity"]):
            continue

        batch32_val = pd.to_numeric(row[TP_BATCH_COLUMNS[32]], errors="coerce")
        if pd.notna(batch32_val):
            for batch_size in [1, 2, 4, 8, 16]:
                rows.append(
                    {
                        **base,
                        "batch_size": batch_size,
                        "throughput": float(batch32_val) * (batch_size / 32.0),
                    }
                )

        for batch_size, col in TP_BATCH_COLUMNS.items():
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(value):
                rows.append({**base, "batch_size": batch_size, "throughput": float(value)})

    # Dense: use observed point and extrapolate only downward to powers of two down to 1.
    for _, row in cs_dense_raw.iterrows():
        model = normalize_model_name(row["base_model"])
        task = str(row["task"]).lower()
        observed_batch = pd.to_numeric(row["train_batch_size"], errors="coerce")
        observed_tp = pd.to_numeric(row["train_global_rate"], errors="coerce")
        if pd.isna(observed_batch) or pd.isna(observed_tp) or observed_batch <= 0:
            continue

        powers = []
        k = 0
        while 2**k <= observed_batch:
            powers.append(2**k)
            k += 1

        for batch_size in powers:
            rows.append(
                {
                    "device": "Cerebras",
                    "model": model,
                    "task": task,
                    "sparsity": 0,
                    "mode": "dense",
                    "batch_size": batch_size,
                    "throughput": float(observed_tp) * (batch_size / float(observed_batch)),
                }
            )

    return pd.DataFrame(rows)


def build_scaling_frame() -> pd.DataFrame:
    gpu_actual = build_gpu_actual_scaling_frame()
    cs_scaling = build_cerebras_downward_scaling_frame()

    combined = pd.concat([gpu_actual, cs_scaling], ignore_index=True)
    combined["sparsity"] = pd.to_numeric(combined["sparsity"], errors="coerce")
    combined["batch_size"] = pd.to_numeric(combined["batch_size"], errors="coerce")
    combined["throughput"] = pd.to_numeric(combined["throughput"], errors="coerce")
    combined = combined.dropna(subset=["sparsity", "batch_size", "throughput"]).copy()
    return combined.sort_values(["task", "sparsity", "model", "device", "mode", "batch_size"])


scaling_df = build_scaling_frame()
scaling_df.head()


# %% [markdown]
# ## 6) Throughput Scaling Plots
#
# One figure per `(task, sparsity)` pair. Each column is a model. The x-axis is log2.
# Tick marks are explicitly fixed so batch size 16 is always shown when it is in range.

# %%
def plot_scaling_panels(scale_df: pd.DataFrame):
    task_sparsity_pairs = (
        scale_df[["task", "sparsity"]].drop_duplicates().sort_values(["task", "sparsity"]).itertuples(index=False)
    )

    for pair in task_sparsity_pairs:
        task = pair.task
        sparsity = pair.sparsity
        subset = scale_df[(scale_df["task"] == task) & (scale_df["sparsity"] == sparsity)].copy()
        models = sorted(
            subset["model"].unique(),
            key=lambda x: ["llama3", "mistral", "mixtral"].index(x) if x in ["llama3", "mistral", "mixtral"] else 99,
        )
        if not models:
            continue

        fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.9), sharey=True)
        figure_legend_handles = {}
        if len(models) == 1:
            axes = np.array([axes])

        for ax, model in zip(axes, models):
            model_subset = subset[subset["model"] == model].copy()
            gpu_subset_for_oom = model_subset[model_subset["device"] == "GPU"].copy()
            gpu_oom_xs = set()
            for mode in ["dense", "sparse to dense", "sparse to sparse"]:
                mode_subset = gpu_subset_for_oom[gpu_subset_for_oom["mode"] == mode]
                if mode_subset.empty:
                    continue
                max_batch = int(mode_subset["batch_size"].max())
                oom_x = max(1, 2 ** (int(np.ceil(np.log2(max_batch))) + (0 if max_batch & (max_batch - 1) else 1)))
                gpu_oom_xs.add(oom_x)

            all_ticks = build_log2_ticks([1] + model_subset["batch_size"].tolist() + list(gpu_oom_xs))

            ax.set_facecolor("#e6e6e6")
            ax.set_title(title_model_name(model))
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Throughput (samples/s)")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(all_ticks))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xticks(all_ticks)
            ax.tick_params(axis="x", labelsize=9)
            ax.set_xlim(left=1, right=max(all_ticks) * 1.08)

            legend_handles = {}
            for device in ["Cerebras", "GPU"]:
                device_subset = model_subset[model_subset["device"] == device].copy()
                color = DEVICE_COLORS.get(device, "black")

                dense_rows = device_subset[device_subset["mode"] == "dense"].sort_values("batch_size")
                if not dense_rows.empty:
                    ax.plot(
                        dense_rows["batch_size"],
                        dense_rows["throughput"],
                        color=color,
                        linestyle=MODE_STYLES["dense"]["linestyle"],
                        linewidth=MODE_STYLES["dense"]["linewidth"],
                        marker="o",
                    )
                    for _, row in dense_rows.iterrows():
                        ax.annotate(
                            format_rate(row["throughput"]),
                            (row["batch_size"], row["throughput"]),
                            textcoords="offset points",
                            xytext=annotation_offset("dense", dense_xy=(0, 6)),
                            ha="center",
                            color=color,
                            fontsize=11,
                        )
                    label = f"{device} (dense)"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.6,
                        marker="o",
                        label=label,
                    )

                for mode in ["sparse to dense", "sparse to sparse"]:
                    mode_subset = device_subset[device_subset["mode"] == mode].sort_values("batch_size")
                    if mode_subset.empty:
                        continue
                    style = MODE_STYLES[mode]
                    ax.plot(
                        mode_subset["batch_size"],
                        mode_subset["throughput"],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker="o",
                    )
                    for _, row in mode_subset.iterrows():
                        dx, dy = annotation_offset(mode)
                        ax.annotate(
                            format_rate(row["throughput"]),
                            (row["batch_size"], row["throughput"]),
                            textcoords="offset points",
                            xytext=(dx, dy),
                            ha="center",
                            color=color,
                            fontsize=11,
                        )
                    label = f"{device} ({mode})"
                    legend_handles[label] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker="o",
                        label=label,
                    )

                if device == "GPU":
                    ymin, ymax = ax.get_ylim()
                    for oom_x in sorted(gpu_oom_xs):
                        ax.axvline(
                            oom_x,
                            color=DEVICE_COLORS["GPU"],
                            linestyle="--",
                            linewidth=1.3,
                            alpha=0.45,
                        )
                        ax.annotate(
                            "OOM",
                            (oom_x, ymin + 0.12 * (ymax - ymin)),
                            rotation=90,
                            color=DEVICE_COLORS["GPU"],
                            fontsize=10,
                            ha="center",
                            va="bottom",
                            alpha=0.75,
                        )

            figure_legend_handles.update(legend_handles)

        add_compact_figure_legend(fig, figure_legend_handles, max_cols=3)
        fig.suptitle(
            f"Throughput on {task.upper()} at {int(sparsity)}% Sparsity",
            y=1.05,
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()


plot_scaling_panels(scaling_df)


# %% [markdown]
# ## 7) Optional Quick Checks

# %%
print(sorted(accuracy_df["model"].unique()))
print(sorted(final_tp_df["model"].unique()))
print(sorted(scaling_df["model"].unique()))


# %%
accuracy_df.head(10)


# %%
final_tp_df.head(10)


# %%
scaling_df.head(10)
