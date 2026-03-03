from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd()
TABLES = ROOT / "tables"

GPU_ACC_PATH = TABLES / "run_gpu_sparse_acc.csv"
GPU_TP_PATH = TABLES / "run_gpu_sparse_tp_scale.csv"
CS_ACC_PATH = TABLES / "run_cerebras_sparse_results_by_task.csv"
CS_TP_PATH = TABLES / "run_cerebras_sparse_throughput_estimates_by_batch_size.csv"

TASK_METRIC_MAP = {
    "boolq": "acc_none",
    "hellaswag": "acc_norm_none",
    "gsm8k": "exact_match_flexible_extract",
}

MODEL_MAP = {
    "llama3.1-8b": "llama3",
    "llama3": "llama3",
    "mistral-7b": "mistral",
    "mistral": "mistral",
    "mixtral-8x7b": "mixtral",
    "mixtral": "mixtral",
}

MODE_MAP = {
    "sparse_to_dense": "Sparse to Dense",
    "sparse_to_sparse": "Sparse to Sparse",
    "sparse to dense": "Sparse to Dense",
    "sparse to sparse": "Sparse to Sparse",
}


def normalize_model(value):
    return MODEL_MAP.get(value, value)


def normalize_mode(value):
    return MODE_MAP.get(value, value)


def format_batch_label(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.apply(lambda v: f"batch={int(v)}" if pd.notna(v) else "batch=unknown")


def build_hardware_specs():
    return pd.DataFrame(
        [
            {
                "Device": "Cerebras CS-3",
                "Processor": "Wafer-Scale Engine 3 (WSE-3)",
                "On-device memory": "44 GB SRAM",
                "Compute summary": "~125 PFLOPS AI compute (sparse, vendor-cited)",
                "Core / compute units": "~900,000 AI-optimized cores",
                "Memory bandwidth": "~21 PB/s on-wafer bandwidth",
                "Form factor note": "Single appliance system",
            },
            {
                "Device": "NVIDIA A100 40 GB",
                "Processor": "Ampere GA100",
                "On-device memory": "40 GB HBM2",
                "Compute summary": "Up to ~312 TFLOPS Tensor Core (FP16/BF16 with sparsity)",
                "Core / compute units": "6912 CUDA cores, 432 Tensor Cores",
                "Memory bandwidth": "Up to ~1.6 TB/s",
                "Form factor note": "PCIe/SXM variants",
            },
        ]
    )


def load_tables():
    for path in [GPU_ACC_PATH, GPU_TP_PATH, CS_ACC_PATH, CS_TP_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    return {
        "gpu_acc": pd.read_csv(GPU_ACC_PATH),
        "gpu_tp": pd.read_csv(GPU_TP_PATH),
        "cs_acc": pd.read_csv(CS_ACC_PATH),
        "cs_tp": pd.read_csv(CS_TP_PATH),
    }


def build_gpu_accuracy_frame(df):
    rows = []
    for _, row in df.iterrows():
        metric_col = TASK_METRIC_MAP[row["task"]]
        metric_value = row.get(metric_col)
        if pd.isna(metric_value):
            continue
        rows.append(
            {
                "device": "GPU",
                "task": row["task"],
                "sparsity": row["sparsity_fraction"],
                "metric_pct": float(metric_value) * 100.0,
                "training_mode": normalize_mode(row["method"]),
                "model": normalize_model(row["model"]),
                "run_id": row["run_id"],
            }
        )
    return pd.DataFrame(rows)


def build_cs_accuracy_frame(df):
    out = df.rename(
        columns={
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Evaluation Accuracy": "metric_raw",
            "Training Mode": "training_mode",
            "Model Name": "model",
            "Run Identifier": "run_id",
        }
    ).copy()
    out["device"] = "Cerebras"
    out["metric_pct"] = pd.to_numeric(out["metric_raw"], errors="coerce") * 100.0
    out["training_mode"] = out["training_mode"].map(normalize_mode)
    out["model"] = out["model"].map(normalize_model)
    return out[["device", "task", "sparsity", "metric_pct", "training_mode", "model", "run_id"]]


def build_gpu_final_throughput_frame(df):
    out = df.copy()
    out["global_batch_size"] = out["micro_batch"] * out["world_size"]
    out["samples_per_s"] = out["tokens_per_s"] / out["seq_len"]
    idx = out.groupby("run_id")["micro_batch"].idxmax()
    final_points = out.loc[idx].copy()
    final_points["device"] = "GPU"
    final_points["training_mode"] = final_points["method"].map(normalize_mode)
    final_points["model"] = final_points["model"].map(normalize_model)
    final_points["batch_label"] = format_batch_label(final_points["global_batch_size"])
    return final_points[
        [
            "device",
            "task",
            "sparsity_fraction",
            "samples_per_s",
            "training_mode",
            "model",
            "batch_label",
            "run_id",
        ]
    ].rename(columns={"sparsity_fraction": "sparsity"})


def build_cs_final_throughput_frame(df):
    out = df.rename(
        columns={
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Observed Training Throughput at Batch Size 256 (Samples Per Second)": "samples_per_s",
            "Training Mode": "training_mode",
            "Model Name": "model",
            "Run Identifier": "run_id",
            "Observed Batch Size": "batch_size",
        }
    ).copy()
    out["device"] = "Cerebras"
    out["training_mode"] = out["training_mode"].map(normalize_mode)
    out["model"] = out["model"].map(normalize_model)
    out["batch_label"] = format_batch_label(out["batch_size"])
    return out[
        ["device", "task", "sparsity", "samples_per_s", "training_mode", "model", "batch_label", "run_id"]
    ]


def build_gpu_scaling_frame(df):
    out = df.copy()
    out["batch_size"] = out["micro_batch"] * out["world_size"]
    out["samples_per_s"] = out["tokens_per_s"] / out["seq_len"]
    out["device"] = "GPU"
    out["training_mode"] = out["method"].map(normalize_mode)
    out["model"] = out["model"].map(normalize_model)
    return out[
        ["device", "task", "sparsity_fraction", "batch_size", "samples_per_s", "training_mode", "model", "run_id"]
    ].rename(columns={"sparsity_fraction": "sparsity"})


def build_cs_scaling_frame(df):
    batch_cols = {
        32: "Estimated Training Throughput at Batch Size 32 (Samples Per Second, Linear Estimate)",
        64: "Estimated Training Throughput at Batch Size 64 (Samples Per Second, Linear Estimate)",
        128: "Estimated Training Throughput at Batch Size 128 (Samples Per Second, Linear Estimate)",
        256: "Estimated Training Throughput at Batch Size 256 (Samples Per Second, Linear Estimate)",
    }
    renamed = df.rename(
        columns={
            "Task Name": "task",
            "Sparsity Percentage": "sparsity",
            "Training Mode": "training_mode",
            "Model Name": "model",
            "Run Identifier": "run_id",
        }
    ).copy()
    rows = []
    for _, row in renamed.iterrows():
        for batch_size, col in batch_cols.items():
            rows.append(
                {
                    "device": "Cerebras",
                    "task": row["task"],
                    "sparsity": row["sparsity"],
                    "batch_size": batch_size,
                    "samples_per_s": row[col],
                    "training_mode": normalize_mode(row["training_mode"]),
                    "model": normalize_model(row["model"]),
                    "run_id": row["run_id"],
                }
            )
    return pd.DataFrame(rows)


def build_plot_frames():
    tables = load_tables()

    gpu_acc_plot = build_gpu_accuracy_frame(tables["gpu_acc"])
    cs_acc_plot = build_cs_accuracy_frame(tables["cs_acc"])
    acc_plot = pd.concat([gpu_acc_plot, cs_acc_plot], ignore_index=True)

    gpu_final_tp_plot = build_gpu_final_throughput_frame(tables["gpu_tp"])
    cs_final_tp_plot = build_cs_final_throughput_frame(tables["cs_acc"])
    final_tp_plot = pd.concat([gpu_final_tp_plot, cs_final_tp_plot], ignore_index=True)

    gpu_scaling_plot = build_gpu_scaling_frame(tables["gpu_tp"])
    cs_scaling_plot = build_cs_scaling_frame(tables["cs_tp"])
    scaling_plot = pd.concat([gpu_scaling_plot, cs_scaling_plot], ignore_index=True)

    common_models = sorted(set(gpu_acc_plot["model"]) & set(cs_acc_plot["model"]))
    acc_plot = acc_plot[acc_plot["model"].isin(common_models)].copy()
    final_tp_plot = final_tp_plot[final_tp_plot["model"].isin(common_models)].copy()
    scaling_plot = scaling_plot[scaling_plot["model"].isin(common_models)].copy()

    return {
        "hardware_specs": build_hardware_specs(),
        "acc_plot": acc_plot,
        "final_tp_plot": final_tp_plot,
        "scaling_plot": scaling_plot,
        "common_models": common_models,
    }


def plot_accuracy(acc_plot):
    g = sns.relplot(
        data=acc_plot.sort_values(["task", "device", "model", "training_mode", "sparsity"]),
        x="sparsity",
        y="metric_pct",
        row="task",
        col="device",
        hue="training_mode",
        style="model",
        kind="line",
        marker="o",
        dashes=False,
        height=4.0,
        aspect=1.25,
        facet_kws={"sharey": "row"},
    )
    g.set_axis_labels("Sparsity (%)", "Task performance (%)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle("Sparse Finetuning Accuracy Across Devices")
    return g


def plot_final_throughput(final_tp_plot):
    g = sns.relplot(
        data=final_tp_plot.sort_values(["task", "device", "model", "training_mode", "sparsity"]),
        x="sparsity",
        y="samples_per_s",
        row="task",
        col="device",
        hue="training_mode",
        style="model",
        kind="line",
        marker="o",
        dashes=False,
        height=4.0,
        aspect=1.25,
        facet_kws={"sharey": "row"},
    )
    g.set_axis_labels("Sparsity (%)", "Training throughput (samples/s)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for (task, device), ax in g.axes_dict.items():
        subset = final_tp_plot[(final_tp_plot["task"] == task) & (final_tp_plot["device"] == device)]
        for _, row in subset.iterrows():
            ax.annotate(
                row["batch_label"],
                (row["sparsity"], row["samples_per_s"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                alpha=0.85,
            )

    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle("Final Training Throughput Across Devices")
    return g


def plot_scaling(scaling_plot):
    grids = []
    for sparsity in sorted(scaling_plot["sparsity"].dropna().unique()):
        subset = scaling_plot[scaling_plot["sparsity"] == sparsity].sort_values(
            ["task", "device", "model", "training_mode", "batch_size"]
        )
        g = sns.relplot(
            data=subset,
            x="batch_size",
            y="samples_per_s",
            row="task",
            col="device",
            hue="training_mode",
            style="model",
            kind="line",
            marker="o",
            dashes=False,
            height=4.0,
            aspect=1.25,
            facet_kws={"sharey": "row"},
        )
        g.set(xscale="log", xticks=sorted(subset["batch_size"].dropna().unique()))
        g.set_axis_labels("Effective batch size (log2 scale)", "Training throughput (samples/s)")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        for ax in g.axes.flatten():
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        g.fig.subplots_adjust(top=0.93)
        g.fig.suptitle(f"Training Throughput Scaling at {sparsity}% Sparsity")
        grids.append(g)
    return grids


def main():
    frames = build_plot_frames()
    print("Common model families:", ", ".join(frames["common_models"]))
    print()
    print("Hardware reference table:")
    print(frames["hardware_specs"].to_string(index=False))

    plot_accuracy(frames["acc_plot"])
    plot_final_throughput(frames["final_tp_plot"])
    plot_scaling(frames["scaling_plot"])
    plt.show()


if __name__ == "__main__":
    main()
