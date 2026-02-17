#!/usr/bin/env python3
"""
Analyze sparse training results.

Aggregates results from all runs and generates comparison tables.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
import re
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def parse_run_dir(run_dir: Path) -> Dict:
    """
    Extract metadata from a run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dictionary of metadata
    """
    metadata = {
        'run_id': run_dir.name,
        'run_dir': str(run_dir),
    }
    
    # Parse run_id for metadata
    # Format: cs3_{model}_{task}_{method}_s{sparsity}_{mode}_{sha}_{timestamp}
    parts = run_dir.name.split('_')
    try:
        if len(parts) >= 7:
            metadata['hardware'] = parts[0]  # cs3
            metadata['model'] = parts[1]
            metadata['task'] = parts[2]
            metadata['method'] = parts[3]
            
            # Extract sparsity
            sparsity_str = parts[4]
            if sparsity_str.startswith('s'):
                metadata['sparsity'] = int(sparsity_str[1:]) / 100.0
            
            metadata['mode'] = parts[5]
            metadata['git_sha'] = parts[6]
    except:
        pass
    
    # Load sparse config if available
    sparse_config_path = run_dir / "sparse_config.json"
    if sparse_config_path.exists():
        with open(sparse_config_path, 'r') as f:
            sparse_config = json.load(f)
            metadata.update(sparse_config)
    
    # Load validation report if available
    validation_path = run_dir / "validation_report.json"
    if validation_path.exists():
        with open(validation_path, 'r') as f:
            validation = json.load(f)
            metadata['sparsity_valid'] = validation.get('global', {}).get('all_valid', False)
            metadata['sparsity_violations'] = validation.get('global', {}).get('total_violations', 0)
    
    # Parse training log for metrics
    train_log_path = run_dir / "train.log"
    if train_log_path.exists():
        metrics = parse_train_log(train_log_path)
        metadata.update(metrics)
    
    return metadata


def parse_train_log(log_path: Path) -> Dict:
    """
    Parse Cerebras training log for metrics.
    
    Args:
        log_path: Path to training log
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract final loss
    loss_pattern = r'loss[:\s]+([0-9.]+)'
    losses = re.findall(loss_pattern, content, re.IGNORECASE)
    if losses:
        try:
            metrics['final_loss'] = float(losses[-1])
        except:
            pass
    
    # Extract throughput
    throughput_pattern = r'throughput[:\s]+([0-9.]+)'
    throughputs = re.findall(throughput_pattern, content, re.IGNORECASE)
    if throughputs:
        try:
            metrics['throughput'] = float(throughputs[-1])
        except:
            pass
    
    # Extract training time
    time_pattern = r'Total time[:\s]+([0-9.]+)'
    times = re.findall(time_pattern, content, re.IGNORECASE)
    if times:
        try:
            metrics['training_time'] = float(times[-1])
        except:
            pass
    
    return metrics


def collect_results(results_dir: Path) -> pd.DataFrame:
    """
    Collect results from all runs.
    
    Args:
        results_dir: Path to results directory
    
    Returns:
        DataFrame of results
    """
    all_results = []
    
    # Iterate over all run directories
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        if not run_dir.name.startswith('cs3_'):
            continue
        
        print(f"Processing: {run_dir.name}")
        
        try:
            metadata = parse_run_dir(run_dir)
            all_results.append(metadata)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df


def generate_summary_tables(df: pd.DataFrame, output_dir: Path):
    """
    Generate summary tables from results.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save tables
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall summary
    summary_path = output_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Group by model, task, method, sparsity
    if all(col in df.columns for col in ['model', 'task', 'method', 'sparsity']):
        grouped = df.groupby(['model', 'task', 'method', 'sparsity']).agg({
            'final_loss': ['mean', 'std', 'min'],
            'throughput': ['mean', 'std'],
            'sparsity_valid': 'sum',
        }).reset_index()
        
        grouped_path = output_dir / "grouped_summary.csv"
        grouped.to_csv(grouped_path, index=False)
        print(f"✓ Grouped summary saved to: {grouped_path}")
    
    # Sparsity vs performance table
    if 'sparsity' in df.columns and 'final_loss' in df.columns:
        sparsity_perf = df.groupby('sparsity').agg({
            'final_loss': ['mean', 'std'],
            'throughput': ['mean', 'std'],
        }).reset_index()
        
        sparsity_perf_path = output_dir / "sparsity_vs_performance.csv"
        sparsity_perf.to_csv(sparsity_perf_path, index=False)
        print(f"✓ Sparsity vs performance saved to: {sparsity_perf_path}")


def plot_results(df: pd.DataFrame, output_dir: Path):
    """
    Generate plots from results.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # Plot 1: Sparsity vs Loss
    if 'sparsity' in df.columns and 'final_loss' in df.columns:
        plt.figure(figsize=(10, 6))
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            grouped = method_df.groupby('sparsity').agg({
                'final_loss': ['mean', 'std']
            }).reset_index()
            
            plt.errorbar(
                grouped['sparsity'],
                grouped[('final_loss', 'mean')],
                yerr=grouped[('final_loss', 'std')],
                label=method,
                marker='o',
                capsize=5
            )
        
        plt.xlabel('Sparsity')
        plt.ylabel('Final Loss')
        plt.title('Sparsity vs Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / "sparsity_vs_loss.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to: {plot_path}")
    
    # Plot 2: Sparsity vs Throughput
    if 'sparsity' in df.columns and 'throughput' in df.columns:
        plt.figure(figsize=(10, 6))
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            grouped = method_df.groupby('sparsity').agg({
                'throughput': ['mean', 'std']
            }).reset_index()
            
            plt.errorbar(
                grouped['sparsity'],
                grouped[('throughput', 'mean')],
                yerr=grouped[('throughput', 'std')],
                label=method,
                marker='o',
                capsize=5
            )
        
        plt.xlabel('Sparsity')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Sparsity vs Throughput')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / "sparsity_vs_throughput.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to: {plot_path}")
    
    # Plot 3: Method comparison heatmap
    if all(col in df.columns for col in ['method', 'sparsity', 'final_loss']):
        pivot_data = df.pivot_table(
            values='final_loss',
            index='method',
            columns='sparsity',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title('Loss by Method and Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Method')
        
        plot_path = output_dir / "method_comparison_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sparse training results")
    parser.add_argument("--results_dir", type=str, default="./results/runs",
                       help="Directory containing run results")
    parser.add_argument("--output_dir", type=str, default="./results/analysis",
                       help="Directory to save analysis outputs")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("Collecting results...")
    df = collect_results(results_dir)
    
    print(f"\nCollected {len(df)} runs")
    print(f"Columns: {', '.join(df.columns)}")
    
    print("\nGenerating summary tables...")
    generate_summary_tables(df, output_dir)
    
    if args.plot:
        print("\nGenerating plots...")
        plot_results(df, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
