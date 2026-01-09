#!/usr/bin/env python3
"""
GPU vs CPU Comparison Tool for vLLM Evaluation Results.

Reads:
  - GPU results from: vllm_accuracy_eval/results/gpu/w4a16_vllm_stats.json
  - CPU results from: vllm_accuracy_eval/results/cpu/w4a16_vllm_stats.json

Generates:
  - Comparison report: vllm_accuracy_eval/results/comparison_report.md

Usage:
    python vllm_accuracy_eval/scripts/compare_results.py
    python vllm_accuracy_eval/scripts/compare_results.py --threshold 1.0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# Resolve script directory
SCRIPT_DIR = Path(__file__).parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
GPU_RESULTS = RESULTS_DIR / "gpu" / "w4a16_vllm_stats.json"
CPU_RESULTS = RESULTS_DIR / "cpu" / "w4a16_vllm_stats.json"

# Threshold for flagging significant differences (in percentage points)
SIGNIFICANT_DIFF_THRESHOLD = 0.5


def load_json(filepath: Path) -> dict:
    """Load and validate JSON results file."""
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        data = json.load(f)

    if "statistics" not in data:
        print(f"Error: Invalid results file (missing 'statistics'): {filepath}")
        sys.exit(1)

    return data


def calculate_delta(gpu_value: float, cpu_value: float) -> tuple[float, str]:
    """Calculate delta and formatted string.

    Returns (delta_pp, formatted_string)
    Delta is CPU - GPU (positive means CPU is higher)
    """
    delta = (cpu_value - gpu_value) * 100  # Convert to percentage points
    sign = "+" if delta >= 0 else ""
    return delta, f"{sign}{delta:.2f} pp"


def is_significant(delta_pp: float) -> bool:
    """Check if delta exceeds significance threshold."""
    return abs(delta_pp) > SIGNIFICANT_DIFF_THRESHOLD


def generate_comparison_report(gpu_data: dict, cpu_data: dict) -> str:
    """Generate markdown comparison report."""

    gpu_stats = gpu_data["statistics"]
    cpu_stats = cpu_data["statistics"]

    # Get common tasks
    common_tasks = set(gpu_stats.keys()) & set(cpu_stats.keys())
    if not common_tasks:
        return "Error: No common tasks between GPU and CPU results."

    # Build report
    lines = []
    lines.append("# GPU vs CPU vLLM Evaluation Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration section
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | GPU | CPU |")
    lines.append("|-----------|-----|-----|")

    gpu_config = gpu_data.get("config", {})
    cpu_config = cpu_data.get("config", {})

    lines.append(f"| Model | `{gpu_config.get('model', 'N/A')}` | `{cpu_config.get('model', 'N/A')}` |")
    lines.append(f"| Tasks | {gpu_config.get('tasks', 'N/A')} | {cpu_config.get('tasks', 'N/A')} |")
    lines.append(f"| Iterations | {gpu_config.get('iterations', 'N/A')} | {cpu_config.get('iterations', 'N/A')} |")
    lines.append(f"| Batch Size | {gpu_config.get('batch_size', 'N/A')} | {cpu_config.get('batch_size', 'N/A')} |")
    lines.append(f"| Seed | {gpu_config.get('seed', 'N/A')} | {cpu_config.get('seed', 'N/A')} |")
    lines.append(f"| Backend | vLLM | vLLM |")
    lines.append("")

    # Primary metrics comparison
    lines.append("## Primary Metrics Comparison")
    lines.append("")
    lines.append("| Task | Metric | GPU | CPU | Delta | Status |")
    lines.append("|------|--------|-----|-----|-------|--------|")

    primary_metrics = [
        ("lambada_openai", "acc"),
        ("hellaswag", "acc_norm"),
        ("piqa", "acc"),
    ]

    significant_diffs = []

    for task, metric in primary_metrics:
        if task in gpu_stats and task in cpu_stats:
            gpu_task = gpu_stats[task]
            cpu_task = cpu_stats[task]

            if metric in gpu_task and metric in cpu_task:
                gpu_val = gpu_task[metric].get("mean", 0)
                cpu_val = cpu_task[metric].get("mean", 0)
                delta_pp, delta_str = calculate_delta(gpu_val, cpu_val)
                is_sig = is_significant(delta_pp)
                status = "**DIFF**" if is_sig else "OK"

                if is_sig:
                    significant_diffs.append((task, metric, delta_pp))

                lines.append(f"| {task} | {metric} | {gpu_val:.4f} | {cpu_val:.4f} | {delta_str} | {status} |")

    lines.append("")

    # Detailed comparison table
    lines.append("## Detailed Comparison (All Metrics)")
    lines.append("")
    lines.append("| Task | Metric | GPU Mean | CPU Mean | Delta | Significant? |")
    lines.append("|------|--------|----------|----------|-------|--------------|")

    for task in sorted(common_tasks):
        gpu_task = gpu_stats[task]
        cpu_task = cpu_stats.get(task, {})

        common_metrics = set(gpu_task.keys()) & set(cpu_task.keys())

        for metric in sorted(common_metrics):
            gpu_mean = gpu_task[metric].get("mean", 0)
            cpu_mean = cpu_task[metric].get("mean", 0)

            delta_pp, delta_str = calculate_delta(gpu_mean, cpu_mean)
            is_sig = is_significant(delta_pp)
            sig_marker = "**YES**" if is_sig else "No"

            lines.append(f"| {task} | {metric} | {gpu_mean:.4f} | {cpu_mean:.4f} | {delta_str} | {sig_marker} |")

    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")

    if significant_diffs:
        lines.append(f"**WARNING**: {len(significant_diffs)} metric(s) showed significant "
                    f"differences (>{SIGNIFICANT_DIFF_THRESHOLD} pp):")
        lines.append("")
        for task, metric, delta in significant_diffs:
            direction = "higher" if delta > 0 else "lower"
            lines.append(f"- **{task}/{metric}**: CPU is {abs(delta):.2f} pp {direction} than GPU")
        lines.append("")
    else:
        lines.append("**All metrics are within acceptable tolerance** "
                    f"(<{SIGNIFICANT_DIFF_THRESHOLD} pp difference).")
        lines.append("")
        lines.append("GPU and CPU vLLM evaluation results are **consistent**.")
        lines.append("")

    # Timing comparison
    gpu_timing = gpu_data.get("timing_statistics", {})
    cpu_timing = cpu_data.get("timing_statistics", {})

    if gpu_timing or cpu_timing:
        lines.append("## Timing Comparison")
        lines.append("")
        lines.append("| Statistic | GPU (seconds) | CPU (seconds) | Ratio (CPU/GPU) |")
        lines.append("|-----------|---------------|---------------|-----------------|")

        for stat in ["mean", "median", "p90", "p99"]:
            gpu_val = gpu_timing.get(stat, 0)
            cpu_val = cpu_timing.get(stat, 0)
            ratio = cpu_val / gpu_val if gpu_val > 0 else float('inf')
            lines.append(f"| {stat.upper()} | {gpu_val:.1f} | {cpu_val:.1f} | {ratio:.1f}x |")

        lines.append("")

    # Data sources
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- **GPU Results**: `results/gpu/w4a16_vllm_stats.json`")
    lines.append(f"  - Timestamp: {gpu_data.get('timestamp', 'N/A')}")
    lines.append(f"- **CPU Results**: `results/cpu/w4a16_vllm_stats.json`")
    lines.append(f"  - Timestamp: {cpu_data.get('timestamp', 'N/A')}")
    lines.append("")

    # Notes
    lines.append("---")
    lines.append("")
    lines.append(f"*Significance threshold: {SIGNIFICANT_DIFF_THRESHOLD} percentage points (pp)*")
    lines.append("")
    lines.append("*Both evaluations use vLLM backend with identical seed for fair comparison.*")
    lines.append("")

    return "\n".join(lines)


def main():
    global SIGNIFICANT_DIFF_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Compare GPU vs CPU vLLM evaluation results"
    )
    parser.add_argument(
        "--gpu-file",
        default=str(GPU_RESULTS),
        help="Path to GPU results JSON file"
    )
    parser.add_argument(
        "--cpu-file",
        default=str(CPU_RESULTS),
        help="Path to CPU results JSON file"
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "comparison_report.md"),
        help="Output path for comparison report"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Significance threshold in percentage points (default: 0.5)"
    )
    args = parser.parse_args()

    SIGNIFICANT_DIFF_THRESHOLD = args.threshold

    print("=" * 60)
    print("GPU vs CPU vLLM Comparison Tool")
    print("=" * 60)
    print(f"GPU Results: {args.gpu_file}")
    print(f"CPU Results: {args.cpu_file}")
    print(f"Output:      {args.output}")
    print(f"Threshold:   {args.threshold} pp")
    print("=" * 60)

    # Load data
    gpu_data = load_json(Path(args.gpu_file))
    cpu_data = load_json(Path(args.cpu_file))

    # Generate report
    report = generate_comparison_report(gpu_data, cpu_data)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nComparison report saved to: {args.output}")

    # Also print to console
    print("\n" + "=" * 60)
    print("REPORT PREVIEW")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
