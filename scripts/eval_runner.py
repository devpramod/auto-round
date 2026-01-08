#!/usr/bin/env python3
"""
Multi-iteration evaluation runner with statistical analysis.

Runs evaluation multiple times and calculates:
- Mean, Median, P90, P99, Std deviation

Usage:
    python scripts/eval_runner.py --iterations 10
    python scripts/eval_runner.py --model ./llama3.1_8b_int4/... --iterations 5
"""

import argparse
import json
import subprocess
import sys
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np


def parse_eval_output(output: str) -> dict:
    """
    Parse lm-eval table output to extract metrics.

    Returns dict like:
    {
        "lambada_openai": {"acc": 0.7254, "perplexity": 3.4022},
        "hellaswag": {"acc_norm": 0.5821},
        "piqa": {"acc": 0.7612}
    }
    """
    results = {}
    lines = output.strip().split('\n')
    current_task = None

    for line in lines:
        if '|' not in line:
            continue

        # Skip header/separator lines
        if '---' in line or 'Tasks' in line or 'Metric' in line:
            continue

        # Parse table row - keep empty parts for position tracking
        parts = [p.strip() for p in line.split('|')]

        if len(parts) < 8:
            continue

        # lm-eval format: |Task|Version|Filter|n-shot|Metric|↑/↓|Value|±|Stderr|
        # parts[0] is empty (before first |), so task is parts[1]
        task = parts[1]
        if task and task.lower() not in ('tasks', 'task', 'groups', '-', ''):
            current_task = task

        if not current_task:
            continue

        # Find metric (usually parts[5]) and value (parts[7])
        metric = parts[5] if len(parts) > 5 else ''
        value_str = parts[7] if len(parts) > 7 else ''

        if metric in ('acc', 'acc_norm', 'perplexity', 'word_perplexity', 'byte_perplexity', 'bits_per_byte'):
            try:
                value = float(value_str)
                if current_task not in results:
                    results[current_task] = {}
                results[current_task][metric] = value
            except (ValueError, TypeError):
                continue

    return results


def run_single_eval(model_path: str, tasks: str, batch_size: int = 16,
                    seed: int = 42) -> dict:
    """Run a single evaluation and return parsed results."""
    cmd = [
        "auto-round",
        "--model", model_path,
        "--eval",
        "--tasks", tasks,
        "--eval_bs", str(batch_size),
        "--seed", str(seed)
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    if result.returncode != 0:
        print(f"Error running evaluation:\n{result.stderr}")
        return {}

    # Combine stdout and stderr for parsing (lm-eval may print to either)
    full_output = result.stdout + "\n" + result.stderr
    print(full_output)

    return parse_eval_output(full_output)


def calculate_statistics(values: list) -> dict:
    """Calculate mean, median, P90, P99, and std for a list of values."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values)
    }


def print_results_table(stats: dict):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Task':<20} {'Metric':<12} {'Mean':>10} {'Median':>10} {'P90':>10} {'P99':>10} {'Std':>10}")
    print("-" * 90)

    for task, metrics in sorted(stats.items()):
        for metric, stat in sorted(metrics.items()):
            print(f"{task:<20} {metric:<12} {stat['mean']:>10.4f} {stat['median']:>10.4f} "
                  f"{stat['p90']:>10.4f} {stat['p99']:>10.4f} {stat['std']:>10.4f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Multi-iteration evaluation runner")
    parser.add_argument(
        "--model",
        default="./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq",
        help="Path to quantized model"
    )
    parser.add_argument(
        "--tasks",
        default="lambada_openai,hellaswag,piqa",
        help="Comma-separated evaluation tasks"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of evaluation iterations (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)"
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Iteration Evaluation Runner")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Tasks:      {args.tasks}")
    print(f"Iterations: {args.iterations}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Seed:       {args.seed}")
    print("=" * 60)

    # Collect results from all iterations
    all_results = []

    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i + 1}/{args.iterations}")
        print("=" * 60)

        results = run_single_eval(
            args.model, args.tasks, args.batch_size,
            seed=args.seed
        )
        if results:
            all_results.append(results)
            print(f"Iteration {i + 1} results: {results}")
        else:
            print(f"Warning: Iteration {i + 1} returned no results")

    if not all_results:
        print("Error: No successful evaluations!")
        sys.exit(1)

    # Calculate statistics per task/metric
    stats = {}
    for task in all_results[0].keys():
        stats[task] = {}
        for metric in all_results[0][task].keys():
            values = [r[task][metric] for r in all_results if task in r and metric in r[task]]
            stats[task][metric] = calculate_statistics(values)

    # Print results table
    print_results_table(stats)

    # Save to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "tasks": args.tasks,
            "iterations": args.iterations,
            "batch_size": args.batch_size
        },
        "raw_results": all_results,
        "statistics": stats
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
