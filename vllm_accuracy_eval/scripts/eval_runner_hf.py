#!/usr/bin/env python3
"""
HuggingFace backend evaluation runner for GPU/CPU with statistical analysis.

Runs HF-based evaluation multiple times and calculates:
- Mean, Median, P90, P99, Std deviation
- Timing statistics
- Environment fingerprint for reproducibility

Usage:
    python vllm_accuracy_eval/scripts/eval_runner_hf.py --device gpu --iterations 1
    python vllm_accuracy_eval/scripts/eval_runner_hf.py --device cpu --iterations 1
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import functools

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

# Unbuffered print
def print_flush(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


# Resolve script directory
SCRIPT_DIR = Path(__file__).parent.parent

def get_results_dir(device: str) -> Path:
    return SCRIPT_DIR / "results" / f"{device}_hf"

def get_logs_dir(device: str) -> Path:
    return SCRIPT_DIR / "logs" / f"{device}_hf"


def get_environment_fingerprint() -> dict:
    """Capture environment info for reproducibility."""
    fingerprint = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
        "packages": {},
        "env_vars": {},
    }

    # Capture key package versions
    packages = ["torch", "lm_eval", "transformers", "datasets", "numpy", "accelerate"]
    for pkg in packages:
        try:
            if pkg == "torch":
                import torch
                fingerprint["packages"]["torch"] = torch.__version__
                fingerprint["cuda"] = {
                    "available": torch.cuda.is_available(),
                    "version": torch.version.cuda if torch.cuda.is_available() else None,
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            elif pkg == "lm_eval":
                import lm_eval
                fingerprint["packages"]["lm_eval"] = lm_eval.__version__
            elif pkg == "transformers":
                import transformers
                fingerprint["packages"]["transformers"] = transformers.__version__
            elif pkg == "datasets":
                import datasets
                fingerprint["packages"]["datasets"] = datasets.__version__
            elif pkg == "numpy":
                import numpy
                fingerprint["packages"]["numpy"] = numpy.__version__
            elif pkg == "accelerate":
                import accelerate
                fingerprint["packages"]["accelerate"] = accelerate.__version__
        except ImportError:
            fingerprint["packages"][pkg] = None

    # Capture relevant env vars
    env_vars = ["OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                "HF_DATASETS_CACHE", "TOKENIZERS_PARALLELISM"]
    for var in env_vars:
        fingerprint["env_vars"][var] = os.environ.get(var)

    return fingerprint


def get_model_checksum(model_path: str) -> Optional[str]:
    """Compute SHA256 of model config.json for verification."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return None


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

        # Parse table row
        parts = [p.strip() for p in line.split('|')]

        if len(parts) < 8:
            continue

        # lm-eval format: |Task|Version|Filter|n-shot|Metric|Value|Stderr|
        task = parts[1]
        if task and task.lower() not in ('tasks', 'task', 'groups', '-', ''):
            current_task = task

        if not current_task:
            continue

        # Find metric and value
        metric = parts[5] if len(parts) > 5 else ''
        value_str = parts[7] if len(parts) > 7 else ''

        if metric in ('acc', 'acc_norm', 'perplexity', 'word_perplexity',
                      'byte_perplexity', 'bits_per_byte'):
            try:
                value = float(value_str)
                if current_task not in results:
                    results[current_task] = {}
                results[current_task][metric] = value
            except (ValueError, TypeError):
                continue

    return results


def run_single_task_eval(model_path: str, task: str,
                         batch_size: int = 8, seed: int = 42,
                         env: dict = None) -> tuple[dict, float]:
    """Run evaluation for a single task and return parsed results with timing."""

    # Build command with HF backend (no --eval_backend flag = default HF)
    cmd = [
        "auto-round",
        "--model", model_path,
        "--eval",
        "--tasks", task,
        "--eval_bs", str(batch_size),
        "--seed", str(seed),
    ]

    print_flush(f"  Running: {' '.join(cmd)}")

    start_time = time.time()

    # Use Popen for real-time streaming output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        env=env
    )

    # Stream output line by line and collect for parsing
    output_lines = []
    for line in process.stdout:
        line = line.rstrip()
        print_flush(f"    {line}")
        output_lines.append(line)

    process.wait()
    elapsed = time.time() - start_time

    if process.returncode != 0:
        print_flush(f"  Error running {task} (exit code {process.returncode})")
        return {}, elapsed

    # Combine output for parsing
    full_output = "\n".join(output_lines)

    return parse_eval_output(full_output), elapsed


def run_single_eval(model_path: str, tasks: str,
                    batch_size: int = 8, seed: int = 42) -> tuple[dict, float]:
    """Run HF evaluation for all tasks and return aggregated results with timing."""

    # Build environment
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    print_flush(f"  OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")

    # Split tasks and run each separately
    task_list = [t.strip() for t in tasks.split(",")]

    all_results = {}
    total_elapsed = 0.0

    for task in task_list:
        print_flush(f"\n  Task: {task}")
        task_results, elapsed = run_single_task_eval(
            model_path, task, batch_size, seed, env
        )
        total_elapsed += elapsed

        if task_results:
            all_results.update(task_results)
            # Print task result
            for t, metrics in task_results.items():
                metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print_flush(f"    {t}: {metric_str} ({elapsed:.1f}s)")
        else:
            print_flush(f"    {task}: No results (error)")

    return all_results, total_elapsed


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


def print_results_table(stats: dict, timing_stats: dict = None, device: str = "cpu"):
    """Print formatted results table."""
    print_flush("\n" + "=" * 90)
    print_flush(f"HF EVALUATION RESULTS SUMMARY ({device.upper()})")
    print_flush("=" * 90)
    print_flush(f"{'Task':<20} {'Metric':<12} {'Mean':>10} {'Median':>10} "
          f"{'P90':>10} {'P99':>10} {'Std':>10}")
    print_flush("-" * 90)

    for task, metrics in sorted(stats.items()):
        for metric, stat in sorted(metrics.items()):
            print_flush(f"{task:<20} {metric:<12} {stat['mean']:>10.4f} "
                  f"{stat['median']:>10.4f} {stat['p90']:>10.4f} "
                  f"{stat['p99']:>10.4f} {stat['std']:>10.4f}")

    if timing_stats:
        print_flush("-" * 90)
        print_flush(f"{'Timing (seconds)':<20} {'total':<12} "
              f"{timing_stats['mean']:>10.1f} {timing_stats['median']:>10.1f} "
              f"{timing_stats['p90']:>10.1f} {timing_stats['p99']:>10.1f} "
              f"{timing_stats['std']:>10.1f}")
    print_flush("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace backend evaluation runner for GPU/CPU"
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        required=True,
        help="Device to run evaluation on (gpu or cpu)"
    )
    parser.add_argument(
        "--model",
        default="./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq",
        help="Path to quantized model (GPTQ format)"
    )
    parser.add_argument(
        "--tasks",
        default="lambada_openai,hellaswag,piqa",
        help="Comma-separated evaluation tasks"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of evaluation iterations (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (default: 16 for GPU, 8 for CPU)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (default: vllm_accuracy_eval/results/{device}_hf/w4a16_hf_stats.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="Skip environment fingerprinting (faster but less traceable)"
    )
    args = parser.parse_args()

    # Set device-specific defaults
    if args.batch_size is None:
        args.batch_size = 16 if args.device == "gpu" else 8

    # Set output paths based on device
    results_dir = get_results_dir(args.device)
    logs_dir = get_logs_dir(args.device)

    if args.output is None:
        args.output = str(results_dir / "w4a16_hf_stats.json")

    # Ensure directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Capture environment fingerprint
    fingerprint = None
    model_checksum = None
    if not args.skip_fingerprint:
        print_flush("Capturing environment fingerprint...")
        fingerprint = get_environment_fingerprint()
        model_checksum = get_model_checksum(args.model)

    print_flush("=" * 60)
    print_flush(f"HF Evaluation Runner ({args.device.upper()})")
    print_flush("=" * 60)
    print_flush(f"Device:     {args.device.upper()}")
    print_flush(f"Backend:    HuggingFace")
    print_flush(f"Model:      {args.model}")
    if model_checksum:
        print_flush(f"Model Hash: {model_checksum}")
    print_flush(f"Tasks:      {args.tasks}")
    print_flush(f"Iterations: {args.iterations}")
    print_flush(f"Batch Size: {args.batch_size}")
    print_flush(f"Seed:       {args.seed}")
    print_flush(f"Output:     {args.output}")
    if fingerprint:
        print_flush(f"PyTorch:    {fingerprint['packages'].get('torch', 'N/A')}")
        print_flush(f"lm-eval:    {fingerprint['packages'].get('lm_eval', 'N/A')}")
    print_flush("=" * 60)

    # Collect results from all iterations
    all_results = []
    all_timings = []

    for i in range(args.iterations):
        print_flush(f"\n{'='*60}")
        print_flush(f"Iteration {i + 1}/{args.iterations}")
        print_flush("=" * 60)

        results, elapsed = run_single_eval(
            args.model, args.tasks,
            args.batch_size, seed=args.seed
        )
        if results:
            all_results.append(results)
            all_timings.append(elapsed)
            print_flush(f"Iteration {i + 1} results: {results}")
            print_flush(f"Iteration {i + 1} time: {elapsed:.1f}s")
        else:
            print_flush(f"Warning: Iteration {i + 1} returned no results")

    if not all_results:
        print_flush("Error: No successful evaluations!")
        sys.exit(1)

    # Calculate statistics per task/metric
    stats = {}
    for task in all_results[0].keys():
        stats[task] = {}
        for metric in all_results[0][task].keys():
            values = [r[task][metric] for r in all_results
                     if task in r and metric in r[task]]
            stats[task][metric] = calculate_statistics(values)

    timing_stats = calculate_statistics(all_timings)

    # Print results table
    print_results_table(stats, timing_stats, args.device)

    # Save to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "backend": "hf",
        "config": {
            "model": args.model,
            "model_checksum": model_checksum,
            "tasks": args.tasks,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "seed": args.seed
        },
        "raw_results": all_results,
        "raw_timings": all_timings,
        "statistics": stats,
        "timing_statistics": timing_stats,
    }

    # Include environment fingerprint for reproducibility
    if fingerprint:
        output_data["environment"] = fingerprint

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print_flush(f"\nResults saved to: {args.output}")

    # Print reproducibility summary
    if fingerprint:
        print_flush("\nReproducibility Info:")
        print_flush(f"  Model checksum: {model_checksum or 'N/A'}")
        print_flush(f"  Environment fingerprint included in output JSON")


if __name__ == "__main__":
    main()
