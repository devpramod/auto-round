#!/usr/bin/env python3
"""
Unified vLLM evaluation runner for GPU and CPU with statistical analysis.

Runs vLLM-based evaluation multiple times and calculates:
- Mean, Median, P90, P99, Std deviation
- Timing statistics
- Environment fingerprint for reproducibility

Usage:
    python vllm_accuracy_eval/scripts/eval_runner_vllm.py --device gpu --iterations 5
    python vllm_accuracy_eval/scripts/eval_runner_vllm.py --device cpu --iterations 5
    python vllm_accuracy_eval/scripts/eval_runner_vllm.py --device gpu --verify  # Pre-flight check
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
import numpy as np


# Resolve script directory
SCRIPT_DIR = Path(__file__).parent.parent
RESULTS_GPU_DIR = SCRIPT_DIR / "results" / "gpu"
RESULTS_CPU_DIR = SCRIPT_DIR / "results" / "cpu"
LOGS_GPU_DIR = SCRIPT_DIR / "logs" / "gpu"
LOGS_CPU_DIR = SCRIPT_DIR / "logs" / "cpu"


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
    packages = ["torch", "vllm", "lm_eval", "transformers", "datasets", "numpy", "accelerate"]
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
                if torch.cuda.is_available():
                    fingerprint["cuda"]["devices"] = [
                        {"id": i, "name": torch.cuda.get_device_name(i)}
                        for i in range(torch.cuda.device_count())
                    ]
            elif pkg == "vllm":
                import vllm
                fingerprint["packages"]["vllm"] = vllm.__version__
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
    env_vars = ["VLLM_TARGET_DEVICE", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
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


def run_single_task_eval(model_path: str, task: str, device: str,
                         batch_size: int = 16, seed: int = 42,
                         env: dict = None) -> tuple[dict, float]:
    """Run evaluation for a single task and return parsed results with timing."""

    # Build command with vLLM backend
    cmd = [
        "auto-round",
        "--model", model_path,
        "--eval",
        "--eval_backend", "vllm",
        "--tasks", task,
        "--eval_bs", str(batch_size),
        "--seed", str(seed),
    ]

    print(f"  Running: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  Error running {task}:\n{result.stderr[-500:]}")
        return {}, elapsed

    # Combine stdout and stderr for parsing
    full_output = result.stdout + "\n" + result.stderr

    return parse_eval_output(full_output), elapsed


def run_single_eval(model_path: str, tasks: str, device: str,
                    batch_size: int = 16, seed: int = 42) -> tuple[dict, float]:
    """Run vLLM evaluation for all tasks and return aggregated results with timing."""

    # Build environment
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    if device == "cpu":
        env["VLLM_TARGET_DEVICE"] = "cpu"
        env["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
        print(f"  VLLM_TARGET_DEVICE=cpu, OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")

    # Split tasks and run each separately (workaround for lm-eval task parsing)
    task_list = [t.strip() for t in tasks.split(",")]

    all_results = {}
    total_elapsed = 0.0

    for task in task_list:
        print(f"\n  Task: {task}")
        task_results, elapsed = run_single_task_eval(
            model_path, task, device, batch_size, seed, env
        )
        total_elapsed += elapsed

        if task_results:
            all_results.update(task_results)
            # Print task result
            for t, metrics in task_results.items():
                metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"    {t}: {metric_str} ({elapsed:.1f}s)")
        else:
            print(f"    {task}: No results (error)")

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


def print_results_table(stats: dict, timing_stats: dict = None):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("vLLM EVALUATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Task':<20} {'Metric':<12} {'Mean':>10} {'Median':>10} "
          f"{'P90':>10} {'P99':>10} {'Std':>10}")
    print("-" * 90)

    for task, metrics in sorted(stats.items()):
        for metric, stat in sorted(metrics.items()):
            print(f"{task:<20} {metric:<12} {stat['mean']:>10.4f} "
                  f"{stat['median']:>10.4f} {stat['p90']:>10.4f} "
                  f"{stat['p99']:>10.4f} {stat['std']:>10.4f}")

    if timing_stats:
        print("-" * 90)
        print(f"{'Timing (seconds)':<20} {'total':<12} "
              f"{timing_stats['mean']:>10.1f} {timing_stats['median']:>10.1f} "
              f"{timing_stats['p90']:>10.1f} {timing_stats['p99']:>10.1f} "
              f"{timing_stats['std']:>10.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Unified vLLM evaluation runner for GPU and CPU"
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
        default=5,
        help="Number of evaluation iterations (default: 5)"
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
        help="Output JSON file (default: vllm_accuracy_eval/results/{device}/w4a16_vllm_stats.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run environment verification before evaluation"
    )
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="Skip environment fingerprinting (faster but less traceable)"
    )
    args = parser.parse_args()

    # Set device-specific defaults
    if args.batch_size is None:
        args.batch_size = 8 if args.device == "cpu" else 16

    # Set output paths based on device
    if args.output is None:
        results_dir = RESULTS_CPU_DIR if args.device == "cpu" else RESULTS_GPU_DIR
        args.output = str(results_dir / "w4a16_vllm_stats.json")

    logs_dir = LOGS_CPU_DIR if args.device == "cpu" else LOGS_GPU_DIR

    # Ensure directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Capture environment fingerprint
    fingerprint = None
    model_checksum = None
    if not args.skip_fingerprint:
        print("Capturing environment fingerprint...")
        fingerprint = get_environment_fingerprint()
        model_checksum = get_model_checksum(args.model)

    # Run verification if requested
    if args.verify:
        print("\n" + "=" * 60)
        print("Running pre-flight verification...")
        print("=" * 60)
        verify_script = SCRIPT_DIR / "scripts" / "verify_environment.py"
        if verify_script.exists():
            result = subprocess.run(
                [sys.executable, str(verify_script), "--model", args.model],
                capture_output=False
            )
            if result.returncode != 0:
                print("\nVerification failed! Use --skip-fingerprint to bypass.")
                sys.exit(1)
        else:
            print(f"Warning: verify_environment.py not found at {verify_script}")
        print("")

    print("=" * 60)
    print(f"vLLM Evaluation Runner ({args.device.upper()})")
    print("=" * 60)
    print(f"Device:     {args.device.upper()}")
    print(f"Model:      {args.model}")
    if model_checksum:
        print(f"Model Hash: {model_checksum}")
    print(f"Tasks:      {args.tasks}")
    print(f"Iterations: {args.iterations}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    if fingerprint:
        print(f"PyTorch:    {fingerprint['packages'].get('torch', 'N/A')}")
        print(f"vLLM:       {fingerprint['packages'].get('vllm', 'N/A')}")
        print(f"lm-eval:    {fingerprint['packages'].get('lm_eval', 'N/A')}")
    print("=" * 60)

    # Collect results from all iterations
    all_results = []
    all_timings = []

    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i + 1}/{args.iterations}")
        print("=" * 60)

        results, elapsed = run_single_eval(
            args.model, args.tasks, args.device,
            args.batch_size, seed=args.seed
        )
        if results:
            all_results.append(results)
            all_timings.append(elapsed)
            print(f"Iteration {i + 1} results: {results}")
            print(f"Iteration {i + 1} time: {elapsed:.1f}s")
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
            values = [r[task][metric] for r in all_results
                     if task in r and metric in r[task]]
            stats[task][metric] = calculate_statistics(values)

    timing_stats = calculate_statistics(all_timings)

    # Print results table
    print_results_table(stats, timing_stats)

    # Save to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "backend": "vllm",
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

    print(f"\nResults saved to: {args.output}")

    # Print reproducibility summary
    if fingerprint:
        print("\nReproducibility Info:")
        print(f"  Model checksum: {model_checksum or 'N/A'}")
        print(f"  Environment fingerprint included in output JSON")


if __name__ == "__main__":
    main()
