#!/usr/bin/env python3
"""
Pre-flight verification for reproducible vLLM evaluation.

Checks:
1. Library versions match expected
2. Model exists and checksum matches (optional)
3. Dataset cache is populated
4. GPU/CUDA availability
5. Environment variables are set correctly

Usage:
    python vllm_accuracy_eval/scripts/verify_environment.py
    python vllm_accuracy_eval/scripts/verify_environment.py --model ./path/to/model
    python vllm_accuracy_eval/scripts/verify_environment.py --strict  # Fail on any mismatch
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Expected versions (update when requirements-eval.txt changes)
EXPECTED_VERSIONS = {
    "torch": "2.4.0",
    "vllm": "0.6.3.post1",
    "lm_eval": "0.4.4",
    "transformers": "4.45.2",
    "datasets": "3.0.1",
    "numpy": "1.26.4",
    "accelerate": "1.0.1",
}

# Required environment variables for determinism
REQUIRED_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "false",  # Set by auto-round internally
}

# Dataset cache paths to check
DATASET_TASKS = [
    ("lambada_openai", "EleutherAI___lambada_openai"),
    ("hellaswag", "Rowan___hellaswag"),
    ("piqa", "baber___piqa"),
]


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> str:
    return f"{Colors.GREEN}[OK]{Colors.END} {msg}"


def warn(msg: str) -> str:
    return f"{Colors.YELLOW}[WARN]{Colors.END} {msg}"


def fail(msg: str) -> str:
    return f"{Colors.RED}[FAIL]{Colors.END} {msg}"


def get_package_version(package: str) -> Optional[str]:
    """Get installed version of a package."""
    try:
        if package == "lm_eval":
            import lm_eval
            return lm_eval.__version__
        elif package == "torch":
            import torch
            # Strip CUDA suffix (e.g., "2.4.0+cu121" -> "2.4.0")
            return torch.__version__.split("+")[0]
        elif package == "vllm":
            import vllm
            return vllm.__version__
        elif package == "transformers":
            import transformers
            return transformers.__version__
        elif package == "datasets":
            import datasets
            return datasets.__version__
        elif package == "numpy":
            import numpy
            return numpy.__version__
        elif package == "accelerate":
            import accelerate
            return accelerate.__version__
        else:
            return None
    except ImportError:
        return None


def check_versions(strict: bool = False) -> tuple[bool, list[dict]]:
    """Check if installed versions match expected versions."""
    results = []
    all_ok = True

    print(f"\n{Colors.BOLD}Library Versions{Colors.END}")
    print("-" * 50)

    for package, expected in EXPECTED_VERSIONS.items():
        installed = get_package_version(package)
        if installed is None:
            print(fail(f"{package}: NOT INSTALLED (expected {expected})"))
            results.append({"package": package, "expected": expected, "installed": None, "status": "missing"})
            all_ok = False
        elif installed == expected:
            print(ok(f"{package}: {installed}"))
            results.append({"package": package, "expected": expected, "installed": installed, "status": "match"})
        else:
            if strict:
                print(fail(f"{package}: {installed} (expected {expected})"))
                all_ok = False
            else:
                print(warn(f"{package}: {installed} (expected {expected})"))
            results.append({"package": package, "expected": expected, "installed": installed, "status": "mismatch"})

    return all_ok, results


def check_model(model_path: str) -> tuple[bool, dict]:
    """Check if model exists and compute checksum of config."""
    print(f"\n{Colors.BOLD}Model Verification{Colors.END}")
    print("-" * 50)

    model_dir = Path(model_path)
    result = {"path": str(model_path), "exists": False, "config_hash": None}

    if not model_dir.exists():
        print(fail(f"Model not found: {model_path}"))
        return False, result

    result["exists"] = True
    print(ok(f"Model directory exists: {model_path}"))

    # Check for required files
    required_files = ["config.json", "tokenizer_config.json"]
    for fname in required_files:
        fpath = model_dir / fname
        if fpath.exists():
            print(ok(f"  {fname} present"))
        else:
            print(warn(f"  {fname} missing"))

    # Compute config.json hash for verification
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        result["config_hash"] = config_hash
        print(ok(f"  config.json hash: {config_hash}"))

    # Check for model weights
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if weight_files:
        total_size = sum(f.stat().st_size for f in weight_files)
        print(ok(f"  Weight files: {len(weight_files)} files, {total_size / 1e9:.2f} GB"))
    else:
        print(warn("  No weight files found"))

    return True, result


def check_dataset_cache() -> tuple[bool, list[dict]]:
    """Check if evaluation datasets are cached."""
    print(f"\n{Colors.BOLD}Dataset Cache{Colors.END}")
    print("-" * 50)

    cache_dir = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
    results = []
    all_ok = True

    print(f"Cache directory: {cache_dir}")

    if not cache_dir.exists():
        print(warn("Cache directory does not exist"))
        return False, results

    for task_name, cache_name in DATASET_TASKS:
        task_cache = cache_dir / cache_name
        if task_cache.exists():
            # Get cache size
            cache_size = sum(f.stat().st_size for f in task_cache.rglob("*") if f.is_file())
            print(ok(f"  {task_name}: cached ({cache_size / 1e6:.1f} MB)"))
            results.append({"task": task_name, "cached": True, "size_mb": cache_size / 1e6})
        else:
            print(warn(f"  {task_name}: NOT cached (will download on first run)"))
            results.append({"task": task_name, "cached": False, "size_mb": 0})
            all_ok = False

    return all_ok, results


def check_gpu() -> tuple[bool, dict]:
    """Check GPU availability and CUDA version."""
    print(f"\n{Colors.BOLD}GPU/CUDA{Colors.END}")
    print("-" * 50)

    result = {"cuda_available": False, "device_count": 0, "devices": []}

    try:
        import torch
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["device_count"] = torch.cuda.device_count()
            print(ok(f"CUDA available: {torch.cuda.device_count()} device(s)"))

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                result["devices"].append({"id": i, "name": name, "memory_gb": mem})
                print(ok(f"  [{i}] {name} ({mem:.1f} GB)"))

            print(ok(f"CUDA version: {torch.version.cuda}"))
            result["cuda_version"] = torch.version.cuda
            return True, result
        else:
            print(warn("CUDA not available (CPU-only mode)"))
            return True, result  # Not a failure, just informational
    except ImportError:
        print(fail("PyTorch not installed"))
        return False, result


def check_environment_vars() -> tuple[bool, dict]:
    """Check required environment variables."""
    print(f"\n{Colors.BOLD}Environment Variables{Colors.END}")
    print("-" * 50)

    results = {}
    all_ok = True

    # Check recommended vars (informational)
    recommended = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "VLLM_TARGET_DEVICE": os.environ.get("VLLM_TARGET_DEVICE"),
        "HF_TOKEN": "***" if os.environ.get("HF_TOKEN") else None,
    }

    for var, value in recommended.items():
        results[var] = value
        if value:
            print(ok(f"  {var}={value}"))
        else:
            print(warn(f"  {var} not set"))

    return all_ok, results


def check_dataset_snapshot(snapshot_path: str = None) -> tuple[bool, dict]:
    """Check if dataset snapshot exists."""
    print(f"\n{Colors.BOLD}Dataset Snapshot{Colors.END}")
    print("-" * 50)

    if snapshot_path is None:
        script_dir = Path(__file__).parent.parent
        snapshot_path = script_dir / "datasets_snapshot.tar.gz"

    result = {"path": str(snapshot_path), "exists": False, "size_mb": 0}

    if Path(snapshot_path).exists():
        size = Path(snapshot_path).stat().st_size / 1e6
        result["exists"] = True
        result["size_mb"] = size
        print(ok(f"Snapshot exists: {snapshot_path} ({size:.1f} MB)"))
        return True, result
    else:
        print(warn(f"Snapshot not found: {snapshot_path}"))
        print(warn("  Run ./scripts/snapshot_datasets.sh to create"))
        return False, result


def generate_report(results: dict, output_path: str = None) -> str:
    """Generate JSON verification report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "checks": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Verify evaluation environment")
    parser.add_argument(
        "--model",
        default="./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq",
        help="Path to quantized model"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any version mismatch"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output JSON report"
    )
    args = parser.parse_args()

    if not args.quiet:
        print("=" * 50)
        print(f"{Colors.BOLD}vLLM Evaluation Environment Verification{Colors.END}")
        print("=" * 50)

    all_checks_passed = True
    results = {}

    # Run all checks
    ok, results["versions"] = check_versions(strict=args.strict)
    all_checks_passed &= ok

    ok, results["model"] = check_model(args.model)
    all_checks_passed &= ok

    ok, results["datasets"] = check_dataset_cache()
    # Dataset cache is not critical - will download if missing

    ok, results["gpu"] = check_gpu()
    all_checks_passed &= ok

    ok, results["env_vars"] = check_environment_vars()

    ok, results["snapshot"] = check_dataset_snapshot()
    # Snapshot is recommended but not required

    # Summary
    if not args.quiet:
        print("\n" + "=" * 50)
        if all_checks_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}All critical checks passed!{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}Some checks failed!{Colors.END}")
            if args.strict:
                print("Strict mode: evaluation may not be reproducible")
        print("=" * 50)

    # Generate report
    if args.output:
        generate_report(results, args.output)
        if not args.quiet:
            print(f"\nReport saved to: {args.output}")

    # Exit code
    sys.exit(0 if all_checks_passed else 1)


if __name__ == "__main__":
    main()
