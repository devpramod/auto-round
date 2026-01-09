# vLLM Accuracy Evaluation

Fair GPU vs CPU accuracy comparison using vLLM backend for both evaluations.

## Overview

This directory contains infrastructure for running identical vLLM-based evaluations on GPU and CPU, ensuring a fair accuracy comparison with full reproducibility.

## Directory Structure

```
vllm_accuracy_eval/
├── README.md                     # This file
├── requirements-eval.txt         # Locked dependencies
├── scripts/
│   ├── setup_env.sh              # Create reproducible environment
│   ├── verify_environment.py     # Pre-flight verification
│   ├── snapshot_datasets.sh      # Archive evaluation datasets
│   ├── eval_vllm.sh              # Shell wrapper (device as argument)
│   ├── eval_runner_vllm.py       # Unified multi-iteration eval
│   └── compare_results.py        # GPU vs CPU comparison tool
├── results/
│   ├── gpu/
│   │   └── w4a16_vllm_stats.json # GPU vLLM eval results
│   ├── cpu/
│   │   └── w4a16_vllm_stats.json # CPU vLLM eval results
│   └── comparison_report.md      # Generated comparison report
└── logs/
    ├── REGISTRY.md               # Log registry
    ├── gpu/                      # GPU eval logs
    └── cpu/                      # CPU eval logs
```

## Reproducibility

### Why It Matters

For a fair GPU vs CPU comparison, we need to ensure:
- Same library versions (vLLM, lm-eval, transformers)
- Same dataset samples (cached HuggingFace datasets)
- Same random seed (deterministic evaluation)
- Same model weights (verified by checksum)

### Setup Reproducible Environment

```bash
# 1. Create isolated virtual environment with locked deps
./vllm_accuracy_eval/scripts/setup_env.sh venv-eval

# 2. Activate the environment
source venv-eval/bin/activate

# 3. Snapshot datasets (first time only, ~400MB)
./vllm_accuracy_eval/scripts/snapshot_datasets.sh

# 4. Verify environment before evaluation
python vllm_accuracy_eval/scripts/verify_environment.py
```

### What Gets Captured

Each evaluation run automatically records:
- **Environment fingerprint**: Python version, library versions, CUDA info
- **Model checksum**: SHA256 of config.json
- **Seed**: Random seed used (default: 42)
- **Timing**: Per-iteration and aggregate timing stats

This info is saved in the output JSON for later verification.

## Quick Start

### Prerequisites

1. Quantized model at: `./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq/`
2. vLLM installed
3. auto-round CLI available

### Run GPU Evaluation

```bash
# 5 iterations with verification
./vllm_accuracy_eval/scripts/eval_vllm.sh gpu 5

# Or with explicit verification
python vllm_accuracy_eval/scripts/eval_runner_vllm.py \
    --device gpu \
    --iterations 5 \
    --verify
```

### Run CPU Evaluation

```bash
# 5 iterations (recommended)
./vllm_accuracy_eval/scripts/eval_vllm.sh cpu 5
```

### Generate Comparison Report

After both GPU and CPU evaluations complete:

```bash
python vllm_accuracy_eval/scripts/compare_results.py
```

The report will be saved to `vllm_accuracy_eval/results/comparison_report.md`.

## Configuration

### Default Settings

| Setting | GPU | CPU |
|---------|-----|-----|
| Batch Size | 16 | 8 |
| Seed | 42 | 42 |
| Tasks | lambada_openai, hellaswag, piqa | (same) |
| Backend | vLLM | vLLM |

### Reproducibility Flags

```bash
python vllm_accuracy_eval/scripts/eval_runner_vllm.py \
    --device gpu \
    --verify                 # Run pre-flight checks
    --skip-fingerprint       # Skip environment capture (faster)
```

### Custom Settings

```bash
python vllm_accuracy_eval/scripts/eval_runner_vllm.py \
    --device gpu \
    --model /path/to/model \
    --tasks "lambada_openai,hellaswag,piqa,winogrande" \
    --iterations 10 \
    --batch-size 32 \
    --seed 123 \
    --output custom_results.json
```

## Understanding Results

### Output JSON Structure

```json
{
  "timestamp": "2026-01-09T...",
  "device": "gpu",
  "backend": "vllm",
  "config": {
    "model": "./llama3.1_8b_int4/...",
    "model_checksum": "abc123...",
    "tasks": "lambada_openai,hellaswag,piqa",
    "iterations": 5,
    "batch_size": 16,
    "seed": 42
  },
  "statistics": {
    "lambada_openai": {
      "acc": {"mean": 0.7240, "std": 0.0, ...}
    }
  },
  "environment": {
    "packages": {"torch": "2.4.0", "vllm": "0.6.3.post1", ...},
    "cuda": {"available": true, "version": "12.1", ...}
  }
}
```

### Comparison Report

The comparison tool flags differences > 0.5 percentage points as significant:

| Status | Meaning |
|--------|---------|
| OK | Delta < 0.5 pp, results consistent |
| **DIFF** | Delta >= 0.5 pp, investigate |

### Expected Results

With identical seed (42), GPU and CPU vLLM evaluations should produce:
- Identical or near-identical accuracy (within floating-point tolerance)
- CPU evaluations will be slower than GPU

Any significant accuracy differences may indicate:
- Numerical precision differences (expected small delta)
- vLLM version differences
- Model loading issues

## Transferring to Another Machine

### Export (Source Machine)

```bash
# 1. Snapshot datasets
./vllm_accuracy_eval/scripts/snapshot_datasets.sh

# 2. Copy these files:
#    - vllm_accuracy_eval/datasets_snapshot.tar.gz
#    - vllm_accuracy_eval/requirements-eval.txt
#    - The quantized model directory
```

### Import (Target Machine)

```bash
# 1. Restore datasets
mkdir -p ~/.cache/huggingface/datasets
tar -xzf datasets_snapshot.tar.gz -C ~/.cache/huggingface/datasets

# 2. Create environment
./vllm_accuracy_eval/scripts/setup_env.sh venv-eval

# 3. Verify
source venv-eval/bin/activate
python vllm_accuracy_eval/scripts/verify_environment.py --strict
```

## Troubleshooting

### Version Mismatch

```bash
# Check current vs expected versions
python vllm_accuracy_eval/scripts/verify_environment.py

# Reinstall locked versions
pip install -r vllm_accuracy_eval/requirements-eval.txt
```

### CPU Evaluation Fails

```bash
# Ensure vLLM CPU support
export VLLM_TARGET_DEVICE=cpu
export OMP_NUM_THREADS=$(nproc)
```

### Out of Memory

```bash
# Reduce batch size
python vllm_accuracy_eval/scripts/eval_runner_vllm.py --device cpu --batch-size 4
```

### Model Not Found

```bash
# Check model path exists
ls -la ./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq/
```

## Logs

All logs are saved with timestamps:

```bash
# View GPU logs
ls -lt vllm_accuracy_eval/logs/gpu/

# View CPU logs
ls -lt vllm_accuracy_eval/logs/cpu/

# Watch live log
tail -f vllm_accuracy_eval/logs/gpu/eval_vllm_*.log
```

## Factors That Cannot Be Controlled

Even with full reproducibility measures, these may cause small differences:

1. **Floating-point precision**: GPU and CPU have different FP implementations
2. **Kernel implementations**: vLLM may use different code paths
3. **Hardware variations**: Different GPU models may have tiny variations

Expected impact: < 0.1 percentage points. If you see > 0.5 pp difference, investigate library versions first.
