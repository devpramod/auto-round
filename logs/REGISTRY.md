# Log Registry

All experiment logs with timestamps and commands.

## Active Runs

| Log File | Date | Time | Command | Status |
|----------|------|------|---------|--------|
| (none) | - | - | - | - |

## Completed Runs

| Log File | Date | Time | Command | Duration | Result |
|----------|------|------|---------|----------|--------|
| `quantize_light_20260107_155706.log` | 2026-01-07 | 15:57:06 | `auto-round-light --model meta-llama/Llama-3.1-8B --scheme W4A16 --format "auto_round,auto_gptq" --output_dir ./llama3.1_8b_int4` | 585.6s | Success (224/225 layers) |

## Evaluation Runs

| Log File | Date | Time | Model | Tasks | Iterations | Output |
|----------|------|------|-------|-------|------------|--------|
| `baseline_eval_20260107_162629.log` | 2026-01-07 | 16:26:29 | meta-llama/Llama-3.1-8B (BF16) | lambada,hellaswag,piqa | 5 | Deterministic |
| `w4a16_eval_20260107_174206.log` | 2026-01-07 | 17:42:06 | Llama-3.1-8B-w4g128 (W4A16) | lambada,hellaswag,piqa | 5 | Deterministic |
| `w4a16_eval_seed42_20260107_214423.log` | 2026-01-07 | 21:44:23 | Llama-3.1-8B-w4g128 (W4A16) | lambada,hellaswag,piqa | 5 | Deterministic (seed=42, Std=0) |

---

## Log File Naming Convention

- `quantize_light_YYYYMMDD_HHMMSS.log` - auto-round-light quantization
- `quantize_YYYYMMDD_HHMMSS.log` - auto-round (default) quantization
- `baseline_eval_YYYYMMDD_HHMMSS.log` - BF16 baseline evaluation
- `eval_gpu_YYYYMMDD_HHMMSS.log` - GPU quantized model evaluation
- `eval_cpu_YYYYMMDD_HHMMSS.log` - CPU (vLLM) evaluation

## Quick Commands

```bash
# View registry
cat logs/REGISTRY.md

# List all logs by date
ls -lt logs/*.log

# Check running processes
ps aux | grep -E "(auto-round|eval)"
```
