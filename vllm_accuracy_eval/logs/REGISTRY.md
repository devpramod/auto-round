# vLLM Accuracy Evaluation Log Registry

All vLLM evaluation logs for GPU and CPU with timestamps and commands.

## Active Runs

| Device | Log File | Start Time | Command | Status |
|--------|----------|------------|---------|--------|
| (none) | - | - | - | - |

## Completed Runs

### GPU Evaluations

| Log File | Date | Time | Model | Tasks | Iterations | Result |
|----------|------|------|-------|-------|------------|--------|
| (none yet) | - | - | - | - | - | - |

### CPU Evaluations

| Log File | Date | Time | Model | Tasks | Iterations | Result |
|----------|------|------|-------|-------|------------|--------|
| (none yet) | - | - | - | - | - | - |

---

## Log File Naming Convention

- `logs/gpu/eval_vllm_YYYYMMDD_HHMMSS.log` - GPU vLLM evaluation
- `logs/cpu/eval_vllm_YYYYMMDD_HHMMSS.log` - CPU vLLM evaluation

## Quick Commands

```bash
# View registry
cat vllm_accuracy_eval/logs/REGISTRY.md

# List GPU logs
ls -lt vllm_accuracy_eval/logs/gpu/*.log

# List CPU logs
ls -lt vllm_accuracy_eval/logs/cpu/*.log

# Run GPU evaluation
./vllm_accuracy_eval/scripts/eval_vllm.sh gpu 5

# Run CPU evaluation
./vllm_accuracy_eval/scripts/eval_vllm.sh cpu 5

# Generate comparison report
python vllm_accuracy_eval/scripts/compare_results.py
```
