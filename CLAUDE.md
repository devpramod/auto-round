# AutoRound Quantization Project

## Goal
Quantize Llama 3.1 8B to W4A16, compare eval accuracy between GPU (NVIDIA A6000) and CPU (Intel Xeon via vLLM).

## Quick Start

### Environment
```bash
cd /root/auto-round
source .venv/bin/activate
```

### Quantize (auto-round-light)
```bash
# Background with live logging
nohup ./scripts/quantize_light.sh 2>&1 &

# Or direct command
auto-round-light \
    --model meta-llama/Llama-3.1-8B \
    --scheme W4A16 \
    --format "auto_round,auto_gptq" \
    --output_dir ./llama3.1_8b_int4
```

### Evaluate
```bash
# Single eval
auto-round --eval \
    --model ./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq \
    --tasks lambada_openai,hellaswag,piqa \
    --eval_bs 16

# Multi-iteration with statistics
python scripts/eval_runner.py \
    --model ./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq \
    --tasks lambada_openai,hellaswag,piqa \
    --iterations 5 \
    --output eval_results_gpu.json
```

## Log Registry

All logs are tracked in `logs/REGISTRY.md` with:
- Date and time of run
- Command used
- Status/result

```bash
# View log registry
cat logs/REGISTRY.md

# List logs by date
ls -lt logs/*.log
```

## Live Log Monitoring
```bash
# Watch quantization
tail -f logs/quantize_light_*.log

# Watch baseline eval
tail -f logs/baseline_eval_*.log

# Check running processes
ps aux | grep -E "(auto-round|python)"
```

## Scripts
| Script | Description |
|--------|-------------|
| `scripts/quantize_light.sh` | Quantize with auto-round-light (iters=50) |
| `scripts/quantize.sh` | Quantize with auto-round (iters=200) |
| `scripts/run_baseline_eval.sh` | Baseline BF16 evaluation |
| `scripts/eval_runner.py` | Multi-iteration eval with statistics |

## Output Directory
```
llama3.1_8b_int4/
└── Llama-3.1-8B-w4g128/
    ├── auto-round/           # Intel CPU/GPU optimized
    └── auto-round-auto-gptq/ # vLLM CPU compatible
```

---

## TODO: Current Exercise

### Phase 1: GPU Quantization and Evaluation
- [x] Setup environment (venv, auto-round, lm-eval)
- [ ] Run baseline BF16 eval (3 tasks)
- [ ] Quantize with auto-round-light
- [ ] Run 5-iteration GPU eval with statistics
- [ ] Document results (mean, median, P90, P99)

### Phase 2: vLLM CPU Inference
- [ ] Load quantized model with vLLM on Intel Xeon
- [ ] Run same 3 eval tasks on CPU
- [ ] Compare accuracy: GPU vs CPU

### Phase 3: Analysis
- [ ] Document accuracy delta (BF16 vs W4A16)
- [ ] Verify consistency across iterations
- [ ] Compare GPU vs CPU eval results

## Reference
- [AutoRound Paper](https://arxiv.org/pdf/2309.05516)
- [User Guide](./docs/step_by_step.md)
- [vLLM Integration](https://docs.vllm.ai/en/latest/features/quantization/auto_round/)
