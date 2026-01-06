# AutoRound Quantization Guide

## Overview
AutoRound is Intel's advanced quantization toolkit for LLMs. This guide focuses on quantizing Llama 3.1 8B to W4A16 (4-bit weights, 16-bit activations) and evaluating accuracy.

## Quick Start

### Environment Setup
```bash
cd /root/auto-round
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install lm-eval>=0.4.2
```

### Quantize Model
```bash
./scripts/quantize.sh
# Or manually:
auto-round \
    --model meta-llama/Llama-3.1-8B \
    --scheme W4A16 \
    --format "auto_round,auto_gptq" \
    --output_dir ./llama3.1_8b_int4
```

### Evaluate Model
```bash
# Single evaluation
auto-round --model ./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq \
    --eval \
    --tasks lambada_openai,hellaswag,piqa \
    --eval_bs 16

# Multi-iteration with statistics
python scripts/eval_runner.py --iterations 10
```

## Key Commands

| Command | Description |
|---------|-------------|
| `auto-round` | Default recipe (iters=200) |
| `auto-round-light` | Fast recipe (iters=50, 2-3x faster) |
| `auto-round-best` | Best accuracy (iters=1000, slower) |
| `auto-round --eval` | Evaluation only mode |

## Evaluation Tasks
- `lambada_openai` - Language modeling accuracy
- `hellaswag` - Commonsense reasoning
- `piqa` - Physical intuition QA

## Output Formats
- `auto_round` - Best for Intel CPU/GPU inference
- `auto_gptq` - Compatible with vLLM CPU inference
- `auto_awq` - Compatible with AWQ ecosystem

## Directory Structure
```
./llama3.1_8b_int4/
└── Llama-3.1-8B-w4g128/
    ├── auto-round/           # auto_round format
    └── auto-round-auto-gptq/ # vLLM compatible
```

## Scripts
- `scripts/quantize.sh` - Quantize Llama 3.1 8B
- `scripts/run_baseline_eval.sh` - Baseline BF16 evaluation
- `scripts/eval_runner.py` - Multi-iteration eval with statistics

---

## TODO: Quantization Exercise

### Phase 1: GPU Quantization and Evaluation
- [ ] Setup environment and install dependencies
- [ ] Run baseline BF16 evaluation (optional reference)
- [ ] Quantize Llama 3.1 8B to W4A16
- [ ] Evaluate quantized model on GPU
- [ ] Run 5-10 iterations and collect statistics (mean, median, P90, P99)

### Phase 2: vLLM CPU Inference (Future)
- [ ] Load quantized model with vLLM on CPU
- [ ] Run same eval tasks on Intel Xeon
- [ ] Compare accuracy: GPU vs CPU inference

### Phase 3: Analysis
- [ ] Document accuracy delta (BF16 vs W4A16)
- [ ] Verify consistency across iterations
- [ ] Compare GPU vs CPU eval results

## Reference
- [AutoRound Paper](https://arxiv.org/pdf/2309.05516)
- [User Guide](./docs/step_by_step.md)
- [vLLM Integration](https://docs.vllm.ai/en/latest/features/quantization/auto_round/)
