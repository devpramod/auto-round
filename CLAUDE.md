# AutoRound Quantization Project

## Goal
Quantize Llama 3.1 8B to W4A16, compare eval accuracy between GPU (NVIDIA A6000) and CPU (Intel Xeon via vLLM).

## Quick Start

### Environment
```bash
cd /root/auto-round
source .venv/bin/activate          # Main auto-round env
# OR
source venv-eval/bin/activate       # vLLM eval env (locked deps)
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

### Evaluate (vLLM Backend - Recommended)
```bash
# GPU evaluation (5 iterations)
./vllm_accuracy_eval/scripts/eval_vllm.sh gpu 5

# CPU evaluation (5 iterations)
./vllm_accuracy_eval/scripts/eval_vllm.sh cpu 5

# Compare GPU vs CPU
python vllm_accuracy_eval/scripts/compare_results.py
```

### Evaluate (Legacy HF Backend)
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

### vLLM Evaluation Scripts (Reproducible)
| Script | Description |
|--------|-------------|
| `vllm_accuracy_eval/scripts/eval_vllm.sh` | Shell wrapper for GPU/CPU eval |
| `vllm_accuracy_eval/scripts/eval_runner_vllm.py` | Unified vLLM multi-iteration eval |
| `vllm_accuracy_eval/scripts/compare_results.py` | GPU vs CPU comparison report |
| `vllm_accuracy_eval/scripts/setup_env.sh` | Create reproducible venv |
| `vllm_accuracy_eval/scripts/verify_environment.py` | Pre-flight environment check |
| `vllm_accuracy_eval/scripts/snapshot_datasets.sh` | Archive eval datasets |

See `vllm_accuracy_eval/README.md` for full documentation.

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
- [x] Quantize with auto-round-light (W4A16, GPTQ format)
- [x] Create vLLM evaluation infrastructure (`vllm_accuracy_eval/`)
- [x] Setup reproducible eval environment (venv-eval, locked deps)
- [ ] Run 5-iteration GPU vLLM eval with statistics (in progress)
- [ ] Document results (mean, median, P90, P99)

### Phase 2: vLLM CPU Inference
- [ ] Transfer to Intel Xeon machine
- [ ] Run 5-iteration CPU vLLM eval
- [ ] Generate comparison report

### Phase 3: Analysis
- [ ] Document accuracy delta (BF16 vs W4A16)
- [ ] Verify consistency across iterations
- [ ] Compare GPU vs CPU eval results

## Environment Info (venv-eval)

Locked versions for reproducibility:
- vLLM: 0.13.0
- lm-eval: 0.4.9.2
- PyTorch: 2.9.0
- transformers: 4.57.3
- accelerate: 1.12.0

See `vllm_accuracy_eval/requirements-eval.txt` for complete list.

## Reference
- [AutoRound Paper](https://arxiv.org/pdf/2309.05516)
- [User Guide](./docs/step_by_step.md)
- [vLLM Integration](https://docs.vllm.ai/en/latest/features/quantization/auto_round/)
