# Llama 3.1 8B 4-bit Quantization Guide

This guide documents the steps to quantize Llama 3.1 8B to 4-bit and evaluate accuracy.

## Prerequisites

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install auto-round
pip install auto-round
```

## Step 1: Baseline Evaluation (Optional)

Measure the original BF16 model accuracy before quantization:

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks lambada_openai \
    --batch_size 16
```

**Baseline Results (Llama 3.1 8B BF16):**
| Metric     | Value  | Stderr |
|------------|--------|--------|
| Accuracy   | 0.7467 | 0.0061 |
| Perplexity | 3.1458 | 0.0581 |

## Step 2: Quantize the Model

Use `auto-round-light` for faster quantization (~3 hours for 8B model):

```bash
auto-round-light \
    --model meta-llama/Llama-3.1-8B \
    --scheme "W4A16" \
    --format "auto_round,auto_gptq" \
    --output_dir ./llama3.1_8b_int4
```

**Recipe Options:**
- `auto-round-light`: Fastest (2-3x faster), good accuracy
- `auto-round`: Default, balanced speed/accuracy
- `auto-round-best`: Highest accuracy, slowest

**Output Formats:**
- `auto_round`: Best for Intel CPU/GPU inference
- `auto_gptq`: Compatible with GPTQ ecosystem
- `auto_awq`: Compatible with AWQ ecosystem

## Step 3: Evaluate Quantized Model

```bash
auto-round \
    --model ./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq \
    --eval \
    --tasks lambada_openai \
    --eval_bs 16
```

**Quantized Results (W4A16):**
| Metric     | Value  | Stderr |
|------------|--------|--------|
| Accuracy   | 0.7254 | 0.0062 |
| Perplexity | 3.4022 | 0.0653 |

## Results Summary

| Model | Accuracy | Perplexity | Size Reduction |
|-------|----------|------------|----------------|
| BF16 Baseline | 0.7467 | 3.1458 | 1x |
| W4A16 Quantized | 0.7254 | 3.4022 | ~4x |

**Accuracy Loss:** ~2.1 percentage points (97.1% of original accuracy retained)

## Inference

### CPU Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### GPU Inference (CUDA)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

## Tips

1. **Memory**: 4-bit quantization reduces VRAM from ~16GB to ~4GB for 8B models
2. **Speed**: Quantized models may be slightly slower on GPU but faster on CPU
3. **Quality**: For best accuracy, use `auto-round-best` recipe (slower)
4. **Format**: Use `auto_gptq` format for widest compatibility

