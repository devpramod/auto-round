#!/bin/bash
# Run baseline evaluation on BF16 model (before quantization)
# This provides a reference for accuracy comparison

set -e

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
TASKS="${TASKS:-lambada_openai,hellaswag,piqa}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "============================================"
echo "Baseline BF16 Evaluation"
echo "============================================"
echo "Model:      $MODEL"
echo "Tasks:      $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

auto-round \
    --model "$MODEL" \
    --eval \
    --tasks "$TASKS" \
    --eval_bs "$BATCH_SIZE"

echo "============================================"
echo "Baseline evaluation complete!"
echo "============================================"
