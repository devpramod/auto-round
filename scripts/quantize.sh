#!/bin/bash
# Quantize Llama 3.1 8B to W4A16 (4-bit weights, 16-bit activations)
# Output formats: auto_round and auto_gptq (for vLLM CPU compatibility)

set -e

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./llama3.1_8b_int4}"
SCHEME="${SCHEME:-W4A16}"
FORMAT="${FORMAT:-auto_round,auto_gptq}"

echo "============================================"
echo "AutoRound Quantization"
echo "============================================"
echo "Model:      $MODEL"
echo "Output:     $OUTPUT_DIR"
echo "Scheme:     $SCHEME"
echo "Format:     $FORMAT"
echo "============================================"

auto-round \
    --model "$MODEL" \
    --scheme "$SCHEME" \
    --format "$FORMAT" \
    --output_dir "$OUTPUT_DIR"

echo "============================================"
echo "Quantization complete!"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"
ls -la "$OUTPUT_DIR"
