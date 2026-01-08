#!/bin/bash
# Quantize using auto-round-light (fast recipe: iters=50, lr=5e-3)
# Outputs to logs/ with real-time tee

set -e
# Set HF_TOKEN env var before running, or login with `huggingface-cli login`
: "${HF_TOKEN:?Please set HF_TOKEN environment variable}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./llama3.1_8b_int4}"
SCHEME="${SCHEME:-W4A16}"
FORMAT="${FORMAT:-auto_round,auto_gptq}"
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/quantize_light_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "AutoRound-Light Quantization"
echo "============================================"
echo "Model:      $MODEL"
echo "Output:     $OUTPUT_DIR"
echo "Scheme:     $SCHEME"
echo "Format:     $FORMAT"
echo "Log:        $LOG_FILE"
echo "============================================"

auto-round-light \
    --model "$MODEL" \
    --scheme "$SCHEME" \
    --format "$FORMAT" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "============================================"
echo "Quantization complete!"
echo "Output: $OUTPUT_DIR"
echo "Log:    $LOG_FILE"
echo "============================================"
