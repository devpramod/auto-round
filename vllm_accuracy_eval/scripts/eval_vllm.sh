#!/bin/bash
# vLLM Evaluation Shell Wrapper
# Usage: ./eval_vllm.sh gpu|cpu [iterations] [model_path]

set -e

DEVICE="${1:-gpu}"
ITERATIONS="${2:-5}"
MODEL="${3:-./llama3.1_8b_int4/Llama-3.1-8B-w4g128/auto-round-auto-gptq}"

# Validate device
if [[ "$DEVICE" != "gpu" && "$DEVICE" != "cpu" ]]; then
    echo "Error: Device must be 'gpu' or 'cpu'"
    echo "Usage: $0 gpu|cpu [iterations] [model_path]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Set up logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_DIR}/logs/${DEVICE}"
LOG_FILE="${LOG_DIR}/eval_vllm_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Output file
OUTPUT_FILE="${BASE_DIR}/results/${DEVICE}/w4a16_vllm_stats.json"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "============================================" | tee "$LOG_FILE"
echo "vLLM Evaluation (${DEVICE^^})" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Device:     $DEVICE" | tee -a "$LOG_FILE"
echo "Model:      $MODEL" | tee -a "$LOG_FILE"
echo "Iterations: $ITERATIONS" | tee -a "$LOG_FILE"
echo "Output:     $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the Python eval runner
python "${SCRIPT_DIR}/eval_runner_vllm.py" \
    --device "$DEVICE" \
    --model "$MODEL" \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_FILE" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Evaluation complete!" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "Log:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
