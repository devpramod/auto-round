#!/bin/bash
# Snapshot HuggingFace dataset cache for reproducible evaluation
#
# Usage:
#   ./vllm_accuracy_eval/scripts/snapshot_datasets.sh [output_path]
#
# Creates a compressed archive of the evaluation datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT_PATH="${1:-${BASE_DIR}/datasets_snapshot.tar.gz}"
HF_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

# Datasets used in evaluation
DATASETS=(
    "EleutherAI___lambada_openai"
    "Rowan___hellaswag"
    "baber___piqa"
)

echo "============================================"
echo "Dataset Snapshot Tool"
echo "============================================"
echo "HF Cache:   $HF_CACHE"
echo "Output:     $OUTPUT_PATH"
echo "============================================"

# Check if cache exists
if [ ! -d "$HF_CACHE" ]; then
    echo "Error: HuggingFace cache not found at $HF_CACHE"
    echo ""
    echo "Run an evaluation first to download datasets:"
    echo "  auto-round --eval --model ... --tasks lambada_openai,hellaswag,piqa"
    exit 1
fi

# Check which datasets are cached
echo ""
echo "Checking cached datasets..."
MISSING=()
FOUND=()

for ds in "${DATASETS[@]}"; do
    if [ -d "${HF_CACHE}/${ds}" ]; then
        SIZE=$(du -sh "${HF_CACHE}/${ds}" 2>/dev/null | cut -f1)
        echo "  [OK] $ds ($SIZE)"
        FOUND+=("$ds")
    else
        echo "  [MISSING] $ds"
        MISSING+=("$ds")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "Warning: ${#MISSING[@]} dataset(s) not cached"
    echo "Run evaluation first to download missing datasets"
    read -p "Continue with partial snapshot? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ ${#FOUND[@]} -eq 0 ]; then
    echo "Error: No datasets to snapshot"
    exit 1
fi

# Create snapshot
echo ""
echo "Creating snapshot..."

# Build list of directories to archive
DIRS_TO_ARCHIVE=()
for ds in "${FOUND[@]}"; do
    DIRS_TO_ARCHIVE+=("${ds}")
done

# Create archive with relative paths
cd "$HF_CACHE"

# Use pigz for parallel compression if available
if command -v pigz &> /dev/null; then
    echo "Using pigz for parallel compression..."
    tar -cf - "${DIRS_TO_ARCHIVE[@]}" | pigz -9 > "$OUTPUT_PATH"
else
    echo "Using gzip (install pigz for faster compression)..."
    tar -czf "$OUTPUT_PATH" "${DIRS_TO_ARCHIVE[@]}"
fi

cd - > /dev/null

# Generate manifest
MANIFEST_PATH="${OUTPUT_PATH%.tar.gz}.manifest.txt"
echo "Generating manifest..."
{
    echo "# Dataset Snapshot Manifest"
    echo "# Generated: $(date -Iseconds)"
    echo ""
    echo "## Archive Info"
    echo "Archive: $(basename "$OUTPUT_PATH")"
    echo "Size: $(du -h "$OUTPUT_PATH" | cut -f1)"
    echo "SHA256: $(sha256sum "$OUTPUT_PATH" | cut -d' ' -f1)"
    echo ""
    echo "## Datasets"
    for ds in "${FOUND[@]}"; do
        echo "- $ds"
    done
    echo ""
    echo "## File List"
    tar -tzf "$OUTPUT_PATH" | head -100
    echo "... (truncated)"
} > "$MANIFEST_PATH"

# Summary
ARCHIVE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
echo ""
echo "============================================"
echo "Snapshot created successfully!"
echo "============================================"
echo "Archive:  $OUTPUT_PATH ($ARCHIVE_SIZE)"
echo "Manifest: $MANIFEST_PATH"
echo ""
echo "To restore on another machine:"
echo "  mkdir -p ~/.cache/huggingface/datasets"
echo "  tar -xzf $(basename "$OUTPUT_PATH") -C ~/.cache/huggingface/datasets"
echo "============================================"
