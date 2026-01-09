#!/bin/bash
# Setup reproducible evaluation environment
#
# Usage:
#   ./vllm_accuracy_eval/scripts/setup_env.sh [venv_name]
#
# Creates a fresh virtual environment with locked dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BASE_DIR")"

VENV_NAME="${1:-venv-eval}"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"

echo "============================================"
echo "vLLM Evaluation Environment Setup"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Venv path:    $VENV_PATH"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ "3.10" ]] && [[ ! "$PYTHON_VERSION" =~ "3.11" ]]; then
    echo "Warning: Recommended Python version is 3.10 or 3.11"
fi

# Create virtual environment
if [ -d "$VENV_PATH" ]; then
    echo ""
    read -p "Venv already exists. Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
    else
        echo "Using existing venv"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate venv
source "${VENV_PATH}/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch (CUDA 12.1 version)
echo ""
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install locked requirements
echo ""
echo "Installing locked requirements..."
pip install -r "${BASE_DIR}/requirements-eval.txt"

# Install auto-round from local source
echo ""
echo "Installing auto-round from local source..."
pip install -e "${PROJECT_ROOT}"

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import lm_eval; print(f'lm-eval: {lm_eval.__version__}')"
python -c "import auto_round; print(f'auto-round: installed')"

# Save environment fingerprint
FINGERPRINT_FILE="${BASE_DIR}/environment_fingerprint.txt"
echo ""
echo "Saving environment fingerprint..."
{
    echo "# Environment Fingerprint"
    echo "# Generated: $(date -Iseconds)"
    echo ""
    echo "## System"
    echo "Python: $(python --version 2>&1)"
    echo "Platform: $(uname -a)"
    echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
    echo ""
    echo "## Packages"
    pip freeze | grep -iE "(torch|vllm|lm.eval|transformers|accelerate|datasets|numpy|auto.round)"
    echo ""
    echo "## Full pip freeze"
    pip freeze
} > "$FINGERPRINT_FILE"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Fingerprint:   ${FINGERPRINT_FILE}"
echo ""
echo "Next steps:"
echo "  1. Snapshot datasets: ./scripts/snapshot_datasets.sh"
echo "  2. Verify environment: python scripts/verify_environment.py"
echo "  3. Run evaluation: ./scripts/eval_vllm.sh gpu 5"
echo "============================================"
