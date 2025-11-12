#!/usr/bin/env bash

# Run Llama 3.1 finetuning scripts on a specified GPU.
# Usage:
#   ./scripts/run_llama31_training.sh [svf|dora] [epochs] [gpu_id]
#   ./scripts/run_llama31_training.sh svf 5 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_TYPE="${1:-dora}"
EPOCHS="${2:-3}"
GPU_ID="${3:-0}"

if [[ "${TRAIN_TYPE}" != "svf" && "${TRAIN_TYPE}" != "dora" ]]; then
  echo "First argument must be 'svf' or 'dora'."
  exit 1
fi

PYTHON_SCRIPT="training/train_${TRAIN_TYPE}_llama31.py"

if [[ ! -f "${PROJECT_ROOT}/${PYTHON_SCRIPT}" ]]; then
  echo "Python script '${PYTHON_SCRIPT}' not found in project root."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_ROOT}"

python "${PYTHON_SCRIPT}" --epochs "${EPOCHS}"
