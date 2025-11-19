#!/usr/bin/env bash

# Run Llama 3.1 finetuning scripts on a specified GPU.
# Usage:
#   ./scripts/run_llama31_training.sh [svf|dora|lora] [epochs] [gpu_id]
#   ./scripts/run_llama31_training.sh svf 5 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_TYPE="${1:-dora}"
EPOCHS="${2:-3}"
GPU_ID="${3:-0}"
BASE_ADAPTER_PATH="${BASE_ADAPTER_PATH:-}"
OUTPUT_DIR_ROOT="${OUTPUT_DIR_ROOT:-}"

if [[ "${TRAIN_TYPE}" != "svf" && "${TRAIN_TYPE}" != "dora" && "${TRAIN_TYPE}" != "lora" ]]; then
  echo "First argument must be 'svf', 'dora', or 'lora'."
  exit 1
fi

PYTHON_SCRIPT="training/train_${TRAIN_TYPE}_llama31.py"

if [[ ! -f "${PROJECT_ROOT}/${PYTHON_SCRIPT}" ]]; then
  echo "Python script '${PYTHON_SCRIPT}' not found in project root."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${PROJECT_ROOT}"

CMD=(python "${PYTHON_SCRIPT}" --epochs "${EPOCHS}")
if [[ "${TRAIN_TYPE}" == "dora" ]]; then
  if [[ -n "${BASE_ADAPTER_PATH}" ]]; then
    CMD+=("--base_adapter_path" "${BASE_ADAPTER_PATH}")
  fi
  if [[ -n "${OUTPUT_DIR_ROOT}" ]]; then
    CMD+=("--output_dir_root" "${OUTPUT_DIR_ROOT}")
  fi
fi

"${CMD[@]}"
