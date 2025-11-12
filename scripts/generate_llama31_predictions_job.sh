#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/generate_llama31_predictions_output-%j.out
#SBATCH -e logs/generate_llama31_predictions_error-%j.err
#SBATCH --job-name=llama31_generate

set -euo pipefail

WORKDIR=/uufs/chpc.utah.edu/common/home/u1445624/6957_coded_language_llm
cd "${WORKDIR}"

mkdir -p logs

echo "=========================================="
echo "Loading modules..."
echo "=========================================="
module load miniforge3/24.9.0
module load cuda/12.4.0

echo "Activating conda environment..."
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate py311

echo "Setting up HuggingFace cache..."
mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

ADAPTER_PATH=${ADAPTER_PATH:-dora_results_llama31_8b/epochs_3/}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
INPUT_DIR=${INPUT_DIR:-data/encoded_limited_lines_aya}
OUTPUT_DIR=${OUTPUT_DIR:-data/llama31_pred_partial_dora}

echo ""
echo "=========================================="
echo "Generating predictions"
echo "  Adapter path : ${ADAPTER_PATH}"
echo "  Base model   : ${BASE_MODEL}"
echo "  Input dir    : ${INPUT_DIR}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python scripts/generate_llama31_predictions.py \
  --adapter-path "${ADAPTER_PATH}" \
  --base-model "${BASE_MODEL}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

STATUS=$?

if [[ ${STATUS} -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "✓ Prediction generation completed!"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "✗ Prediction generation failed with code ${STATUS}"
  echo "=========================================="
  exit "${STATUS}"
fi

echo ""
echo "Job finished at: $(date)"

