#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/eval_llama31_metrics_output-%j.out
#SBATCH -e logs/eval_llama31_metrics_error-%j.err
#SBATCH --job-name=llama31_eval

set -euo pipefail

WORKDIR=/uufs/chpc.utah.edu/common/home/u1445624/6957_coded_language_llm
cd "${WORKDIR}"

mkdir -p logs

echo "=========================================="
echo "Loading modules..."
echo "=========================================="
module load miniforge3/24.9.0

echo "Activating conda environment..."
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate py311

export CUDA_VISIBLE_DEVICES=""

echo ""
echo "=========================================="
echo "Running metrics"
echo "=========================================="

python data/bert_score_llama_dora.py
python data/bert_score_llama_svf.py

echo ""
echo "=========================================="
echo "✓ Metrics completed!"
echo "=========================================="

echo ""
echo "Job finished at: $(date)"

