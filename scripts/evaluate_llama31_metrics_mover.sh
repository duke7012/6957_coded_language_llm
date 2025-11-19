#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/eval_llama31_metrics_output-%j.out
#SBATCH -e logs/eval_llama31_metrics_error-%j.err
#SBATCH --job-name=llama31_metrics_mover

set -euo pipefail

WORKDIR=/uufs/chpc.utah.edu/common/home/u1445624/6957_coded_language_llm
cd "$WORKDIR"

mkdir -p logs

module load miniforge3/24.9.0

echo "Activating conda environment..."
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate py311

echo "Running MoverScore evaluations"
#python data/mover_score_llama_dora.py
#python data/mover_score_llama_svf.py
#python data/mover_score_llama_lora.py
python data/mover_score_llama_loradora.py

echo "MoverScore evaluation completed at: $(date)"
