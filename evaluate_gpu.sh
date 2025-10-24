#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=50GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/eval_output-%j.out
#SBATCH -e logs/eval_error-%j.err
#SBATCH --job-name=llama_dora_eval

# Set working directory
WORKDIR=/uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm
cd $WORKDIR

# Load required modules
echo "=========================================="
echo "Loading modules..."
echo "=========================================="
module load miniforge3/24.9.0
module load cuda/12.4.0

# Source conda and activate environment
echo "Activating conda environment..."
source /uufs/chpc.utah.edu/sys/installdir/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate py311

# Setup HuggingFace cache on scratch
echo "Setting up HuggingFace cache..."
mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

# Display system information
echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Check GPU availability
echo "GPU Information:"
nvidia-smi
echo ""

# Set checkpoint path (modify this to your latest checkpoint)
# You can pass this as an argument: sbatch evaluate_gpu.sh checkpoint-3000
CHECKPOINT=${1:-checkpoints/checkpoint-latest}

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Using checkpoint: $CHECKPOINT"
echo ""

# Run evaluation with BERTScore
python evaluate_model.py \
  --checkpoint $CHECKPOINT \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results/ \
  --batch_size 8 \
  --max_samples 1000

# Check evaluation status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Evaluation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: evaluation_results/"
    cat evaluation_results/evaluation_report.txt
else
    echo ""
    echo "=========================================="
    echo "✗ Evaluation failed with error code $?"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job finished at: $(date)"

