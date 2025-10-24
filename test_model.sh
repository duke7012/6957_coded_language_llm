#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/test_model_output-%j.out
#SBATCH -e logs/test_model_error-%j.err
#SBATCH --job-name=test_dora_model

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

# Setup HuggingFace cache
echo "Setting up HuggingFace cache..."
mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Running Model Tests"
echo "=========================================="
echo ""

# Run the quick test
python quick_test.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Job finished at: $(date)"

