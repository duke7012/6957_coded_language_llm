#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=20GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/test_output-%j.out
#SBATCH -e logs/test_error-%j.err
#SBATCH --job-name=llama_dora_test

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

echo "=========================================="
echo "Testing CUDA and Environment Setup"
echo "=========================================="
echo ""

# Run comprehensive tests
python test_setup.py

# Check test status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Some tests failed"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job finished at: $(date)"

