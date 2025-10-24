#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/slurm_output-%j.out
#SBATCH -e logs/slurm_error-%j.err
#SBATCH --job-name=llama_dora_mlm

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

# Setup HuggingFace cache on scratch (faster and more space)
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

# Check CUDA availability in PyTorch
echo "PyTorch CUDA Status:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Display Python and package versions
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

echo "=========================================="
echo "Starting Training"
echo "=========================================="

# Run training
cd training
python train.py --config ../configs/dora_config.yaml

# Check training status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed with error code $?"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Job finished at: $(date)"

