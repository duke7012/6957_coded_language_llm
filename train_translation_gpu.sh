#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u1445624@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o logs/translation_train_output-%j.out
#SBATCH -e logs/translation_train_error-%j.err
#SBATCH --job-name=translation_train

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
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/24.9.0/etc/profile.d/conda.sh
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

echo "PyTorch CUDA Status:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
echo ""

echo "=========================================="
echo "Starting Translation Training"
echo "=========================================="
echo ""
echo "Training GenZ Slang → Formal English translation"
echo "Using MLM checkpoint as starting point"
echo "Dataset: 6,404 training examples from HuggingFace GenZ slang dataset"
echo ""

# Run training
python train_translation.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Job finished at: $(date)"