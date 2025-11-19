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
#SBATCH -o logs/slurm_llama31_output-%j.out
#SBATCH -e logs/slurm_llama31_error-%j.err
#SBATCH --job-name=llama31_ft

# Working directory for this project
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

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

echo "PyTorch CUDA Status:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

TRAIN_TYPE=${TRAIN_TYPE:-dora}
EPOCHS=${EPOCHS:-3}
GPU_ID=${GPU_ID:-0}

echo "=========================================="
echo "Starting Training -> type: ${TRAIN_TYPE}, epochs: ${EPOCHS}, gpu: ${GPU_ID}"
echo "=========================================="

if ./scripts/run_llama31_training.sh "${TRAIN_TYPE}" "${EPOCHS}" "${GPU_ID}"; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
else
    STATUS=$?
    echo ""
    echo "=========================================="
    echo "✗ Training failed with error code ${STATUS}"
    echo "=========================================="
    exit "${STATUS}"
fi

echo ""
echo "Job finished at: $(date)"

