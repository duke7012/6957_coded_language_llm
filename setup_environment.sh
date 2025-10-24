#!/bin/bash
# Environment setup script for CHPC cluster

echo "=========================================="
echo "Setting up LLaMA DoRA MLM Environment"
echo "=========================================="

# Load required module
echo "Loading miniforge3 module..."
module load miniforge3/24.9.0

# Activate conda environment
echo "Activating conda environment py311..."
conda activate py311

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "✓ Environment activated successfully"
else
    echo "✗ Failed to activate environment"
    exit 1
fi

# Display Python version
echo ""
echo "Python version:"
python --version

# Display PyTorch version (if installed)
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed yet"

# Display CUDA availability
echo ""
echo "CUDA available:"
python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "Unable to check (PyTorch not installed)"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Install PEFT: pip install git+https://github.com/huggingface/peft.git -q"
echo "  3. Login to HuggingFace: huggingface-cli login"
echo "  4. Start training: cd training && python train.py"
echo ""

