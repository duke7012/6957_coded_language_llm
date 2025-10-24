#!/bin/bash
# Install dependencies for llama_dora_mlm

echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="

# Load modules
module load miniforge3/24.9.0
source /uufs/chpc.utah.edu/sys/installdir/miniforge3/24.9.0/etc/profile.d/conda.sh

# Activate environment
conda activate py311

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Installing PEFT with DoRA support..."
pip install git+https://github.com/huggingface/peft.git -q

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import yaml; import torch; import transformers; import peft; import bert_score; print('✓ All packages imported successfully!')"

