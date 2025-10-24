# SLURM Job Submission Guide for LLaMA DoRA MLM

This guide explains how to use the SLURM scripts to run your training, evaluation, and testing on GPU nodes.

## 📋 Available Scripts

### 1. `train_gpu.sh` - Full Training
Runs the complete DoRA training on a GPU node.

**Resources:**
- 1 GPU
- 100GB RAM
- 8 hours max time
- Logs saved to `logs/slurm_output-[JOBID].out`

**Usage:**
```bash
sbatch train_gpu.sh
```

### 2. `evaluate_gpu.sh` - Model Evaluation
Evaluates a trained checkpoint with BERTScore on GPU.

**Resources:**
- 1 GPU
- 50GB RAM
- 2 hours max time
- Logs saved to `logs/eval_output-[JOBID].out`

**Usage:**
```bash
# Use default checkpoint
sbatch evaluate_gpu.sh

# Or specify a checkpoint
sbatch evaluate_gpu.sh checkpoints/checkpoint-3000
```

### 3. `test_gpu.sh` - Environment Testing
Quick test to verify CUDA, PyTorch, and environment setup.

**Resources:**
- 1 GPU
- 20GB RAM
- 30 minutes max time
- Logs saved to `logs/test_output-[JOBID].out`

**Usage:**
```bash
sbatch test_gpu.sh
```

## 🚀 Quick Start

### Step 1: First Time Setup (On Login Node)
```bash
cd /uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm

# Load modules and activate environment
module load miniforge3/24.9.0
conda activate py311

# Install dependencies if not already done
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft.git -q

# Login to HuggingFace (if needed for model access)
huggingface-cli login
```

### Step 2: Test Environment on GPU
```bash
sbatch test_gpu.sh
```

Check the output:
```bash
# Wait a minute for job to complete, then check
cat logs/test_output-*.out
```

### Step 3: Start Training
```bash
sbatch train_gpu.sh
```

### Step 4: Monitor Your Job
```bash
# Check job status
squeue -u $USER

# Watch the training log in real-time
tail -f logs/slurm_output-[JOBID].out

# Or check training progress
tail -f logs/training.log
```

### Step 5: Evaluate After Training
```bash
# Find your checkpoint
ls checkpoints/

# Run evaluation
sbatch evaluate_gpu.sh checkpoints/checkpoint-3000

# Check results
cat evaluation_results/evaluation_report.txt
```

## 📊 Monitoring Jobs

### Check Job Status
```bash
# See all your jobs
squeue -u $USER

# See detailed job info
scontrol show job [JOBID]
```

### View Output Logs
```bash
# Training output
tail -f logs/slurm_output-[JOBID].out

# Evaluation output
tail -f logs/eval_output-[JOBID].out

# Training logs (from the script itself)
tail -f logs/training.log
```

### Cancel a Job
```bash
scancel [JOBID]
```

## 🔧 Customizing Scripts

### Change Time Limit
Edit the script and modify:
```bash
#SBATCH --time=12:00:00  # 12 hours instead of 8
```

### Request More Memory
```bash
#SBATCH --mem=150GB  # 150GB instead of 100GB
```

### Request Specific GPU Type
```bash
#SBATCH --gres=gpu:a100:1  # Request A100 GPU
#SBATCH --gres=gpu:v100:1  # Request V100 GPU
```

### Change Email Notifications
```bash
#SBATCH --mail-user=your_email@utah.edu
#SBATCH --mail-type=BEGIN,FAIL,END  # Get email when job starts too
```

### Use Different Partition
```bash
#SBATCH --account your-account
#SBATCH --partition your-partition
```

## 💡 Tips and Best Practices

### 1. **Start with a Test Job**
Always run `test_gpu.sh` first to verify your environment works on the GPU nodes.

### 2. **Check GPU Compatibility**
The GT 1030 on login nodes won't work, but the cluster GPUs (V100, A100, etc.) will!

### 3. **Use Scratch Space**
The scripts automatically use `/scratch/general/vast/$USER/` for HuggingFace cache - this is faster and has more space.

### 4. **Monitor Resource Usage**
While job is running, you can SSH to the node and run:
```bash
# Find your node
squeue -u $USER

# SSH to it (e.g., if on node012)
ssh node012

# Check GPU usage
nvidia-smi

# Check memory usage
free -h
```

### 5. **Save Checkpoints Regularly**
The training script automatically saves checkpoints. Make sure you have enough space in your home directory or use scratch:
```yaml
# In configs/dora_config.yaml
training:
  save_steps: 500  # Save every 500 steps
  save_total_limit: 3  # Keep only last 3 checkpoints
```

### 6. **Interactive Testing**
For quick interactive testing (not recommended for long training):
```bash
# Request interactive session
salloc --account=soc-gpu-np --partition=soc-gpu-np --gres=gpu:1 --mem=50GB --time=2:00:00

# Once allocated, run your tests
module load miniforge3/24.9.0 cuda/12.4.0
conda activate py311
python test_setup.py

# Exit when done
exit
```

## 🐛 Troubleshooting

### Job Doesn't Start
```bash
# Check queue position
squeue -u $USER

# Check available resources
sinfo -p soc-gpu-np
```

### Out of Memory Error
Reduce batch size in `configs/dora_config.yaml`:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

Or enable 4-bit quantization:
```yaml
model:
  use_4bit: true
```

### Job Times Out
Increase time limit or save checkpoints more frequently to resume:
```bash
#SBATCH --time=24:00:00  # 24 hours

# Resume from checkpoint
# Edit train.py to load from last checkpoint
```

### CUDA Out of Memory
Check which model you're using:
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # 7B model
  # Use smaller model if needed:
  # name: "meta-llama/Llama-3.2-1B"  # 1B model
```

## 📁 Directory Structure

```
llama_dora_mlm/
├── train_gpu.sh          # SLURM script for training
├── evaluate_gpu.sh       # SLURM script for evaluation
├── test_gpu.sh          # SLURM script for testing
├── logs/                # Job output logs
│   ├── slurm_output-*.out
│   ├── eval_output-*.out
│   ├── test_output-*.out
│   └── training.log     # Training logs from script
├── checkpoints/         # Saved model checkpoints
├── evaluation_results/  # Evaluation outputs
└── configs/
    └── dora_config.yaml # Training configuration
```

## 📝 Example Workflow

```bash
# 1. Submit test job
sbatch test_gpu.sh
# Wait 2-3 minutes, then check: cat logs/test_output-*.out

# 2. Submit training job
sbatch train_gpu.sh
# Note the job ID, e.g., 1234567

# 3. Monitor training
tail -f logs/slurm_output-1234567.out

# 4. After training completes (may take hours)
# Submit evaluation
sbatch evaluate_gpu.sh checkpoints/checkpoint-3000

# 5. Check results
cat evaluation_results/evaluation_report.txt
```

## 🔗 Useful Commands

```bash
# View your jobs
squeue -u $USER

# Cancel a job
scancel [JOBID]

# Cancel all your jobs
scancel -u $USER

# See finished jobs (last 24h)
sacct --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# Check account balance
mybalance

# See available partitions
sinfo

# See GPU availability
sinfo -p soc-gpu-np -o "%20P %5D %14F %10m %11l %N"
```

## 📚 Additional Resources

- [CHPC Documentation](https://www.chpc.utah.edu/documentation/)
- [SLURM Quick Start](https://slurm.schedmd.com/quickstart.html)
- [Project README](README.md)
- [Quick Start Guide](QUICKSTART.md)

---

Happy training! 🚀

For questions about SLURM or cluster usage, check the CHPC documentation or contact CHPC support.

