# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Load Environment & Install Dependencies

```bash
cd /uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm

# Load miniforge module (CHPC specific)
module load miniforge3/24.9.0

# Activate conda environment
conda activate py311

# Install requirements
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft.git -q
```

### Step 2: Verify Setup

```bash
python test_setup.py
```

This will check:
- ✓ All packages are installed
- ✓ Configuration is valid
- ✓ Directory structure is correct
- ✓ Custom modules can be imported
- ✓ CUDA availability

### Step 3: Get LLaMA Access

1. Request access to LLaMA models on HuggingFace:
   - [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)
   - [LLaMA 3](https://huggingface.co/meta-llama/Llama-3-8b)

2. Login to HuggingFace:
```bash
huggingface-cli login
```

### Step 4: Configure Your Training (Already Done!)

The config is pre-configured with your datasets in `configs/dora_config.yaml`:

```yaml
# Model
model:
  name: "meta-llama/Llama-2-7b-hf"  # or change to Llama-3-8b

# Datasets (pre-configured)
dataset:
  dataset_name_1: "MLBtrio/genz-slang-dataset"
  dataset_name_2: "Pbeau/criminal-code-expert-multiple"
  test_size: 0.15  # 15% held out for testing
```

You can adjust the model or other settings if needed.

### Step 5: Start Training!

```bash
# Make sure environment is loaded
module load miniforge3/24.9.0
conda activate py311

cd training
python train.py --config ../configs/dora_config.yaml
```

**Note**: The system automatically:
- Loads both datasets from HuggingFace
- Creates train/validation/test splits (15% held out for testing)
- Saves checkpoints to `checkpoints/`
- Logs to `logs/`

## 📊 Monitor Training

### TensorBoard
```bash
tensorboard --logdir logs/
```
Then open: http://localhost:6006

### Weights & Biases (Optional)
Add to config:
```yaml
training:
  report_to: ["tensorboard", "wandb"]
```

## 📈 Evaluate with BERTScore

After training, evaluate on the held-out test set:

```bash
# Load environment
module load miniforge3/24.9.0
conda activate py311

# Evaluate with BERTScore
python evaluate_model.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results/

# View results
cat evaluation_results/evaluation_report.txt
```

**Output includes:**
- BERTScore (F1, Precision, Recall)
- Exact match accuracy
- Sample predictions
- Results saved to `evaluation_results/`

## 🧪 Test Your Model

### Option 1: Interactive Testing

```bash
python inference_example.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --interactive

# Then enter text with <mask>:
# > The capital of France is <mask>.
```

### Option 2: Test with New Data

```bash
# From text file
python test_new_data.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --data_source file \
  --file_path my_test_data.txt \
  --output_dir new_data_results/

# From HuggingFace dataset
python test_new_data.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --data_source huggingface \
  --dataset_name "your/dataset" \
  --split test \
  --output_dir new_data_results/

# From command line text
python test_new_data.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --data_source text \
  --text "This is <mask> test." "Another <mask> example." \
  --output_dir new_data_results/
```

## 💡 Quick Tips

### For Limited GPU Memory:

1. **Use 4-bit quantization (QDoRA)**:
```yaml
model:
  use_4bit: true
```

2. **Reduce batch size**:
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

### For Better Performance:

1. **Start with lower rank**:
```yaml
dora:
  r: 8  # or 16
```

2. **Adjust learning rate**:
```yaml
training:
  learning_rate: 5e-5  # Try 5e-5 to 2e-4
```

### Popular Dataset Combinations:

**Option 1: Text Understanding**
```yaml
dataset_name_1: "wikitext"
dataset_config_1: "wikitext-103-raw-v1"
dataset_name_2: "bookcorpus"
```

**Option 2: News & Articles**
```yaml
dataset_name_1: "cc_news"
dataset_name_2: "ag_news"
```

**Option 3: Diverse Web Text**
```yaml
dataset_name_1: "openwebtext"
dataset_name_2: null  # Just one dataset
```

## 🔍 Check Training Progress

```bash
# View logs
tail -f logs/training.log

# Check checkpoints
ls -lh checkpoints/

# View TensorBoard
tensorboard --logdir logs/
```

## ⚠️ Common Issues

**Issue: CUDA Out of Memory**
- Solution: Enable 4-bit quantization or reduce batch size

**Issue: Can't access LLaMA model**
- Solution: Request access on HuggingFace and run `huggingface-cli login`

**Issue: Slow training**
- Solution: Enable fp16/bf16 and check CUDA availability

**Issue: Dataset not found**
- Solution: Check dataset name/config on HuggingFace Datasets Hub

## 📚 Next Steps

1. **Experiment with hyperparameters** - Try different ranks, learning rates
2. **Add more datasets** - Combine domain-specific datasets
3. **Evaluate performance** - Test on downstream tasks
4. **Fine-tune further** - Continue training on your specific data

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Review `logs/training.log` for error messages
- Visit [DoRA GitHub](https://github.com/NVlabs/DoRA) for issues
- Check [PEFT docs](https://huggingface.co/docs/peft) for DoRA details

Good luck with your training! 🚀

