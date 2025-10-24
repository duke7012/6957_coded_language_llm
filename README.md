# LLaMA DoRA Training for Masked Language Modeling

This project implements training of LLaMA models using DoRA (Weight-Decomposed Low-Rank Adaptation) for masked word prediction tasks. The implementation uses HuggingFace's PEFT library with two specific datasets: [genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) and [criminal-code-expert-multiple](https://huggingface.co/datasets/Pbeau/criminal-code-expert-multiple).

## 📁 Project Structure

```
llama_dora_mlm/
├── configs/
│   └── dora_config.yaml          # Main configuration file (pre-configured with your datasets)
├── data/
│   ├── __init__.py
│   └── dataset_loader.py         # Dataset loading with train/val/test splits
├── models/
│   ├── __init__.py
│   └── model_config.py           # Model and DoRA configuration
├── training/
│   ├── __init__.py
│   └── train.py                  # Main training script
├── utils/
│   ├── __init__.py
│   ├── helpers.py                # Helper functions
│   └── evaluation.py             # BERTScore evaluation utilities
├── checkpoints/                  # Model checkpoints (created during training)
├── logs/                         # Training logs (created during training)
├── evaluation_results/           # Evaluation outputs (created after evaluation)
├── requirements.txt              # Python dependencies
├── test_setup.py                 # Verify installation
├── evaluate_model.py             # Evaluate with BERTScore
├── inference_example.py          # Test trained models
├── QUICKSTART.md                 # Quick start guide
└── README.md                     # This file
```

## 🚀 Environment Setup (CHPC Cluster)

### 1. Load Required Modules & Activate Conda Environment

```bash
# Navigate to project directory
cd /uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm

# Load miniforge module
module load miniforge3/24.9.0

# Activate your conda environment
conda activate py311
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install PEFT with DoRA support
pip install git+https://github.com/huggingface/peft.git -q
```

### 3. Verify Setup

```bash
python test_setup.py
```

This will check:
- ✓ All packages are installed
- ✓ Configuration is valid
- ✓ Directory structure is correct
- ✓ Custom modules can be imported
- ✓ CUDA availability

### 4. Login to HuggingFace

4.1. Install the Llama CLI
```bash
pip install llama-stack -U
```
OR

Use -U option to update llama-stack if a previous version is already installed:

```bash
pip install llama-stack -U
```

4.2. Find models list & Select a model

See latest available models by running the following command and determine the model ID you wish to download:

```bash
llama model list --show-all
```

Select a desired model by running:
```bash
llama model download --source meta --model-id  MODEL_ID
```

Then paste you unique URL when the script asks:
- Llama 4 Scout
```bash
https://llama4.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaTlhOW03czdkcm5vNGU1YzB2bTBiOXhiIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWE0LmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzNzA2NzN9fX1dfQ__&Signature=AhpjrtzcU5TKT0YfJU%7EwSeFgSNsEYaaTNQlznElZCQh3AEFHXHNTZvf%7ETLb3-5M%7ExCmSGRrETDT5wSx-EuD2J-Nf2NmHKSgyl0C2hVEGpgg8ZEf%7ET0Kuun3kCLC20GDmlcnnK47h3vBemOutJ4EUKZ-7yXM-xlFWyfvANfAnFELo1bkaEg8x%7Ep0uz1Pt1Lb8iCTq3XYtJmgT1D8i8E%7EtJGYhElrAE660gwOgCqdg1HoR5ZyfZhUBRPi8RlcmlEBxZNwEUee1A5K-3xIIbJWagVr4zYSnsYNPWEPR8bBPa0rpLaWKobmrEkImx%7Ep7qiizk6tdwbjQMinX3hJ9B8suOg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1373960947585242
```
- Llama 4 Maverick
```bash
https://llama4.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ2lpMXlqeG41aW0xcmE4bWU5YWlqeXprIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWE0LmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzNzA2NzN9fX1dfQ__&Signature=TmiKkvQGLcQz0Fyr8Y8aqes4Xf4-TKFF2Cd1GfbihXXyhFly7%7E%7EzuG72MOIjmkXzpx9-Po8gBavw5JcR4DYQxtwhXTLNM6Lgv%7EyDHsIFqqINQisizibTc%7Eukvt33rfXKnw3b2mGPazkKbUvMCArrTTXZjNqG8nixZ9Qi0znq04RU%7Ey5MzlQCn9glbl1nwdlWS3j0r32t3L3fsyOZK2vxMFoLBSkrITIM0B3swCei4FnVEkpuwRp8lpsmr5sCF88LIMrF-fEFbjOVhtNg7oZPuxPy0Lj0YYZPo4RoRlg6cgIHHrjUecrPTZapDr6JgphMnXVIXhzkrPMMaoHDHa--hQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1055089396613973
```

- Llama 3.3: 70B
```bash
https://llama3-3.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiejNzeDVkZmZwMzQwdzdudDBtYTVudjloIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTMubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MTM3MDY3M319fV19&Signature=DUKD05QThD0jx1IxkYNx0nRkHwR7iWyckVyKr00-LQaADTjBWLqyDHcfA-%7EwNx721J8blDu7RKATbffc1-KtltHhmtnrsfHECoEI4zU6NS74X9eQHZ78fT8nUy5qLWi7yGn79mH6LW2GBWLWcQdmxvW%7E7rRAu43t6mSbwyZsN1F6wojKj9SpnLDdHYwbbvJs81vL0Q%7EAu3Xp98IOTcxEJ-4-mvfReqxrgi-AH%7ERaD7b%7EVoG1K2zgig%7EBrFoDVczrXaNBx6jKImiK3fTHTO5Hz7TiYpRR8%7EMnv1l16zbxcA3gHwoKDRfVEkqvJlOzSVl%7EUsI2suD71DNCxJkuwwf8xA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1518111336047890
```

- Llama 3.2: 1B & 3B (Lightweight)
```bash
https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZGRhYWNmZnR4amZtd2k0dzJpaXAwc2VkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MTM3MDY3M319fV19&Signature=TavFtnKzJvTj7UAwXB5t-xRSrKRNZgc3fSbrpyyiArZOoKa29WP5EHIgSR31HA6GlMmpNPS3fsc6MZxcscBgSiue7U2dtIoEIDAr0GpDU-QnMy7Qx9bSYvVPtXmlYsSPjjIIud2W48RcY0e88TAL4oWMyVTo7kRfPec2nKoCx5SfJtIwi7b9YMQ7f-Y-LCwUR8o0uJVS3zSuzqcYQcQrLj2CX7MaEjujKEHkSneFkT18yCzy6joPmL%7E5R6-R0-pnhuDrRy9esnOoqL4ORveRRSSyZLRX0MN%7EgP2dnOayqkWWq7jVkzKRosSKIObvZ8gaFo6FNggiIakxNd7y9RdL1A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=838724419004734
```
*Please save copies of the unique custom URLs provided above, they will remain valid for 48 hours to download each model up to 5 times, and requests can be submitted multiple times. An email with the download instructions will also be sent to the email address you used to request the models.*

## 📊 Datasets

This project is pre-configured with two datasets:

### 1. [GenZ Slang Dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset)
- **Size**: ~1,779 rows
- **Content**: GenZ slang terms, descriptions, examples, and context
- **Fields**: `Slang`, `Description`, `Example`, `Context`
- **Purpose**: Understanding modern internet language patterns

### 2. [Criminal Code Expert Multiple](https://huggingface.co/datasets/Pbeau/criminal-code-expert-multiple)
- **Content**: Legal and criminal code related text
- **Purpose**: Formal legal language understanding

Both datasets are automatically loaded and combined during training. The system automatically:
- Downloads datasets from HuggingFace
- Combines them into a single training corpus
- Creates train/validation/test splits (default: 15% held out for testing)
- Tokenizes and prepares for masked language modeling

## 🎯 Training with DoRA

### Configuration

The configuration is already set up in `configs/dora_config.yaml` with your datasets:

```yaml
# Model
model:
  name: "meta-llama/Llama-2-7b-hf"  # Change to Llama-3-8b if desired
  use_4bit: false  # Set to true for memory-efficient training

# Datasets (pre-configured)
dataset:
  dataset_name_1: "MLBtrio/genz-slang-dataset"
  dataset_name_2: "Pbeau/criminal-code-expert-multiple"
  test_size: 0.15  # 15% of data held out for testing
  mlm_probability: 0.15  # 15% of tokens masked

# DoRA settings
dora:
  r: 16  # Rank
  lora_alpha: 32
  lora_dropout: 0.05

# Training
training:
  learning_rate: 1e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
```

### Start Training

```bash
# Load environment (if not already loaded)
module load miniforge3/24.9.0
conda activate py311

# Navigate to training directory
cd training

# Start training
python train.py --config ../configs/dora_config.yaml
```

**Monitor Training:**
```bash
# View logs
tail -f ../logs/training.log

# TensorBoard (in another terminal)
tensorboard --logdir ../logs/
```

### Training Output

During training, the system will:
1. **Load both datasets** and combine them
2. **Create splits**: 
   - Training set (~70%)
   - Validation set (~15%)
   - Test set (~15% - held out for final evaluation)
3. **Save test dataset info** to `checkpoints/test_dataset_info.json`
4. **Save checkpoints** to `checkpoints/` directory
5. **Log metrics** to TensorBoard and console

## 🧪 Testing with New Data

### Option 1: Use Holdout Test Set (Recommended)

The training automatically holds out 15% of your data for testing. After training, evaluate on this holdout set:

```bash
# Load environment
module load miniforge3/24.9.0
conda activate py311

# Evaluate on holdout test set
python evaluate_model.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results/
```

This will:
- Load the held-out test data (never seen during training)
- Generate predictions for masked tokens
- Compute **BERTScore** (Precision, Recall, F1)
- Compute exact match accuracy
- Save results to `evaluation_results/`

### Option 2: Test with Completely New Data

You can test the model on new data from several sources:

#### A. New HuggingFace Dataset

```python
# test_new_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM
import torch

# Load your trained model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "checkpoints/checkpoint-XXXX")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load new dataset
new_dataset = load_dataset("your/new-dataset", split="test")

# Use the evaluator
from utils.evaluation import MLMEvaluator
evaluator = MLMEvaluator(model, tokenizer)

# Prepare texts (with masks)
test_texts = [example["text"] for example in new_dataset]

# Evaluate
results = evaluator.comprehensive_evaluation(test_texts)
print(f"BERTScore F1: {results['bertscore_f1_mean']:.4f}")
```

#### B. Custom Text File

```python
# test_custom_text.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM
from utils.evaluation import MLMEvaluator, create_masked_samples_from_text

# Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "checkpoints/checkpoint-XXXX")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load your text file
with open("your_test_data.txt", 'r') as f:
    texts = f.readlines()

# Create masked versions
masked_texts = create_masked_samples_from_text(texts, tokenizer)

# Evaluate
evaluator = MLMEvaluator(model, tokenizer)
results = evaluator.comprehensive_evaluation(masked_texts)
print(f"BERTScore F1: {results['bertscore_f1_mean']:.4f}")
```

#### C. Interactive Testing

```bash
# Interactive mode - test on-the-fly
python inference_example.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --interactive

# Then enter text with <mask> tokens:
# > The capital of France is <mask>.
```

### Option 3: Adjust Train/Test Split

To save more data for testing, edit `configs/dora_config.yaml`:

```yaml
dataset:
  test_size: 0.25  # Hold out 25% instead of 15%
```

This will keep 25% of your data untouched during training for final evaluation.

## 📈 Evaluation with BERTScore

### What is BERTScore?

[BERTScore](https://github.com/Tiiiger/bert_score) uses contextual embeddings from BERT to evaluate text generation. Unlike exact match, it measures semantic similarity between predicted and reference texts.

**Advantages for MLM:**
- Captures semantic similarity (not just exact matches)
- Handles paraphrases and synonyms
- More robust for evaluating masked word predictions
- Provides Precision, Recall, and F1 scores

### Running Evaluation

After training, evaluate your model:

```bash
# Load environment
module load miniforge3/24.9.0
conda activate py311

# Run evaluation with BERTScore
python evaluate_model.py \
  --checkpoint checkpoints/checkpoint-XXXX \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results/ \
  --batch_size 8 \
  --max_samples 1000  # Optional: limit evaluation samples
```

### Evaluation Outputs

The evaluation script creates:

#### 1. `evaluation_results/evaluation_results.json`
```json
{
  "bertscore_f1_mean": 0.8542,
  "bertscore_f1_std": 0.0234,
  "bertscore_precision_mean": 0.8621,
  "bertscore_recall_mean": 0.8465,
  "exact_match_accuracy": 0.6234,
  "num_samples": 1000,
  "sample_predictions": [...]
}
```

#### 2. `evaluation_results/evaluation_report.txt`
Human-readable report with:
- Overall BERTScore metrics (F1, Precision, Recall)
- Exact match accuracy
- Sample predictions with scores
- Detailed breakdown

#### 3. Console Output
```
======================================================================
EVALUATION RESULTS
======================================================================
Number of Samples: 1000

BERTScore Metrics:
  F1:        0.8542 ± 0.0234
  Precision: 0.8621 ± 0.0198
  Recall:    0.8465 ± 0.0245

Exact Match Accuracy: 0.6234
  Exact Matches: 623/1000
======================================================================
```

### Interpreting BERTScore

- **F1 Score** (0.0 - 1.0): Balanced measure of precision and recall
  - > 0.90: Excellent semantic similarity
  - 0.80 - 0.90: Good semantic similarity
  - 0.70 - 0.80: Moderate semantic similarity
  - < 0.70: Poor semantic similarity

- **Precision**: How much of the prediction is relevant
- **Recall**: How much of the reference is captured

### Export Scores for Analysis

```bash
# Evaluation outputs are saved to evaluation_results/
# - evaluation_results.json: Machine-readable scores
# - evaluation_report.txt: Human-readable report

# You can also export to CSV:
python -c "
import json
import csv

with open('evaluation_results/evaluation_results.json', 'r') as f:
    results = json.load(f)

with open('evaluation_scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['BERTScore F1', results['bertscore_f1_mean']])
    writer.writerow(['BERTScore Precision', results['bertscore_precision_mean']])
    writer.writerow(['BERTScore Recall', results['bertscore_recall_mean']])
    writer.writerow(['Exact Match Accuracy', results['exact_match_accuracy']])
    
print('Scores exported to evaluation_scores.csv')
"
```

## 🔄 Complete Workflow

### Full Training & Evaluation Pipeline

```bash
# 1. Setup environment
module load miniforge3/24.9.0
conda activate py311
cd /uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm

# 2. Verify setup
python test_setup.py

# 3. Train model (creates train/val/test splits automatically)
cd training
python train.py --config ../configs/dora_config.yaml

# 4. Evaluate on held-out test set with BERTScore
cd ..
python evaluate_model.py \
  --checkpoint checkpoints/checkpoint-3000 \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results/

# 5. Review results
cat evaluation_results/evaluation_report.txt

# 6. Test interactively
python inference_example.py \
  --checkpoint checkpoints/checkpoint-3000 \
  --interactive
```

## 📊 Understanding Data Splits

### Default Split Strategy

When you train, the data is automatically split:

```
Total Data (100%)
├── Training Set (70%)    # Used for training
├── Validation Set (15%)  # Used for hyperparameter tuning
└── Test Set (15%)        # Held out for final evaluation ✓
```

### How Test Data is Created

The `dataset_loader.py` automatically:
1. Loads both datasets from HuggingFace
2. Combines them into one corpus
3. Splits into train/val/test (the test set is NEVER seen during training)
4. Saves test set info to `checkpoints/test_dataset_info.json`

### Accessing Test Data Later

The test data is preserved and can be loaded for evaluation:

```python
# The evaluate_model.py script automatically loads the correct test data
python evaluate_model.py --checkpoint checkpoints/checkpoint-XXXX
```

Or manually:
```python
from datasets import load_dataset

# Load and split exactly as during training
dataset1 = load_dataset("MLBtrio/genz-slang-dataset", split="train")
dataset2 = load_dataset("Pbeau/criminal-code-expert-multiple", split="train")

# Combine and split (using same seed=42)
from datasets import concatenate_datasets
combined = concatenate_datasets([dataset1, dataset2])
split = combined.train_test_split(test_size=0.15, seed=42)
test_data = split["test"]  # This is your held-out test set
```

## 🔧 Advanced Configuration

### For Limited GPU Memory

```yaml
# In configs/dora_config.yaml
model:
  use_4bit: true  # Enable 4-bit quantization (QDoRA)

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  fp16: true
```

### For Better Performance

```yaml
dora:
  r: 8  # Lower rank (DoRA excels at low ranks)

training:
  learning_rate: 5e-5  # Adjust learning rate
  num_train_epochs: 5  # More epochs
```

### Using Different Models

```yaml
model:
  name: "meta-llama/Llama-3-8b"  # Use LLaMA 3
  # or
  name: "meta-llama/Llama-2-13b-hf"  # Larger model
```

## 📚 Key Features

✅ **DoRA Integration**: Uses HuggingFace PEFT with `use_dora=True`  
✅ **Two Pre-configured Datasets**: GenZ slang + Criminal code  
✅ **Automatic Train/Val/Test Splits**: 15% held out for testing  
✅ **BERTScore Evaluation**: Semantic similarity metrics  
✅ **Exact Match Accuracy**: Traditional accuracy metrics  
✅ **Multiple Testing Options**: Holdout set, new data, interactive  
✅ **Comprehensive Reports**: JSON, TXT, and CSV export  
✅ **CHPC Environment Ready**: Module load + conda activation  
✅ **Memory Efficient**: Gradient checkpointing, mixed precision, quantization  

## 🎓 Understanding the Results

### Training Metrics (TensorBoard)
- **Loss**: Should decrease over time
- **Eval Loss**: Validation performance (watch for overfitting)
- **Learning Rate**: Follows cosine schedule

### Evaluation Metrics

1. **BERTScore F1**: Primary metric for semantic similarity
   - Use this to compare different models/checkpoints
   - Higher is better (max 1.0)

2. **Exact Match Accuracy**: Percentage of perfect predictions
   - Useful but overly strict for MLM
   - Lower than BERTScore (synonyms count as wrong)

3. **Sample Predictions**: Review in `evaluation_report.txt`
   - See where model succeeds/fails
   - Identify patterns and improvements

## 🆘 Troubleshooting

### CUDA Out of Memory
```yaml
# Enable quantization
model:
  use_4bit: true

# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### Can't Access LLaMA Model
```bash
# Request access on HuggingFace first
# Then login
huggingface-cli login
```

### Dataset Not Found
```bash
# Verify dataset names at:
# https://huggingface.co/datasets/MLBtrio/genz-slang-dataset
# https://huggingface.co/datasets/Pbeau/criminal-code-expert-multiple
```

### BERTScore Installation Issues
```bash
# Reinstall bert-score
pip uninstall bert-score -y
pip install bert-score>=0.3.13
```

## 🔗 References

- [DoRA Paper (ICML 2024)](https://arxiv.org/abs/2402.09353)
- [DoRA GitHub](https://github.com/NVlabs/DoRA)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [BERTScore GitHub](https://github.com/Tiiiger/bert_score)
- [GenZ Slang Dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset)
- [Criminal Code Dataset](https://huggingface.co/datasets/Pbeau/criminal-code-expert-multiple)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [LLaMA Models](https://huggingface.co/meta-llama)

## 📝 Example Commands Summary

```bash
# Setup
module load miniforge3/24.9.0
conda activate py311
cd /uufs/chpc.utah.edu/common/home/u1445624/llama_dora_mlm

# Train
cd training && python train.py --config ../configs/dora_config.yaml

# Evaluate with BERTScore
python evaluate_model.py --checkpoint checkpoints/checkpoint-3000 --output_dir evaluation_results/

# Test interactively
python inference_example.py --checkpoint checkpoints/checkpoint-3000 --interactive

# View results
cat evaluation_results/evaluation_report.txt
```

## 🎯 Next Steps

1. **Run training** with the pre-configured datasets
2. **Monitor progress** with TensorBoard
3. **Evaluate** on held-out test set with BERTScore
4. **Analyze results** in evaluation_results/
5. **Test interactively** with new examples
6. **Experiment** with different hyperparameters
7. **Try different models** (LLaMA 3, different sizes)

Happy training! 🚀

For questions or issues, check the logs in `logs/training.log` or review the evaluation reports.
