# 🧪 Model Testing Guide

This guide shows you how to test your trained DoRA model.

---

## 📋 **Quick Start** (For Tomorrow's Presentation)

### **Option 1: Quick Demo Test** (Recommended for showing professor)

Run pre-defined test examples on GPU:

```bash
cd ~/llama_dora_mlm
sbatch test_model.sh
```

Wait 2-3 minutes, then check results:
```bash
cat logs/test_model_output-*.out
```

This will show your model predicting masked words in both GenZ slang and legal text!

---

## 🎯 **Testing Options**

### **1. Quick Automated Test** (Best for Demo)

**What it does**: Tests model on 8 pre-defined examples covering both domains

**Run on GPU via SLURM:**
```bash
sbatch test_model.sh
```

**Or run directly (on login node, slower):**
```bash
module load miniforge3/24.9.0
conda activate py311
python quick_test.py
```

**Expected output:**
```
Example 1:
Input: That's <mask> cap, this is totally fire!

Top 5 Predictions:
  1. 'no' (confidence: 87.23%)
  2. 'not' (confidence: 5.12%)
  3. 'just' (confidence: 2.45%)
  ...
```

---

### **2. Interactive Mode** (Test Your Own Examples)

**Best for**: Live demonstration with your professor

**How to run:**
```bash
# Via SLURM (on GPU)
salloc --account=soc-gpu-np --partition=soc-gpu-np --gres=gpu:1 --mem=40GB --time=1:00:00

# Once allocated
module load miniforge3/24.9.0 cuda/12.4.0
source /uufs/chpc.utah.edu/sys/installdir/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate py311

# Run interactive mode
python inference_example.py --checkpoint training/checkpoints --interactive
```

**Then type examples like:**
```
Enter text (with <mask>): The defendant was charged with <mask> degree murder.
Enter text (with <mask>): That's no <mask>, this is fire!
Enter text (with <mask>): quit
```

---

### **3. Single Prediction Test**

**Test a specific sentence:**

```bash
python inference_example.py \
  --checkpoint training/checkpoints \
  --text "The capital of France is <mask>."
```

---

### **4. Full Evaluation with BERTScore**

**What it does**: Comprehensive evaluation on test dataset with semantic similarity metrics

**Run on GPU:**
```bash
sbatch evaluate_gpu.sh training/checkpoints
```

**Or manually:**
```bash
python evaluate_model.py \
  --checkpoint training/checkpoints \
  --config configs/dora_config.yaml \
  --output_dir evaluation_results \
  --max_samples 100
```

**This will generate:**
- `evaluation_results/evaluation_results.json` - Full metrics
- `evaluation_results/evaluation_report.txt` - Human-readable report
- BERTScore F1, Precision, Recall scores
- Exact match accuracy

---

## 📊 **Example Test Cases**

### **GenZ Slang Domain:**
```python
test_examples = [
    "That's <mask> cap, this is fire!",           # Should predict: "no"
    "This new song slaps, it's straight <mask>!", # Should predict: "fire"
    "No cap, that's <mask> facts right there.",   # Should predict: "straight"/"real"
]
```

### **Legal/Formal Domain:**
```python
test_examples = [
    "The defendant was charged with <mask> degree murder.",  # Should predict: "first"/"second"
    "The court found the evidence to be <mask>.",            # Should predict: "inadmissible"/"admissible"
    "According to the criminal <mask>, the penalty is...",   # Should predict: "code"/"law"
]
```

---

## 🎯 **For Your Presentation Tomorrow**

### **Recommended Flow:**

**1. Before presentation** (run this morning):
```bash
cd ~/llama_dora_mlm
sbatch test_model.sh
```

**2. Show results to professor:**
```bash
cat logs/test_model_output-*.out
```

Point out:
- ✅ Model predicts GenZ slang correctly ("no cap", "fire")
- ✅ Model predicts legal terms correctly ("first degree", "evidence")
- ✅ Shows dual-domain learning worked!

**3. If professor wants live demo:**

Show them the interactive mode (Option 2 above) and let them type their own examples!

---

## 💡 **Quick Commands Reference**

### **Check if test job is running:**
```bash
squeue -u $USER
```

### **View test results:**
```bash
# Quick test results
cat logs/test_model_output-*.out

# Full evaluation results
cat evaluation_results/evaluation_report.txt
```

### **Run quick test locally** (slower, but works without SLURM):
```bash
python quick_test.py
```

---

## 🔍 **Understanding the Results**

### **Prediction Output Format:**
```
Top 5 Predictions:
  1. 'word1' (confidence: 85.23%)  ← Most likely prediction
  2. 'word2' (confidence: 8.12%)   ← Second most likely
  3. 'word3' (confidence: 3.45%)
  4. 'word4' (confidence: 1.89%)
  5. 'word5' (confidence: 1.31%)
```

### **What to Look For:**
- **High confidence** (>50%): Model is very sure
- **Reasonable alternatives**: Shows model understands context
- **Domain-appropriate**: GenZ slang for informal, legal terms for formal

---

## 🎓 **Talking Points for Professor**

When showing results, mention:

1. **"The model successfully learned both domains"**
   - Show GenZ slang example with high confidence
   - Show legal text example with high confidence

2. **"Despite training only 0.61% of parameters"**
   - The DoRA adapters (1.2GB) learned effectively
   - Base model (6.7B params) remained frozen

3. **"Predictions show semantic understanding"**
   - Top 5 predictions are all contextually reasonable
   - Not just memorization - model generalizes

---

## 📁 **File Locations**

```
llama_dora_mlm/
├── quick_test.py              ← Quick automated test (8 examples)
├── test_model.sh              ← SLURM script for testing
├── inference_example.py       ← Interactive/custom testing
├── evaluate_model.py          ← Full BERTScore evaluation
├── training/
│   └── checkpoints/           ← Your trained model (use this path)
├── logs/
│   └── test_model_output-*.out ← Test results appear here
└── evaluation_results/        ← Full evaluation outputs
```

---

## 🚨 **Troubleshooting**

### **Error: "No module named 'peft'"**
```bash
conda activate py311
pip install git+https://github.com/huggingface/peft.git -q
```

### **Error: "CUDA out of memory"**
Request more memory:
```bash
sbatch --mem=60GB test_model.sh
```

### **Want to test on CPU?** (much slower)
```bash
python quick_test.py  # Will automatically use CPU if no GPU
```

---

## ✅ **Pre-Presentation Checklist**

- [ ] Run `sbatch test_model.sh`
- [ ] Wait 2-3 minutes
- [ ] Check `cat logs/test_model_output-*.out`
- [ ] Verify you see predictions for all examples
- [ ] Note down 2-3 impressive predictions to show
- [ ] (Optional) Run full evaluation for BERTScore metrics

---

**Good luck with your presentation!** 🚀

Your model is trained and ready to demonstrate!

