import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========================
# CONFIGURATION
# ========================

BASE_MODEL = "CohereForAI/aya-expanse-8b"
DORA_ADAPTER_PATH = "./doro_results_aya8b_translator/4d_results_10/checkpoint-2500"
INPUT_FILE = "data/finetune-data-full\combined_test_data.csv"  
TARGET_LANGUAGE = "English" 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"dora_translation_results_{timestamp}.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# ========================
# SETUP LOGGING
# ========================

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ========================
# LOAD MODEL
# ========================

log(f"🚀 Loading base Aya model on {device} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=dtype,
    device_map=None
)

log("🔗 Attaching DoRA adapter...")
model = PeftModel.from_pretrained(base_model, DORA_ADAPTER_PATH)
model.to(device)
model.eval()

log("✅ DoRA model loaded successfully.\n")


# ========================
# LOAD INPUT SENTENCES
# ========================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

log(f"📚 Loaded {len(sentences)} sentences from {INPUT_FILE}\n")


# ========================
# TRANSLATION LOOP
# ========================

log(f"🌍 Starting translation to {TARGET_LANGUAGE}...\n")

for idx, sentence in enumerate(sentences, 1):
    prompt = f"Translate this to {TARGET_LANGUAGE}: {sentence}"
    log("=" * 80)
    log(f"🧠 Sentence {idx}: {sentence}")
    log(f"Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    log(f"📝 Translation {idx}:\n{translation}\n")

log("=" * 80)
log(f"✅ All {len(sentences)} translations completed.")
log(f"📄 Results saved to: {os.path.abspath(LOG_FILE)}")
