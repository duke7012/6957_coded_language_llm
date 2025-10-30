import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import random
import torch

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
print("Loading dataset...")
dataset = load_dataset("MLBtrio/genz-slang-dataset")

# The dataset has a single split called "genz_slang_dataset"
split_name = list(dataset.keys())[0]
df = pd.DataFrame(dataset[split_name])
print(f"Loaded {len(df)} slang entries with columns: {list(df.columns)}")

# Normalize column names to lowercase
df.columns = [c.lower() for c in df.columns]

# Expected columns now: slang, description, example, context
if not all(col in df.columns for col in ["slang", "description"]):
    raise KeyError("Dataset must contain 'slang' and 'description' columns.")

df["source"] = "real"

# -------------------------------
# 2. INITIALIZE TEXT GENERATOR
# -------------------------------
print("Loading Hugging Face model pipeline...")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto" if device == "cuda" else None,
    max_new_tokens=200,
)

print(f"Loaded {MODEL_NAME} on {device}.")

# -------------------------------
# 3. GENERATE SYNTHETIC EXAMPLES
# -------------------------------
new_rows = []

for idx, row in df.iterrows():
    slang = row["slang"]
    meaning = row["description"]

    prompt = (
        f"Generate 3 to 5 short, natural example sentences using the Gen Z slang term '{slang}', "
        f"which means '{meaning}'. Each example should sound authentic and casual. "
        f"Output each example on a new line."
    )

    print(f"\n🔹 Generating examples for slang: '{slang}'")

    try:
        response = generator(prompt, temperature=0.8, num_return_sequences=1)[0]["generated_text"]

        # Extract sentences that include the slang word
        examples = [line.strip("-• \n") for line in response.split("\n") if slang.lower() in line.lower()]
        examples = [ex for ex in examples if 3 < len(ex) < 200]
        examples = examples[: random.randint(3, 5)]

        for ex in examples:
            new_rows.append({
                "slang": slang,
                "description": meaning,
                "example": ex,
                "context": "",
                "source": "synthetic"
            })

    except Exception as e:
        print(f"⚠️ Error generating for '{slang}': {e}")

# -------------------------------
# 4. COMBINE AND SAVE
# -------------------------------
expanded_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
output_file = "expanded_genz_slang.csv"
expanded_df.to_csv(output_file, index=False)

print(f"\n✅ Done! Saved expanded dataset to '{output_file}' with {len(expanded_df)} rows.")
print(f"Original: {len(df)} | Synthetic: {len(new_rows)}")
