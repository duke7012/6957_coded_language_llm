import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

################ CONFIG ################
MODEL_BASE = "CohereForAI/aya-expanse-8b"
LORA_CHECKPOINT = "results/.../checkpoint-2500"
MERGED_MODEL_DIR = "merged_lora_model"

USE_4BIT = True
TRAIN_BATCH_SIZE = 16
GRAD_ACC_STEPS = 2
USE_GRAD_CHECKPOINTING = True


################ STEP 1: MERGE LORA ################
print("Loading base model + LoRA...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, LORA_CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)


################ STEP 2: DORA-LIKE TRAINING ################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    print("Loading merged model for LoRA (DoRA-style) training...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------
    # LoRAConfig tuned for DoRA-like behavior
    # ------------------------------
    lora_config = LoraConfig(
        r=64,  # High rank: captures richer directional info (as in DoRA)
        lora_alpha=128,  # Similar to DoRA’s scaling magnitude
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # Regularization
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------
    #  normalize LoRA weights (approximates DoRA decomposition)
    # ------------------------------
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                weight_norm = torch.norm(module.lora_A.weight, dim=1, keepdim=True)
                module.lora_A.weight.div_(weight_norm + 1e-6)

    # ------------------------------
    # Dataset + formatting
    # ------------------------------
    train_dataset = load_dataset(
        "csv",
        data_files={"train": "data/finetune-data-full/combined_train_data.csv"}
    )["train"]

    def format_sample(ex):
        system_prompt = "You are a translator..."
        return f"<|SYSTEM|>{system_prompt}<|USER|>{ex['input']}<|ASSISTANT|>{ex['gold']}"

    training_args = TrainingArguments(
        output_dir=f"dora_like_results/epochs_{args.epochs}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        gradient_checkpointing=USE_GRAD_CHECKPOINTING,
        learning_rate=1e-3,
        bf16=True,
        save_steps=50,
        logging_steps=10
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        formatting_func=format_sample
    )

    trainer.train()
    trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

