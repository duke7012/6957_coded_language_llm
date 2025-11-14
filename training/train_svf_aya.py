from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
from trl import SFTTrainer
import argparse
import os

USE_GPU = True
device = "cuda:0" if USE_GPU else "cpu"

# Configuration parameters depending on GPU setup
QUANTIZE_4BIT = True
USE_GRAD_CHECKPOINTING = True
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_SEQ_LENGTH = 512
USE_FLASH_ATTENTION = False
GRAD_ACC_STEPS = 2

MODEL_NAME = "CohereForAI/aya-expanse-8b"


def main():
    parser = argparse.ArgumentParser(description="Train an SVF model for translation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    # --- Load model ---
    quantization_config = None
    if QUANTIZE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_implementation = "flash_attention_2" if USE_FLASH_ATTENTION else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Dataset ---
    train_dataset = load_dataset(
        "csv",
        data_files={"train": "data/finetune-data-full/combined_train_data.csv"}
    )["train"]

    train_dataset = train_dataset.filter(lambda x: x["input"] is not None and x["gold"] is not None)

    def formatting_prompts_func(example):
        system_prompt = (
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
            "You are a translator that converts encoded or foreign text into plain English. "
            "When given input text, translate it accurately to English."
            "<|END_OF_TURN_TOKEN|>"
        )
        text = (
            f"{system_prompt}"
            f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate this text to English: {example['input']}"
            f"<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['gold']}"
        )
        return text

    # --- Training arguments ---
    training_arguments = TrainingArguments(
        output_dir=f"svf_results_aya8b_translator/epochs_{args.epochs}/",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        gradient_checkpointing=USE_GRAD_CHECKPOINTING,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-3,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # --- SVF configuration manually define with LoRA config as base ---
    svf_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # experimental approximation of SVF:
        use_dora=True   
    )

    model = get_peft_model(model, svf_config)
    model.print_trainable_parameters()

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=svf_config,
        args=training_arguments,
        formatting_func=formatting_prompts_func
    )

    # --- Train ---
    trainer.train()

    # Save both adapter and base model safely
    output_dir = os.path.join(os.getcwd(), f"svf_results_aya8b_translator/4d_results{args.epochs}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# ---  Save adapter weights ---
    model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    model.config.use_cache = True
    model.eval()


if __name__ == "__main__":
    main()
