import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

################ CONFIG ################
MODEL_BASE = "CohereForAI/aya-expanse-8b"
LORA_CHECKPOINT = "results_aya8b_translator/results_10/checkpoint-19970"
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


################ STEP 2: SVF TRAINING ################
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

    print("Loading merged model for SVF training...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Create LoRA adapters again — these are what SVF modifies
    # --------------------------------------------------------
    svf_rank = 64

    lora_config = LoraConfig(
        r=svf_rank,
        lora_alpha=svf_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Add SVF scalar parameters per-LoRA-direction
    # --------------------------------------------------------
    print("Injecting SVF parameters...")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            # A and B are in ModuleDict, extract the single layer
            A = next(iter(module.lora_A.values()))
            B = next(iter(module.lora_B.values()))

            # SVF scaling vector: one scalar per rank direction
            svf_scale = torch.nn.Parameter(torch.ones(A.weight.size(0)))
            module.svf_scale = svf_scale

            # Wrap A/B forward to include svf_scale
            original_forward = module.forward

            def svf_forward(m, x, orig_fwd=original_forward):
                # Apply original forward
                out = orig_fwd(x)

                # LoRA output = (B @ A) * scale
                A = next(iter(m.lora_A.values()))
                B = next(iter(m.lora_B.values()))
                scale = m.svf_scale.view(-1, 1)  # (r, 1)

                lora_update = (B.weight @ (A.weight * scale)).to(out.dtype)
                out = out + x @ lora_update.T

                return out

            module.forward = svf_forward.__get__(module, module.__class__)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = load_dataset(
        "csv",
        data_files={"train": "data/finetune-data-full/combined_train_data.csv"}
    )["train"]

    def format_sample(ex):
        system_prompt = "You are a translator..."
        return f"<|SYSTEM|>{system_prompt}<|USER|>{ex['input']}<|ASSISTANT|>{ex['gold']}"

    # --------------------------------------------------------
    # Training args
    # --------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=f"svf_results/epochs_{args.epochs}",
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
