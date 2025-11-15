import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import torch.nn as nn

############ CONFIG ############
MODEL_BASE = "CohereForAI/aya-expanse-8b"
MERGED_MODEL = "merged_lora_model"

# 4-bit DISABLED because bitsandbytes cannot run on Windows
USE_4BIT = False

TRAIN_BATCH_SIZE = 16
GRAD_ACC_STEPS = 2
USE_GRAD_CHECKPOINTING = True


############ SVF - WRAPPER ############
class SVFLinear(nn.Module):
    """Wraps PEFT LoRA linear layers with SVF scaling."""
    def __init__(self, linear, lora_A, lora_B):
        super().__init__()
        self.linear = linear
        self.lora_A = lora_A
        self.lora_B = lora_B

        # One scalar per-rank direction
        self.svf_scale = nn.Parameter(torch.ones(lora_A.weight.size(0)))

    def forward(self, x):
        out = self.linear(x)

        A = self.lora_A.weight           # (r, in)
        B = self.lora_B.weight           # (out, r)

        # Apply SVF per-rank scaling to A
        A_scaled = A * self.svf_scale[:, None]

        # LoRA update
        lora_update = x @ A_scaled.T @ B.T

        return out + lora_update


############ TRAINING ############
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("Loading merged model (bf16, no quantization)...")

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.bfloat16,   # Runs on GPU fine
        device_map="auto"             # Automatically places layers on GPU
    )

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #### Install new LoRA adapters (SVF modifies these)
    rank = 64
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    #### Apply SVF replacing LoRA Linear layers
    print("Injecting SVF...")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            # Extract LoRA matrices
            A = next(iter(module.lora_A.values()))
            B = next(iter(module.lora_B.values()))

            # Replace only the linear base layer
            if hasattr(module, "base_layer"):
                linear = module.base_layer
                wrapped = SVFLinear(linear, A, B)
                module.base_layer = wrapped

    #### Dataset
    train_dataset = load_dataset(
        "csv",
        data_files={"train": "data/finetune-data-full/combined_train_data.csv"}
    )["train"]

    def format_sample(ex):
        system_prompt = "You are a translator..."
        return f"<|SYSTEM|>{system_prompt}<|USER|>{ex['input']}<|ASSISTANT|>{ex['gold']}"

    #### Training args
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


