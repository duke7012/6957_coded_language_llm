from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
from datasets import load_dataset
from trl import SFTTrainer
import argparse
import os

USE_GPU = True
if USE_GPU:
    device = "cuda:0"
else:
    device = "cpu"

# you may want to change the following parameters depending on your GPU configuration

# free T4 instance
# QUANTIZE_4BIT = True
# USE_GRAD_CHECKPOINTING = True
# TRAIN_BATCH_SIZE = 2
# TRAIN_MAX_SEQ_LENGTH = 512
# USE_FLASH_ATTENTION = False
# GRAD_ACC_STEPS = 16

# equivalent A100 setting
QUANTIZE_4BIT = True
USE_GRAD_CHECKPOINTING = True
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_SEQ_LENGTH = 512
USE_FLASH_ATTENTION = False
GRAD_ACC_STEPS = 2

MODEL_NAME = "CohereForAI/aya-expanse-8b"

def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model for translation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()
    # Load Model
    quantization_config = None
    if QUANTIZE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_implementation = None
    if USE_FLASH_ATTENTION:
        attn_implementation="flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            )
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def get_message_format(prompts, system_prompt):
        messages = []

        for p in prompts:
            messages.append(
                [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}]
            )

        return messages

    def generate_aya(
            model,
            prompts,
            system_prompt = "You are a translator that converts encoded or foreign text into plain English. When given input text, translate it accurately to English.",
            temperature=0.75,
            top_p=1.0,
            top_k=0,
            max_new_tokens=1024
            ):

        messages = get_message_format(prompts, system_prompt)

        input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
            )
        input_ids = input_ids.to(model.device)
        prompt_padded_len = len(input_ids[0])

        gen_tokens = model.generate(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # get only generated tokens
        gen_tokens = [
            gt[prompt_padded_len:] for gt in gen_tokens
            ]

        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return gen_text
    
    train_dataset = load_dataset(
    "csv",
    data_files={"train": "data/finetune-data-full/combined_train_data.csv"}
)

    train_dataset = train_dataset["train"]
    train_dataset = train_dataset.filter(lambda x: x["input"] is not None and x["gold"] is not None)

    def formatting_prompts_func(example):
        system_prompt = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a translator that converts encoded or foreign text into plain English. When given input text, translate it accurately to English.<|END_OF_TURN_TOKEN|>"
        text = f"{system_prompt}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate this text to English: {example['input']}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['gold']}"
        return text
    
    training_arguments = TrainingArguments(
        output_dir="results_aya8b_translator/results_" + str(args.epochs) + "/",
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

    peft_config = LoraConfig(
        lora_alpha=32,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_arguments,
        formatting_func=formatting_prompts_func
    )

    # Train the model
    trainer.train()

    output_dir = os.getcwd() + "/loro_results_aya8b_translator/4d_results" + str(args.epochs) + "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model to disk
    trainer.model.save_pretrained(save_directory=output_dir)
    model.config.use_cache = True
    model.eval()

if __name__ == "__main__":
    main()