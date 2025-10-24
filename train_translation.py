"""
Train LLaMA with DoRA for GenZ slang translation task.
Continues from the MLM checkpoint.
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_translation_data():
    """Load the prepared translation dataset."""
    logger.info("Loading translation training data...")
    
    train_dataset = load_dataset('json', data_files='data/translation/train.json', split='train')
    val_dataset = load_dataset('json', data_files='data/translation/val.json', split='train')
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def format_instruction(example, tokenizer):
    """Format examples as instruction-following prompts."""
    
    prompt = f"""### Instruction:
{example['input']}

### Response:
{example['output']}"""
    
    # Tokenize
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    # Add labels for training
    result["labels"] = result["input_ids"].copy()
    
    return result


def main():
    """Main training function."""
    
    logger.info("="*70)
    logger.info("GENZ SLANG TRANSLATION TRAINING")
    logger.info("="*70)
    
    # Load model from MLM checkpoint
    logger.info("\nLoading base model and MLM checkpoint...")
    base_model_name = "meta-llama/Llama-2-7b-hf"
    mlm_checkpoint = "training/checkpoints"  # Your trained MLM model
    
    # Load tokenizer from MLM checkpoint (has extended vocab with mask token)
    logger.info("Loading tokenizer from MLM checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(mlm_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with matching vocab size
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Resize model embeddings to match tokenizer vocab size
    logger.info(f"Resizing model embeddings to match tokenizer vocab size: {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load the MLM DoRA adapters as starting point
    logger.info("Loading DoRA adapters from MLM checkpoint...")
    model = PeftModel.from_pretrained(base_model, mlm_checkpoint)
    
    # Enable training mode
    model.train()
    model.enable_input_require_grads()
    
    logger.info(f"Model loaded with DoRA adapters")
    model.print_trainable_parameters()
    
    # Load translation data
    train_dataset, val_dataset = load_translation_data()
    
    # Format datasets
    logger.info("\nFormatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset"
    )
    
    val_dataset = val_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=val_dataset.column_names,
        desc="Formatting val dataset"
    )
    
    # Training arguments
    output_dir = "./translation_checkpoints"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,
        bf16=True,  # Use bf16 instead of fp16 for better stability with PEFT
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        run_name="llama_dora_translation",
        push_to_hub=False,
        gradient_checkpointing=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create trainer
    logger.info("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("="*70)
    logger.info("STARTING TRANSLATION TRAINING")
    logger.info("="*70)
    
    train_result = trainer.train()
    
    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Final evaluation
    logger.info("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()

