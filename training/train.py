"""
Main training script for LLaMA with DoRA on masked language modeling.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_config import load_model_for_training
from data.dataset_loader import MLMDatasetLoader, DataCollatorForMaskedLM
from utils.helpers import setup_logging, compute_metrics, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_datasets(config: dict, tokenizer, output_dir: str = "./checkpoints"):
    """
    Prepare training and validation datasets.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer
        output_dir: Directory to save dataset info
        
    Returns:
        Tuple of (train_dataset, val_dataset, data_collator)
    """
    dataset_config = config["dataset"]
    
    # Create dataset loader
    loader = MLMDatasetLoader(
        tokenizer=tokenizer,
        max_length=dataset_config["max_length"],
        mlm_probability=dataset_config["mlm_probability"],
        preprocessing_num_workers=dataset_config.get("preprocessing_num_workers", 4)
    )
    
    # Prepare dataset configurations
    dataset_configs = []
    
    # First dataset
    if dataset_config.get("dataset_name_1"):
        dataset_configs.append({
            "name": dataset_config["dataset_name_1"],
            "config": dataset_config.get("dataset_config_1")
        })
    
    # Second dataset
    if dataset_config.get("dataset_name_2"):
        dataset_configs.append({
            "name": dataset_config["dataset_name_2"],
            "config": dataset_config.get("dataset_config_2")
        })
    
    if not dataset_configs:
        raise ValueError("At least one dataset must be specified in config")
    
    # Load and preprocess datasets
    train_dataset, val_dataset, test_dataset = loader.prepare_datasets(
        dataset_configs=dataset_configs,
        train_split=dataset_config["split_train"],
        validation_split=dataset_config["split_validation"],
        test_size=dataset_config.get("test_size", 0.15)
    )
    
    # Save test dataset info for later evaluation
    import json
    test_info_file = os.path.join(output_dir, "test_dataset_info.json")
    with open(test_info_file, 'w') as f:
        json.dump({
            "test_size": len(test_dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset)
        }, f, indent=2)
    logger.info(f"Test dataset info saved to {test_info_file}")
    
    # Create data collator for MLM
    data_collator = DataCollatorForMaskedLM(
        tokenizer=tokenizer,
        mlm_probability=dataset_config["mlm_probability"]
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, data_collator


def create_training_arguments(config: dict) -> TrainingArguments:
    """
    Create training arguments from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingArguments object
    """
    training_config = config["training"]
    logging_config = config.get("logging", {})
    
    # Create output directory
    output_dir = training_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logging directory
    log_dir = logging_config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        logging_dir=log_dir,
        logging_steps=training_config["logging_steps"],
        eval_strategy="steps",  # Updated from evaluation_strategy for newer transformers
        eval_steps=training_config["eval_steps"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        optim=training_config.get("optim", "adamw_torch"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=training_config.get("seed", 42),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", ["tensorboard"]),
        run_name=logging_config.get("run_name", "llama_dora_mlm"),
        push_to_hub=False,
        gradient_checkpointing=True,  # Save memory
    )
    
    return training_args


def train(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config.get("logging", {}))
    
    # Save config to output directory
    output_dir = config["training"]["output_dir"]
    save_config(config, os.path.join(output_dir, "config.yaml"))
    
    # Load model and tokenizer with DoRA
    logger.info("=" * 50)
    logger.info("Loading model with DoRA...")
    logger.info("=" * 50)
    model, tokenizer = load_model_for_training(config)
    
    # Prepare datasets
    logger.info("=" * 50)
    logger.info("Preparing datasets...")
    logger.info("=" * 50)
    train_dataset, val_dataset, test_dataset, data_collator = prepare_datasets(config, tokenizer, output_dir)
    
    # Create training arguments
    logger.info("=" * 50)
    logger.info("Setting up training arguments...")
    logger.info("=" * 50)
    training_args = create_training_arguments(config)
    
    # Create Trainer
    logger.info("=" * 50)
    logger.info("Initializing Trainer...")
    logger.info("=" * 50)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Start training
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    
    train_result = trainer.train()
    
    # Save final model
    logger.info("=" * 50)
    logger.info("Saving final model...")
    logger.info("=" * 50)
    trainer.save_model()
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Final evaluation
    logger.info("=" * 50)
    logger.info("Running final evaluation...")
    logger.info("=" * 50)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train LLaMA with DoRA for MLM")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/dora_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Run training
    train(args.config)


if __name__ == "__main__":
    main()

