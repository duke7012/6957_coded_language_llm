"""
Helper utilities for training and evaluation.
"""

import os
import yaml
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(logging_config: Dict):
    """
    Setup logging configuration.
    
    Args:
        logging_config: Logging configuration dictionary
    """
    log_dir = logging_config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )


def save_config(config: Dict, path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {path}")


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for MLM.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Get predictions for masked tokens only
    mask = labels != -100
    
    if predictions.ndim == 3:
        # If predictions are logits, get argmax
        predictions = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy on masked tokens
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct_predictions": int(correct),
        "total_predictions": int(total)
    }


def print_training_summary(model, train_dataset, val_dataset, training_args):
    """
    Print a summary of the training setup.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args: Training arguments
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    
    # Model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.config._name_or_path if hasattr(model, 'config') else 'Unknown'}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # Dataset info
    logger.info(f"\nTraining samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    
    # Training config
    logger.info(f"\nEpochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size (per device): {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Weight decay: {training_args.weight_decay}")
    logger.info(f"Warmup steps: {training_args.warmup_steps}")
    logger.info(f"Output directory: {training_args.output_dir}")
    
    logger.info("=" * 70 + "\n")


def estimate_training_time(train_dataset, training_args, samples_per_second=1.0):
    """
    Estimate total training time.
    
    Args:
        train_dataset: Training dataset
        training_args: Training arguments
        samples_per_second: Estimated throughput (samples/second)
        
    Returns:
        Estimated time in hours
    """
    total_samples = len(train_dataset) * training_args.num_train_epochs
    effective_batch_size = (
        training_args.per_device_train_batch_size 
        * training_args.gradient_accumulation_steps
    )
    
    total_steps = total_samples / effective_batch_size
    estimated_seconds = total_steps / samples_per_second
    estimated_hours = estimated_seconds / 3600
    
    return estimated_hours


def create_inference_example(model, tokenizer, text: str, device="cuda"):
    """
    Create an example inference with masked tokens.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text with <mask> tokens
        device: Device to run inference on
        
    Returns:
        Predicted text
    """
    import torch
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    
    # Decode
    predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
    
    return predicted_text

