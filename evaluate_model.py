"""
Script to evaluate a trained DoRA model using BERTScore and other metrics.
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.evaluation import evaluate_model_on_dataset, MLMEvaluator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(base_model_name: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load trained DoRA model and tokenizer.
    
    Args:
        base_model_name: Name of base model
        checkpoint_path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    logger.info(f"Loading DoRA checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def load_test_dataset(config: dict):
    """
    Load test dataset from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test dataset
    """
    dataset_config = config["dataset"]
    
    # Try to load test split first
    dataset_configs = []
    
    if dataset_config.get("dataset_name_1"):
        dataset_configs.append({
            "name": dataset_config["dataset_name_1"],
            "config": dataset_config.get("dataset_config_1")
        })
    
    if dataset_config.get("dataset_name_2"):
        dataset_configs.append({
            "name": dataset_config["dataset_name_2"],
            "config": dataset_config.get("dataset_config_2")
        })
    
    # Load and concatenate test datasets
    from datasets import concatenate_datasets
    test_datasets = []
    
    for ds_config in dataset_configs:
        dataset_name = ds_config["name"]
        dataset_cfg = ds_config.get("config")
        
        try:
            # Try to load test split
            logger.info(f"Loading test split from {dataset_name}...")
            if dataset_cfg:
                dataset = load_dataset(dataset_name, dataset_cfg, split="test")
            else:
                dataset = load_dataset(dataset_name, split="test")
            test_datasets.append(dataset)
            logger.info(f"Loaded {len(dataset)} test samples from {dataset_name}")
        except:
            try:
                # Try validation split
                logger.info(f"Test split not found, trying validation split from {dataset_name}...")
                if dataset_cfg:
                    dataset = load_dataset(dataset_name, dataset_cfg, split="validation")
                else:
                    dataset = load_dataset(dataset_name, split="validation")
                test_datasets.append(dataset)
                logger.info(f"Loaded {len(dataset)} validation samples from {dataset_name}")
            except:
                # Use a portion of train split
                logger.info(f"No test/validation split found, using portion of train split from {dataset_name}...")
                if dataset_cfg:
                    dataset = load_dataset(dataset_name, dataset_cfg, split="train")
                else:
                    dataset = load_dataset(dataset_name, split="train")
                
                # Take last 15% as test
                test_size = int(len(dataset) * 0.15)
                dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
                test_datasets.append(dataset)
                logger.info(f"Loaded {len(dataset)} test samples from {dataset_name}")
    
    if len(test_datasets) > 1:
        test_dataset = concatenate_datasets(test_datasets)
    else:
        test_dataset = test_datasets[0]
    
    logger.info(f"Total test samples: {len(test_dataset)}")
    return test_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DoRA model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dora_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load model
    base_model_name = config["model"]["name"]
    model, tokenizer = load_trained_model(base_model_name, args.checkpoint, args.device)
    
    # Load test dataset
    test_dataset = load_test_dataset(config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = MLMEvaluator(model, tokenizer, device=args.device)
    
    # Prepare test texts with masks
    logger.info("Preparing test texts...")
    test_texts = []
    
    for i, example in enumerate(test_dataset):
        if args.max_samples and i >= args.max_samples:
            break
        
        # Find text field in the dataset
        text = None
        for key in ["text", "content", "sentence", "input", "Slang", "Description", "question", "answer"]:
            if key in example and example[key]:
                text = str(example[key])
                break
        
        if not text:
            continue
        
        # Add mask token if not present
        import numpy as np
        if "<mask>" not in text and (tokenizer.mask_token is None or tokenizer.mask_token not in text):
            # Randomly mask a word
            words = text.split()
            if len(words) > 1:
                mask_idx = np.random.randint(0, len(words))
                words[mask_idx] = tokenizer.mask_token or "<mask>"
                text = " ".join(words)
        
        test_texts.append(text)
    
    logger.info(f"Prepared {len(test_texts)} test samples")
    
    # Run evaluation
    results = evaluator.comprehensive_evaluation(test_texts, batch_size=args.batch_size)
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of Samples: {results['num_samples']}")
    print(f"\nBERTScore Metrics:")
    print(f"  F1:        {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}")
    print(f"  Precision: {results['bertscore_precision_mean']:.4f} ± {results['bertscore_precision_std']:.4f}")
    print(f"  Recall:    {results['bertscore_recall_mean']:.4f} ± {results['bertscore_recall_std']:.4f}")
    print(f"\nExact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"  Exact Matches: {results['exact_matches']}/{results['total_samples']}")
    print("=" * 70)
    
    # Save detailed report
    report_file = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {base_model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of Samples: {results['num_samples']}\n\n")
        f.write("BERTScore Metrics:\n")
        f.write(f"  F1:        {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}\n")
        f.write(f"  Precision: {results['bertscore_precision_mean']:.4f} ± {results['bertscore_precision_std']:.4f}\n")
        f.write(f"  Recall:    {results['bertscore_recall_mean']:.4f} ± {results['bertscore_recall_std']:.4f}\n\n")
        f.write(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}\n")
        f.write(f"  Exact Matches: {results['exact_matches']}/{results['total_samples']}\n\n")
        f.write("=" * 70 + "\n\n")
        f.write("Sample Predictions:\n\n")
        for i, sample in enumerate(results['sample_predictions'], 1):
            f.write(f"Sample {i}:\n")
            f.write(f"  Original:     {sample['original']}\n")
            f.write(f"  Prediction:   {sample['prediction']}\n")
            f.write(f"  Ground Truth: {sample['ground_truth']}\n")
            f.write(f"  BERTScore F1: {sample['bertscore_f1']:.4f}\n\n")
    
    logger.info(f"Detailed report saved to {report_file}")
    
    print(f"\nFull results saved to: {args.output_dir}")
    print(f"  - JSON: {results_file}")
    print(f"  - Report: {report_file}")


if __name__ == "__main__":
    main()

