"""
Script to test trained model with new data from various sources.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import PeftModel, AutoModelForCausalLM
from datasets import load_dataset

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.evaluation import MLMEvaluator, create_masked_samples_from_text

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(base_model_name: str, checkpoint_path: str, device: str = "cuda"):
    """Load trained model and tokenizer."""
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def test_from_file(
    model,
    tokenizer,
    file_path: str,
    output_dir: str,
    mask_probability: float = 0.15
):
    """Test model with data from a text file."""
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(texts)} lines")
    
    # Create masked versions
    logger.info("Creating masked samples...")
    masked_texts = create_masked_samples_from_text(texts, tokenizer, mask_probability)
    
    # Evaluate
    evaluator = MLMEvaluator(model, tokenizer)
    logger.info("Running evaluation...")
    results = evaluator.comprehensive_evaluation(masked_texts)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "new_data_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def test_from_huggingface(
    model,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    output_dir: str,
    max_samples: int = None
):
    """Test model with data from HuggingFace dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Extract texts
    test_texts = []
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Find text field
        text = None
        for key in ["text", "content", "sentence", "input", "Slang", "Description", "question"]:
            if key in example and example[key]:
                text = str(example[key])
                break
        
        if text:
            # Create masked version if no mask present
            import numpy as np
            if "<mask>" not in text and (tokenizer.mask_token is None or tokenizer.mask_token not in text):
                words = text.split()
                if len(words) > 1:
                    mask_idx = np.random.randint(0, len(words))
                    words[mask_idx] = tokenizer.mask_token or "<mask>"
                    text = " ".join(words)
            
            test_texts.append(text)
    
    logger.info(f"Prepared {len(test_texts)} test samples")
    
    # Evaluate
    evaluator = MLMEvaluator(model, tokenizer)
    logger.info("Running evaluation...")
    results = evaluator.comprehensive_evaluation(test_texts)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"new_data_results_{dataset_name.replace('/', '_')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def test_from_text_list(
    model,
    tokenizer,
    texts: list,
    output_dir: str
):
    """Test model with a list of texts."""
    logger.info(f"Testing on {len(texts)} samples")
    
    # Evaluate
    evaluator = MLMEvaluator(model, tokenizer)
    results = evaluator.comprehensive_evaluation(texts)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "new_data_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS ON NEW DATA")
    print("=" * 70)
    print(f"Number of Samples: {results['num_samples']}")
    print(f"\nBERTScore Metrics:")
    print(f"  F1:        {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}")
    print(f"  Precision: {results['bertscore_precision_mean']:.4f} ± {results['bertscore_precision_std']:.4f}")
    print(f"  Recall:    {results['bertscore_recall_mean']:.4f} ± {results['bertscore_recall_std']:.4f}")
    print(f"\nExact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"  Exact Matches: {results['exact_matches']}/{results['total_samples']}")
    print("=" * 70)
    
    # Show sample predictions
    if 'sample_predictions' in results and len(results['sample_predictions']) > 0:
        print("\nSample Predictions:")
        print("-" * 70)
        for i, sample in enumerate(results['sample_predictions'][:5], 1):
            print(f"\n{i}. Original:     {sample['original']}")
            print(f"   Prediction:   {sample['prediction']}")
            print(f"   Ground Truth: {sample['ground_truth']}")
            print(f"   BERTScore F1: {sample['bertscore_f1']:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test trained model with new data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["file", "huggingface", "text"],
        required=True,
        help="Source of new data"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to text file (for file source)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="HuggingFace dataset name (for huggingface source)"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="HuggingFace dataset config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (for huggingface source)"
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs='+',
        help="Text samples to test (for text source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="new_data_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to test"
    )
    parser.add_argument(
        "--mask_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens (for file/text source)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.checkpoint, args.device)
    
    # Test based on data source
    if args.data_source == "file":
        if not args.file_path:
            parser.error("--file_path required for file source")
        results = test_from_file(
            model, tokenizer, args.file_path, args.output_dir, args.mask_probability
        )
    
    elif args.data_source == "huggingface":
        if not args.dataset_name:
            parser.error("--dataset_name required for huggingface source")
        results = test_from_huggingface(
            model, tokenizer, args.dataset_name, args.dataset_config,
            args.split, args.output_dir, args.max_samples
        )
    
    elif args.data_source == "text":
        if not args.text:
            parser.error("--text required for text source")
        results = test_from_text_list(
            model, tokenizer, args.text, args.output_dir
        )
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()

