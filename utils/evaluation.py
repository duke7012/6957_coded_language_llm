"""
Evaluation utilities including BERTScore for masked language modeling.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from bert_score import BERTScorer
from transformers import PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLMEvaluator:
    """
    Evaluator for Masked Language Modeling with BERTScore and other metrics.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Trained model
            tokenizer: Tokenizer
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Initialize BERTScorer
        logger.info("Initializing BERTScorer...")
        self.bert_scorer = BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            device=device
        )
        logger.info("BERTScorer initialized")
    
    def predict_masked_tokens(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Predict masked tokens in texts.
        
        Args:
            texts: List of texts with <mask> tokens
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (original_texts, predicted_texts, ground_truth_texts)
        """
        original_texts = []
        predicted_texts = []
        ground_truth_texts = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
            
            # Decode predictions
            for j, (input_ids, pred_ids) in enumerate(zip(inputs["input_ids"], predictions)):
                # Store original with masks
                original_texts.append(batch_texts[j])
                
                # Decode prediction
                predicted_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                predicted_texts.append(predicted_text)
                
                # For ground truth, decode the original input_ids
                ground_truth = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                ground_truth_texts.append(ground_truth)
        
        return original_texts, predicted_texts, ground_truth_texts
    
    def evaluate_with_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions using BERTScore.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BERTScore metrics
        """
        logger.info("Computing BERTScore...")
        
        # Compute BERTScore
        P, R, F1 = self.bert_scorer.score(predictions, references)
        
        # Convert to numpy
        P = P.cpu().numpy()
        R = R.cpu().numpy()
        F1 = F1.cpu().numpy()
        
        results = {
            "bertscore_precision_mean": float(P.mean()),
            "bertscore_precision_std": float(P.std()),
            "bertscore_recall_mean": float(R.mean()),
            "bertscore_recall_std": float(R.std()),
            "bertscore_f1_mean": float(F1.mean()),
            "bertscore_f1_std": float(F1.std()),
        }
        
        logger.info(f"BERTScore F1: {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}")
        
        return results, P, R, F1
    
    def evaluate_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate exact match accuracy.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with accuracy metrics
        """
        exact_matches = sum(
            pred.strip().lower() == ref.strip().lower()
            for pred, ref in zip(predictions, references)
        )
        
        total = len(predictions)
        accuracy = exact_matches / total if total > 0 else 0.0
        
        return {
            "exact_match_accuracy": accuracy,
            "exact_matches": exact_matches,
            "total_samples": total
        }
    
    def comprehensive_evaluation(
        self,
        test_texts: List[str],
        batch_size: int = 8
    ) -> Dict:
        """
        Perform comprehensive evaluation including BERTScore and accuracy.
        
        Args:
            test_texts: List of texts with masked tokens
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Get predictions
        original_texts, predicted_texts, ground_truth_texts = self.predict_masked_tokens(
            test_texts,
            batch_size=batch_size
        )
        
        # Compute BERTScore
        bert_scores, P, R, F1 = self.evaluate_with_bertscore(
            predicted_texts,
            ground_truth_texts
        )
        
        # Compute accuracy
        accuracy_scores = self.evaluate_accuracy(
            predicted_texts,
            ground_truth_texts
        )
        
        # Combine results
        results = {
            **bert_scores,
            **accuracy_scores,
            "num_samples": len(test_texts)
        }
        
        # Store individual scores for detailed analysis
        results["individual_scores"] = {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist()
        }
        
        # Store sample predictions for inspection
        results["sample_predictions"] = [
            {
                "original": orig,
                "prediction": pred,
                "ground_truth": gt,
                "bertscore_f1": float(f1)
            }
            for orig, pred, gt, f1 in zip(
                original_texts[:10],  # First 10 samples
                predicted_texts[:10],
                ground_truth_texts[:10],
                F1[:10]
            )
        ]
        
        return results


def evaluate_model_on_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    output_dir: str,
    batch_size: int = 8,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate a trained model on a dataset and save results.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Test dataset
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None = all)
        
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = MLMEvaluator(model, tokenizer)
    
    # Prepare test texts
    # Extract text from dataset (adapt based on dataset structure)
    test_texts = []
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Try to find text field
        text = None
        for key in ["text", "content", "sentence", "input", "Slang", "question"]:
            if key in example:
                text = example[key]
                break
        
        if text:
            # Add mask token if not present
            if "<mask>" not in text and self.tokenizer.mask_token not in text:
                # Randomly mask a word
                words = text.split()
                if len(words) > 1:
                    mask_idx = np.random.randint(0, len(words))
                    words[mask_idx] = "<mask>"
                    text = " ".join(words)
            
            test_texts.append(text)
    
    logger.info(f"Evaluating on {len(test_texts)} samples...")
    
    # Run evaluation
    results = evaluator.comprehensive_evaluation(test_texts, batch_size=batch_size)
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"BERTScore F1: {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}")
    print(f"BERTScore Precision: {results['bertscore_precision_mean']:.4f} ± {results['bertscore_precision_std']:.4f}")
    print(f"BERTScore Recall: {results['bertscore_recall_mean']:.4f} ± {results['bertscore_recall_std']:.4f}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Total Samples: {results['num_samples']}")
    print("=" * 70)
    
    # Save detailed report
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write("EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"BERTScore Metrics:\n")
        f.write(f"  - F1: {results['bertscore_f1_mean']:.4f} ± {results['bertscore_f1_std']:.4f}\n")
        f.write(f"  - Precision: {results['bertscore_precision_mean']:.4f} ± {results['bertscore_precision_std']:.4f}\n")
        f.write(f"  - Recall: {results['bertscore_recall_mean']:.4f} ± {results['bertscore_recall_std']:.4f}\n\n")
        f.write(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}\n")
        f.write(f"Total Samples: {results['num_samples']}\n\n")
        f.write("=" * 70 + "\n\n")
        f.write("Sample Predictions:\n\n")
        for i, sample in enumerate(results['sample_predictions'], 1):
            f.write(f"Sample {i}:\n")
            f.write(f"  Original: {sample['original']}\n")
            f.write(f"  Prediction: {sample['prediction']}\n")
            f.write(f"  Ground Truth: {sample['ground_truth']}\n")
            f.write(f"  BERTScore F1: {sample['bertscore_f1']:.4f}\n\n")
    
    logger.info(f"Detailed report saved to {report_file}")
    
    return results


def create_masked_samples_from_text(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    mask_probability: float = 0.15
) -> List[str]:
    """
    Create masked versions of texts for evaluation.
    
    Args:
        texts: List of texts
        tokenizer: Tokenizer
        mask_probability: Probability of masking each token
        
    Returns:
        List of texts with masked tokens
    """
    masked_texts = []
    
    for text in texts:
        words = text.split()
        if len(words) == 0:
            continue
        
        # Randomly select words to mask
        num_masks = max(1, int(len(words) * mask_probability))
        mask_indices = np.random.choice(len(words), size=num_masks, replace=False)
        
        for idx in mask_indices:
            words[idx] = tokenizer.mask_token or "<mask>"
        
        masked_texts.append(" ".join(words))
    
    return masked_texts

