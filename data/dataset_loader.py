"""
Dataset loader for HuggingFace datasets with masked language modeling preprocessing.
"""

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLMDatasetLoader:
    """
    Loads and preprocesses datasets from HuggingFace for masked language modeling.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        preprocessing_num_workers: int = 4
    ):
        """
        Args:
            tokenizer: Tokenizer to use for preprocessing
            max_length: Maximum sequence length
            mlm_probability: Probability of masking tokens
            preprocessing_num_workers: Number of workers for preprocessing
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.preprocessing_num_workers = preprocessing_num_workers
        
    def load_datasets(
        self,
        dataset_configs: List[Dict[str, str]],
        split: str = "train"
    ) -> Dataset:
        """
        Load multiple datasets from HuggingFace and concatenate them.
        
        Args:
            dataset_configs: List of dataset configurations
                Each config should have: {"name": str, "config": Optional[str]}
            split: Dataset split to load (train/validation/test)
            
        Returns:
            Concatenated dataset
        """
        datasets = []
        
        for config in dataset_configs:
            dataset_name = config["name"]
            dataset_config = config.get("config", None)
            
            logger.info(f"Loading dataset: {dataset_name} (config: {dataset_config})")
            
            try:
                if dataset_config:
                    dataset = load_dataset(dataset_name, dataset_config, split=split)
                else:
                    dataset = load_dataset(dataset_name, split=split)
                    
                datasets.append(dataset)
                logger.info(f"Successfully loaded {len(dataset)} examples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {e}")
                raise
        
        # Concatenate all datasets
        if len(datasets) > 1:
            combined_dataset = concatenate_datasets(datasets)
            logger.info(f"Combined dataset size: {len(combined_dataset)}")
        else:
            combined_dataset = datasets[0]
            
        return combined_dataset
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Tokenize and prepare texts for masked language modeling.
        
        Args:
            examples: Batch of examples with text fields
            
        Returns:
            Tokenized examples
        """
        # Combine all text fields into a single text
        # Handle different dataset structures
        texts = []
        
        # Get all keys
        keys = list(examples.keys())
        
        # Determine how many examples in this batch
        num_examples = len(examples[keys[0]])
        
        for i in range(num_examples):
            text_parts = []
            for key in keys:
                value = examples[key][i]
                # Only include string values
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            
            # Combine all text parts with space
            combined_text = " ".join(text_parts)
            texts.append(combined_text)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        
        return tokenized
    
    def mask_tokens(
        self,
        inputs: np.ndarray,
        special_tokens_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens and labels for masked language modeling.
        
        Args:
            inputs: Input token IDs
            special_tokens_mask: Mask for special tokens (1 for special, 0 for normal)
            
        Returns:
            Tuple of (masked inputs, labels)
        """
        labels = inputs.copy()
        
        # Create probability matrix
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is not None:
            probability_matrix = np.where(special_tokens_mask, 0.0, probability_matrix)
        
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with [MASK] token
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0,
            high=len(self.tokenizer),
            size=labels.shape,
            dtype=np.int64
        )
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%) keep the masked input tokens unchanged
        
        return inputs, labels
    
    def prepare_datasets(
        self,
        dataset_configs: List[Dict[str, str]],
        train_split: str = "train",
        validation_split: str = "validation",
        test_size: float = 0.15
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and preprocess training, validation, and test datasets.
        
        Args:
            dataset_configs: List of dataset configurations
            train_split: Training split name
            validation_split: Validation split name
            test_size: Size of test set (fraction or number of samples)
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        # Load datasets
        train_dataset = self.load_datasets(dataset_configs, split=train_split)
        
        # Try to load validation dataset
        try:
            val_dataset = self.load_datasets(dataset_configs, split=validation_split)
            # Split validation into val and test
            val_test_split = val_dataset.train_test_split(test_size=0.5, seed=42)
            val_dataset = val_test_split["train"]
            test_dataset = val_test_split["test"]
        except Exception as e:
            logger.warning(f"Could not load validation split, creating from train: {e}")
            # Split train into train, val, and test
            # First split: separate test set
            train_val_split = train_dataset.train_test_split(test_size=test_size, seed=42)
            test_dataset = train_val_split["test"]
            
            # Second split: separate validation from remaining train
            train_val = train_val_split["train"]
            final_split = train_val.train_test_split(test_size=0.1, seed=42)
            train_dataset = final_split["train"]
            val_dataset = final_split["test"]
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Preprocess datasets
        logger.info("Tokenizing training dataset...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        
        logger.info("Tokenizing validation dataset...")
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset",
        )
        
        logger.info("Tokenizing test dataset...")
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test dataset",
        )
        
        return train_dataset, val_dataset, test_dataset


# Data collator for masked language modeling
class DataCollatorForMaskedLM:
    """
    Data collator that dynamically masks tokens for masked language modeling.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
    def __call__(self, features: List[Dict]) -> Dict:
        import torch
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        }
        
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.numpy()
        
        # Mask tokens
        inputs = batch["input_ids"].numpy()
        masked_inputs, labels = self._mask_tokens(inputs, special_tokens_mask)
        
        batch["input_ids"] = torch.tensor(masked_inputs, dtype=torch.long)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch
    
    def _mask_tokens(
        self,
        inputs: np.ndarray,
        special_tokens_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mask tokens for MLM."""
        labels = inputs.copy()
        
        # Create probability matrix
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is not None:
            probability_matrix = np.where(special_tokens_mask, 0.0, probability_matrix)
        
        masked_indices = np.random.binomial(1, probability_matrix).astype(bool)
        labels[~masked_indices] = -100
        
        # 80% mask, 10% random, 10% original
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0,
            high=len(self.tokenizer),
            size=labels.shape,
            dtype=np.int64
        )
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels

