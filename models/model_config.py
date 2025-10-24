"""
Model configuration and initialization with DoRA.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoRAModelConfig:
    """
    Configures and initializes a LLaMA model with DoRA for masked language modeling.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing model and DoRA settings
        """
        self.config = config
        self.model_config = config["model"]
        self.dora_config = config["dora"]
        
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration for memory-efficient training.
        
        Returns:
            BitsAndBytesConfig object
        """
        use_4bit = self.model_config.get("use_4bit", False)
        use_8bit = self.model_config.get("use_8bit", False)
        
        if use_4bit:
            logger.info("Using 4-bit quantization (QDoRA)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            return None
    
    def load_base_model(self):
        """
        Load the base LLaMA model.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = self.model_config["name"]
        logger.info(f"Loading base model: {model_name}")
        
        # Create quantization config if needed
        quantization_config = self.create_quantization_config()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_config.get("trust_remote_code", True),
            use_fast=True,
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Add mask token for MLM
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})
            logger.info("Added <mask> token to tokenizer")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=self.model_config.get("device_map", "auto"),
            trust_remote_code=self.model_config.get("trust_remote_code", True),
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        # Resize token embeddings if we added new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        # Prepare model for k-bit training if quantization is used
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")
        
        logger.info(f"Base model loaded successfully")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        
        return model, tokenizer
    
    def create_dora_config(self) -> LoraConfig:
        """
        Create DoRA configuration using PEFT.
        
        Returns:
            LoraConfig with DoRA enabled
        """
        logger.info("Creating DoRA configuration")
        
        dora_config = LoraConfig(
            r=self.dora_config["r"],
            lora_alpha=self.dora_config["lora_alpha"],
            target_modules=self.dora_config["target_modules"],
            lora_dropout=self.dora_config["lora_dropout"],
            bias=self.dora_config["bias"],
            task_type=self.dora_config.get("task_type", "CAUSAL_LM"),
            use_dora=self.dora_config.get("use_dora", True),  # This enables DoRA!
        )
        
        logger.info(f"DoRA config: rank={dora_config.r}, alpha={dora_config.lora_alpha}, "
                   f"dropout={dora_config.lora_dropout}")
        logger.info(f"Target modules: {dora_config.target_modules}")
        
        return dora_config
    
    def apply_dora(self, model):
        """
        Apply DoRA to the model using PEFT.
        
        Args:
            model: Base model to apply DoRA to
            
        Returns:
            Model with DoRA applied
        """
        dora_config = self.create_dora_config()
        
        logger.info("Applying DoRA to model...")
        model = get_peft_model(model, dora_config)
        
        # Enable input to require gradients (needed for gradient checkpointing with PEFT)
        model.enable_input_require_grads()
        logger.info("Enabled input gradients for gradient checkpointing compatibility")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"All parameters: {all_params:,}")
        
        model.print_trainable_parameters()
        
        return model
    
    def get_model_and_tokenizer(self):
        """
        Complete setup: load base model, apply DoRA, and return everything.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model and tokenizer
        model, tokenizer = self.load_base_model()
        
        # Apply DoRA
        model = self.apply_dora(model)
        
        return model, tokenizer


def load_model_for_training(config: dict):
    """
    Convenience function to load and configure model for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = DoRAModelConfig(config)
    return model_config.get_model_and_tokenizer()

