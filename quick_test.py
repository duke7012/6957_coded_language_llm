"""
Quick test script for demonstrating the trained DoRA model.
"""

import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set HuggingFace cache to use the cached models
os.environ['HF_HOME'] = '/scratch/general/vast/u1445624/huggingface_cache'

def load_model(checkpoint_path="training/checkpoints"):
    """Load the trained model from local checkpoint."""
    print("Loading model from checkpoint...")
    print("(This will take about 30 seconds)")
    
    # Load the full model with adapters merged from checkpoint
    # Uses cached base model from HF_HOME
    from peft import AutoPeftModelForCausalLM
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,  # Use cached model only, no downloads
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    print("✓ Model loaded successfully!\n")
    return model, tokenizer


def test_predictions(model, tokenizer, test_examples):
    """Test the model on example texts."""
    
    print("=" * 80)
    print("TESTING TRAINED MODEL - Masked Word Prediction")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, text in enumerate(test_examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {i}:")
        print(f"Input:  {text}")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        
        # Find mask positions
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            print("⚠️  No <mask> found in text")
            continue
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Show top 5 predictions
        for batch_idx, pos_idx in zip(mask_positions[0], mask_positions[1]):
            logits = predictions[batch_idx, pos_idx, :]
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 5)
            
            print(f"\nTop 5 Predictions:")
            for j, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs), 1):
                token = tokenizer.decode([idx]).strip()
                print(f"  {j}. '{token}' (confidence: {prob*100:.2f}%)")
    
    print(f"\n{'=' * 80}")
    print("Test Complete!")
    print("=" * 80)


def main():
    """Main function."""
    
    # Test examples covering both domains (GenZ slang + legal)
    test_examples = [
        # GenZ Slang examples
        "That's <mask> cap, this is totally fire!",
        "This new song slaps, it's straight <mask>!",
        "No cap, that's <mask> facts right there.",
        
        # Legal/Formal examples  
        "The defendant was charged with <mask> degree murder.",
        "The court found the evidence to be <mask>.",
        "According to the criminal <mask>, the penalty is severe.",
        
        # Mixed/General examples
        "The suspect was found <mask> by the jury.",
        "That's <mask> lit, not gonna lie.",
    ]
    
    # Load model
    model, tokenizer = load_model()
    
    # Run tests
    test_predictions(model, tokenizer, test_examples)
    
    print("\n\n💡 TIP: To test your own examples interactively, run:")
    print("   python inference_example.py --checkpoint training/checkpoints --interactive")


if __name__ == "__main__":
    main()

