"""
Example script for inference with trained DoRA model.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def load_trained_model(base_model_name: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained DoRA model from checkpoint.
    
    Args:
        base_model_name: Name of the base LLaMA model
        checkpoint_path: Path to the DoRA checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    print(f"Loading DoRA checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def predict_masked_words(
    model,
    tokenizer,
    text: str,
    device: str = "cuda",
    top_k: int = 5
):
    """
    Predict masked words in text.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Text with <mask> tokens
        device: Device to run on
        top_k: Number of top predictions to show
        
    Returns:
        Predictions dictionary
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # Find mask token positions
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        print("Warning: No mask token found in tokenizer")
        return None
    
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
    
    if len(mask_positions[0]) == 0:
        print("No <mask> tokens found in input text")
        return None
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get top-k predictions for each mask
    results = []
    for batch_idx, position_idx in zip(mask_positions[0], mask_positions[1]):
        logits = predictions[batch_idx, position_idx, :]
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
        top_k_probs = top_k_probs.cpu().numpy()
        
        results.append({
            "position": position_idx.item(),
            "predictions": list(zip(top_k_tokens, top_k_probs))
        })
    
    return results


def interactive_mode(model, tokenizer, device: str = "cuda"):
    """
    Interactive mode for testing predictions.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device to run on
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter text with <mask> tokens to predict masked words.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")
    
    while True:
        text = input("\nEnter text (with <mask>): ").strip()
        
        if text.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        
        if not text:
            continue
        
        if "<mask>" not in text:
            print("⚠️  No <mask> token found in text. Please include at least one <mask>.")
            continue
        
        print("\nPredicting...")
        results = predict_masked_words(model, tokenizer, text, device)
        
        if results:
            print("\n" + "=" * 70)
            print("PREDICTIONS:")
            print("=" * 70)
            for i, result in enumerate(results, 1):
                print(f"\nMask #{i} (position {result['position']}):")
                for j, (token, prob) in enumerate(result['predictions'], 1):
                    print(f"  {j}. '{token.strip()}' (probability: {prob:.4f})")
            print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inference with trained DoRA model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base LLaMA model name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DoRA checkpoint"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text with <mask> tokens to predict"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(
        args.base_model,
        args.checkpoint,
        args.device
    )
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, args.device)
    elif args.text:
        # Single prediction
        print(f"\nInput: {args.text}")
        results = predict_masked_words(
            model,
            tokenizer,
            args.text,
            args.device,
            args.top_k
        )
        
        if results:
            print("\n" + "=" * 70)
            print("PREDICTIONS:")
            print("=" * 70)
            for i, result in enumerate(results, 1):
                print(f"\nMask #{i} (position {result['position']}):")
                for j, (token, prob) in enumerate(result['predictions'], 1):
                    print(f"  {j}. '{token.strip()}' (probability: {prob:.4f})")
            print("=" * 70)
    else:
        # Demo examples
        demo_texts = [
            "The capital of France is <mask>.",
            "Python is a <mask> programming language.",
            "The <mask> is the largest planet in our solar system.",
        ]
        
        print("\n" + "=" * 70)
        print("DEMO MODE - Running example predictions")
        print("=" * 70)
        
        for text in demo_texts:
            print(f"\n\nInput: {text}")
            results = predict_masked_words(
                model,
                tokenizer,
                text,
                args.device,
                args.top_k
            )
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nTop {args.top_k} predictions:")
                    for j, (token, prob) in enumerate(result['predictions'], 1):
                        print(f"  {j}. '{token.strip()}' (probability: {prob:.4f})")


if __name__ == "__main__":
    main()

