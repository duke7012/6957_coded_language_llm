"""
Prepare translation training data using the existing GenZ slang dataset.
Uses MLBtrio/genz-slang-dataset which has Slang, Description, Example, Context fields.
"""

from datasets import load_dataset
import json
import os


def prepare_translation_data():
    """Load and prepare the GenZ slang dataset for translation training."""
    
    print("Loading GenZ slang dataset from HuggingFace...")
    dataset = load_dataset("MLBtrio/genz-slang-dataset", split="train")
    
    print(f"Loaded {len(dataset)} slang terms")
    print(f"\nDataset columns: {dataset.column_names}")
    print(f"\nSample entry:")
    print(dataset[0])
    print()
    
    # Create training examples in instruction format
    training_data = []
    
    for item in dataset:
        slang = item.get('Slang', '')
        description = item.get('Description', '')
        example = item.get('Example', '')
        context = item.get('Context', '')
        
        if not slang or not description:
            continue
        
        # Format 1: Direct translation of slang term
        training_data.append({
            "input": f"Translate this GenZ slang to formal English: {slang}",
            "output": description
        })
        
        # Format 2: Explain what it means
        training_data.append({
            "input": f"What does '{slang}' mean?",
            "output": description
        })
        
        # Format 3: Use example if available
        if example:
            training_data.append({
                "input": f"Translate this GenZ slang sentence to formal English: {example}",
                "output": f"The slang '{slang}' means {description}. In this context: {example}"
            })
        
        # Format 4: Context-based if available
        if context:
            training_data.append({
                "input": f"Explain '{slang}' in the context of: {context}",
                "output": description
            })
    
    print(f"Created {len(training_data)} training examples from {len(dataset)} slang terms")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    return train_data, val_data


def save_training_data(train_data, val_data):
    """Save training data to files."""
    
    os.makedirs("data/translation", exist_ok=True)
    
    # Save train
    train_path = "data/translation/train.json"
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"\nTrain data saved to: {train_path}")
    
    # Save validation
    val_path = "data/translation/val.json"
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Validation data saved to: {val_path}")
    
    # Save samples for inspection
    sample_path = "data/translation/samples.txt"
    with open(sample_path, 'w') as f:
        f.write("SAMPLE TRAINING EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        for i, example in enumerate(train_data[:10], 1):
            f.write(f"Example {i}:\n")
            f.write(f"Input:  {example['input']}\n")
            f.write(f"Output: {example['output']}\n")
            f.write("-" * 80 + "\n\n")
    print(f"Sample examples saved to: {sample_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING TRANSLATION TRAINING DATA")
    print("=" * 80)
    print()
    
    train_data, val_data = prepare_translation_data()
    save_training_data(train_data, val_data)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nSample training example:")
    print(f"Input:  {train_data[0]['input']}")
    print(f"Output: {train_data[0]['output']}")
    print()

