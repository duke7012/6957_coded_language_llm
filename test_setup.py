"""
Quick test script to verify the setup is correct.
"""

import sys
import yaml
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("yaml", "PyYAML"),
    ]
    
    failed = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT FOUND")
            failed.append(package_name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True


def test_config():
    """Test if configuration file is valid."""
    print("\nTesting configuration...")
    
    config_path = Path(__file__).parent / "configs" / "dora_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ["model", "dora", "dataset", "training"]
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing section: {section}")
                return False
            print(f"✓ Section '{section}' found")
        
        print("\n✓ Configuration file is valid!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def test_directory_structure():
    """Test if all required directories exist."""
    print("\nTesting directory structure...")
    
    base_path = Path(__file__).parent
    required_dirs = [
        "configs",
        "data",
        "models",
        "training",
        "utils",
        "checkpoints",
        "logs"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✓ All directories exist!")
        return True
    else:
        print("\n⚠️  Some directories are missing")
        return False


def test_module_imports():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    modules_to_test = [
        ("data.dataset_loader", "Dataset Loader"),
        ("models.model_config", "Model Config"),
        ("utils.helpers", "Helper Utils"),
    ]
    
    failed = []
    for module_path, module_name in modules_to_test:
        try:
            __import__(module_path)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name} - ERROR: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All custom modules can be imported!")
        return True


def test_cuda_availability():
    """Test if CUDA is available."""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA is NOT available - training will use CPU (very slow!)")
            return False
            
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("SETUP VERIFICATION TEST")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Custom Modules", test_module_imports()))
    results.append(("CUDA", test_cuda_availability()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results[:-1])  # Exclude CUDA from critical tests
    
    if all_passed:
        print("\n✓ Setup is complete! You're ready to start training.")
        print("\nNext steps:")
        print("1. Edit configs/dora_config.yaml to customize your training")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Run training: cd training && python train.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before training.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

