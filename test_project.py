#!/usr/bin/env python3
"""
Test script for AutoOpticalDiagnostics project.
This script verifies the project structure and basic functionality.
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test if all required files and directories exist."""
    print("Testing project structure...")
    
    # Required directories
    required_dirs = [
        "src",
        "src/data_generation",
        "src/model", 
        "src/training",
        "src/evaluation",
        "data/synthetic/train/images",
        "data/synthetic/train/masks",
        "data/synthetic/val/images", 
        "data/synthetic/val/masks",
        "models/saved_models",
        "models/checkpoints",
        "outputs/evaluation_results",
        "outputs/visualizations",
        "outputs/reports",
        "outputs/logs",
        "outputs/plots",
        "configs"
    ]
    
    # Required files
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/utils.py",
        "src/data_generation/__init__.py",
        "src/data_generation/lsci_simulator.py",
        "src/data_generation/oct_simulator.py", 
        "src/data_generation/noise_models.py",
        "src/data_generation/main_generator.py",
        "src/model/__init__.py",
        "src/model/unet_model.py",
        "src/model/dataset.py",
        "src/model/loss.py",
        "src/training/__init__.py",
        "src/training/train.py",
        "src/evaluation/__init__.py",
        "src/evaluation/evaluate.py",
        "main.py",
        "requirements.txt",
        "README.md",
        "configs/data_config.yaml",
        "configs/training_config.yaml",
        "configs/evaluation_config.yaml",
        "configs/full_config.yaml"
    ]
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    # Report results
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    else:
        print("✅ All required directories exist")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    else:
        print("✅ All required files exist")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_imports():
    """Test if all modules can be imported."""
    print("\nTesting imports...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path("src")))
        
        # Test basic imports
        import config
        print("✅ config module imported")
        
        import utils
        print("✅ utils module imported")
        
        import data_generation
        print("✅ data_generation module imported")
        
        import model
        print("✅ model module imported")
        
        import training
        print("✅ training module imported")
        
        import evaluation
        print("✅ evaluation module imported")
        
        # Test specific classes
        from config import Config, ImagingMode, DefectType
        print("✅ Config classes imported")
        
        from data_generation import DataGenerator
        print("✅ DataGenerator imported")
        
        from model.unet_model import UNet
        print("✅ UNet model imported")
        
        from training.train import Trainer
        print("✅ Trainer imported")
        
        from evaluation.evaluate import Evaluator
        print("✅ Evaluator imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        # Test default configuration
        config = Config()
        print("✅ Default configuration created")
        
        # Test configuration validation
        errors = config.validate()
        if errors:
            print(f"⚠️  Configuration warnings: {errors}")
        else:
            print("✅ Configuration validation passed")
        
        # Test device detection
        device = config.get_device()
        print(f"✅ Device detected: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import setup_logging, get_device_info, set_random_seed
        
        # Test logging setup
        logger = setup_logging()
        print("✅ Logging setup successful")
        
        # Test device info
        device_info = get_device_info()
        print("✅ Device info retrieved")
        
        # Test random seed
        set_random_seed(42)
        print("✅ Random seed set")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility function error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("AutoOpticalDiagnostics Project Test")
    print("=" * 60)
    
    # Run tests
    structure_ok = test_project_structure()
    imports_ok = test_imports()
    config_ok = test_configuration()
    utils_ok = test_utility_functions()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Project Structure", structure_ok),
        ("Module Imports", imports_ok),
        ("Configuration", config_ok),
        ("Utility Functions", utils_ok)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Project is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Generate data: python main.py --mode generate_data")
        print("3. Train model: python main.py --mode train")
        print("4. Evaluate model: python main.py --mode evaluate")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())