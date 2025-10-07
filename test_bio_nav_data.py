#!/usr/bin/env python3
"""
Simple test script for Bio-Inspired Navigation Data Generation Package

This script provides basic tests to verify the package functionality.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from bio_nav_data import BioNavDataGenerator, Config
        from bio_nav_data.data_generators import TrajectoryDataGenerator, EnergyDataGenerator
        from bio_nav_data.visualizers.plots import plot_trajectory, plot_energy_consumption
        from bio_nav_data.utils.config import Config
        from bio_nav_data.utils.logger import get_logger
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration functionality."""
    print("Testing configuration...")
    
    try:
        from bio_nav_data.utils.config import Config
        
        config = Config()
        
        # Test parameter updates
        config.update_trajectory_params(n_points=200)
        config.update_energy_params(n_trials=150)
        
        # Test validation
        validation = config.validate_config()
        
        if all(validation.values()):
            print("✅ Configuration test passed")
            return True
        else:
            print(f"❌ Configuration validation failed: {validation}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_generation():
    """Test data generation functionality."""
    print("Testing data generation...")
    
    try:
        from bio_nav_data.data_generators import TrajectoryDataGenerator, EnergyDataGenerator
        
        # Test trajectory generation
        traj_gen = TrajectoryDataGenerator(n_points=50)
        trajectory_df = traj_gen.generate()
        
        if len(trajectory_df) == 50:
            print("✅ Trajectory generation successful")
        else:
            print(f"❌ Trajectory generation failed: expected 50, got {len(trajectory_df)}")
            return False
        
        # Test energy generation
        energy_gen = EnergyDataGenerator(n_trials=20)
        energy_df = energy_gen.generate()
        
        if len(energy_df) == 80:  # 4 methods * 20 trials
            print("✅ Energy generation successful")
        else:
            print(f"❌ Energy generation failed: expected 80, got {len(energy_df)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("Testing visualization...")
    
    try:
        from bio_nav_data.visualizers.plots import plot_trajectory, plot_energy_consumption
        from bio_nav_data.data_generators import TrajectoryDataGenerator, EnergyDataGenerator
        
        # Generate test data
        traj_gen = TrajectoryDataGenerator(n_points=20)
        trajectory_df = traj_gen.generate()
        
        energy_gen = EnergyDataGenerator(n_trials=10)
        energy_df = energy_gen.generate()
        
        # Test plotting (without saving)
        fig1 = plot_trajectory(trajectory_df)
        fig2 = plot_energy_consumption(energy_df)
        
        print("✅ Visualization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_main_pipeline():
    """Test the main pipeline functionality."""
    print("Testing main pipeline...")
    
    try:
        from bio_nav_data import BioNavDataGenerator
        
        # Create a temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize generator with temp directory
            generator = BioNavDataGenerator()
            
            # Test data generation
            data = generator.generate_all_data()
            
            if all(key in data for key in ['trajectory', 'energy', 'performance']):
                print("✅ Main pipeline test passed")
                return True
            else:
                print(f"❌ Main pipeline test failed: missing data keys")
                return False
                
    except Exception as e:
        print(f"❌ Main pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("BIO-NAV-DATA PACKAGE TESTS")
    print("="*50)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_generation,
        test_visualization,
        test_main_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 